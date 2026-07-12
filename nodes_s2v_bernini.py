"""Bernini in-context conditioning (context_latents) support for Wan S2V models.

For Bernini trunks running with the Wan2.2-S2V module stacked on (e.g. via
DiffusionModelLoaderKJ extra_state_dict with
https://huggingface.co/drozbay/Wan2.2-S2V-14B-module): the native S2V forward
ignores the context_latents set by BerniniConditioning while WanModel._forward
has already appended their rope freqs, crashing attention on the length
mismatch. A DIFFUSION_MODEL wrapper patch-embeds the context streams and a dit
block-0 replace patch inserts the tokens after the main video tokens before the
first attention runs; unpatchify() drops them at the output.

Namespaced to coexist with the standalone wan_s2v_module_patch.py single-file
node: different node ID, wrapper key, and transformer_options keys. If both
nodes are chained the block-0 replace slot is exclusive, so tokens are still
inserted exactly once.
"""

import inspect
import logging

import torch

import comfy.ldm.common_dit
from comfy.ldm.wan import model as wan
from comfy.patcher_extension import WrappersMP

_TOKENS = "_wanex_s2v_bernini_tokens"
_MAIN_LEN = "_wanex_s2v_bernini_main_len"
_DISABLED = "_wanex_s2v_bernini_disabled"

_S2V_CONDS = ("audio_embed", "reference_latent", "control_video", "reference_motion")

_native_support = {}


def _has_native_support(cls):
    if cls not in _native_support:
        try:
            _native_support[cls] = "context_latents" in inspect.getsource(cls.forward_orig)
        except (OSError, TypeError):
            _native_support[cls] = False
    return _native_support[cls]


def _drop_reason(model, context_latents):
    cls = type(model)
    if cls is not wan.WanModel_S2V:
        return f"{cls.__name__} is not supported"
    if cls._forward is not wan.WanModel._forward:
        return "WanModel_S2V no longer inherits WanModel._forward"
    ch = model.patch_embedding.weight.shape[1]
    if any(lat.shape[1] != ch for lat in context_latents):
        return f"model expects {ch}-channel latents"
    return None


def _embed_context_latents(executor, x, timestep, context, clip_fea=None, time_dim_concat=None, transformer_options={}, **kwargs):
    model = executor.class_obj
    if transformer_options.get(_DISABLED, False):
        kwargs = {k: v for k, v in kwargs.items() if k not in _S2V_CONDS}
    context_latents = kwargs.get("context_latents", None)
    transformer_options.pop(_TOKENS, None)
    if context_latents is not None and not _has_native_support(type(model)):
        reason = _drop_reason(model, context_latents)
        if reason is not None:
            logging.warning(f"WanEx_S2VBerniniPatch: dropping context_latents: {reason}")
            kwargs = {k: v for k, v in kwargs.items() if k != "context_latents"}
        else:
            p = model.patch_size
            xp = comfy.ldm.common_dit.pad_to_patch_size(x, p)
            t_len = xp.shape[-3]
            if time_dim_concat is not None:
                t_len += comfy.ldm.common_dit.pad_to_patch_size(time_dim_concat, p).shape[-3]
            transformer_options[_MAIN_LEN] = (t_len // p[0]) * (xp.shape[-2] // p[1]) * (xp.shape[-1] // p[2])
            tokens = []
            for lat in context_latents:
                lat = comfy.ldm.common_dit.pad_to_patch_size(lat, p)
                tokens.append(model.patch_embedding(lat.float().to(x.device)).flatten(2).transpose(1, 2))
            transformer_options[_TOKENS] = torch.cat(tokens, dim=1).to(x.dtype)
    return executor(x, timestep, context, clip_fea, time_dim_concat, transformer_options, **kwargs)


def _insert_tokens_block0(args, extra):
    transformer_options = args["transformer_options"]
    tokens = transformer_options.get(_TOKENS, None)
    if tokens is not None:
        n = transformer_options[_MAIN_LEN]
        img = args["img"]
        args = {**args, "img": torch.cat([img[:, :n], tokens.to(img.dtype), img[:, n:]], dim=1)}
    return extra["original_block"](args)


def _scale_s2v_weights(m, strength):
    # These weights sit on additive paths (audio injector residuals, the
    # trainable_cond_mask embedding, the cond_encoder control-video add), so
    # scaling them scales that influence exactly. Reference image/motion tokens
    # enter attention through trunk weights and cannot be scaled this way; they
    # are only removed at strength 0 via the _DISABLED conditioning strip.
    sd = m.model_state_dict("diffusion_model.")
    for k, w in sd.items():
        if (".audio_injector.injector." in k and (k.endswith(".o.weight") or k.endswith(".o.bias"))) \
                or k.endswith(".trainable_cond_mask.weight") \
                or ".cond_encoder." in k:
            m.add_patches({k: (w,)}, 0.0, strength)


class WanEx_S2VBerniniPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "s2v_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                       "tooltip": "Scales the audio, control-video, and token-type-embedding influence. A connected reference image still applies at full strength, even at 0.0 (use disable_s2v to remove it)."}),
            "disable_s2v": ("BOOLEAN", {"default": False,
                                        "tooltip": "Fully disables S2V for this model line: drops all S2V conditioning (audio, reference image, control video, motion) and runs as the plain trunk, e.g. pure Bernini. Overrides s2v_strength."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "WanExperiments"
    DESCRIPTION = "Enables Bernini context_latents conditioning (BerniniConditioning node) on Wan S2V models, e.g. a Bernini trunk with the S2V module stacked on, and controls the strength of the S2V layers per model line."

    def patch(self, model, s2v_strength=1.0, disable_s2v=False):
        m = model.clone()
        m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "wanex_s2v_bernini", _embed_context_latents)
        m.set_model_patch_replace(_insert_tokens_block0, "dit", "double_block", 0)
        if disable_s2v:
            _scale_s2v_weights(m, 0.0)
            m.model_options["transformer_options"][_DISABLED] = True
        elif s2v_strength != 1.0:
            _scale_s2v_weights(m, s2v_strength)
        return (m,)


NODE_CLASS_MAPPINGS = {"WanEx_S2VBerniniPatch": WanEx_S2VBerniniPatch}
NODE_DISPLAY_NAME_MAPPINGS = {"WanEx_S2VBerniniPatch": "WanEx S2VBerniniPatch"}
