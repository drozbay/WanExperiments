import torch
import torch.nn as nn
import types
import logging
import comfy.conds
import comfy.ldm.wan.model
import comfy.model_base
import comfy.supported_models
import comfy.latent_formats

BINDWEAVE_DEBUG = True

log = logging.getLogger("WanExperiments")


class WanBindweaveModel(comfy.ldm.wan.model.WanModel):
    def __init__(self, *args, **kwargs):
        kwargs['model_type'] = 'i2v'

        super().__init__(*args, **kwargs)

        text_dim = self.text_dim  # Should be 4096 for WAN 2.1

        # Create 3-layer MLP: 3584 -> 4096 -> 4096
        self.text_projection = nn.Sequential(
            torch.nn.Linear(3584, text_dim, device=kwargs.get('device'), dtype=kwargs.get('dtype')),
            nn.GELU(approximate='tanh'),
            torch.nn.Linear(text_dim, text_dim, device=kwargs.get('device'), dtype=kwargs.get('dtype'))
        )

        if BINDWEAVE_DEBUG:
            log.info(f"[WanExperiments] Created WanBindweaveModel with text_projection: 3584 -> {text_dim} -> {text_dim}")

    def forward_orig(self, x, t, context, y=None, control=None, transformer_options={}, **kwargs):
        add_text_emb = kwargs.get("add_text_emb", None)

        if add_text_emb is not None:
            if BINDWEAVE_DEBUG:
                log.info(f"[WanExperiments] Processing add_text_emb in forward_orig:")
                log.info(f"  Input shape: {add_text_emb.shape}")
                log.info(f"  Context shape before: {context.shape}")

            projected = self.text_projection(add_text_emb)

            if BINDWEAVE_DEBUG:
                log.info(f"  Projected shape: {projected.shape}")

            context = torch.cat([projected, context], dim=1)

            if BINDWEAVE_DEBUG:
                log.info(f"  Context shape after prepending: {context.shape}")

        return super().forward_orig(x, t, context, y=y, control=control,
                                   transformer_options=transformer_options, **kwargs)


class WAN21_Bindweave(comfy.model_base.WAN21):

    def __init__(self, *args, **kwargs):
        original_model_class = comfy.ldm.wan.model.WanModel
        comfy.ldm.wan.model.WanModel = WanBindweaveModel

        try:
            super().__init__(*args, **kwargs)
        finally:
            comfy.ldm.wan.model.WanModel = original_model_class

        if BINDWEAVE_DEBUG:
            log.info("[WanExperiments] Created WAN21_Bindweave model wrapper")

    def extra_conds(self, **kwargs):
        """handle add_text_emb conditioning."""
        out = super().extra_conds(**kwargs)

        add_text_emb = kwargs.get("add_text_emb", None)
        if add_text_emb is not None:
            out['add_text_emb'] = comfy.conds.CONDRegular(add_text_emb)

            if BINDWEAVE_DEBUG:
                log.info(f"[WanExperiments] Added add_text_emb to conditioning: shape={add_text_emb.shape}")

        return out

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        """
        When c_concat has more frames than x (reference frames prepended),
        we pad x with zeros at the beginning and trim output after forward pass.
        """
        if c_concat is not None and c_concat.shape[2] > x.shape[2]:
            prepended_frames = c_concat.shape[2] - x.shape[2]

            if BINDWEAVE_DEBUG:
                log.info(f"[WanExperiments] Temporal mismatch detected in _apply_model:")
                log.info(f"  c_concat temporal: {c_concat.shape[2]}, x temporal: {x.shape[2]}")
                log.info(f"  Prepending {prepended_frames} zero frames to noise")

            zero_pad = torch.zeros(
                [x.shape[0], x.shape[1], prepended_frames, x.shape[3], x.shape[4]],
                device=x.device,
                dtype=x.dtype
            )
            x = torch.cat([zero_pad, x], dim=2)

            if BINDWEAVE_DEBUG:
                log.info(f"  Padded x shape: {x.shape}")

            # Call parent _apply_model with padded x
            output = super()._apply_model(x, t, c_concat=c_concat, c_crossattn=c_crossattn,
                                         control=control, transformer_options=transformer_options, **kwargs)

            # Trim prepended frames from output
            output = output[:, :, prepended_frames:, :, :]

            if BINDWEAVE_DEBUG:
                log.info(f"  Output shape after trimming: {output.shape}")

            return output
        else:
            return super()._apply_model(x, t, c_concat=c_concat, c_crossattn=c_crossattn,
                                       control=control, transformer_options=transformer_options, **kwargs)


class WAN21_BindweaveConfig(comfy.supported_models_base.BASE):
    """
    This tells ComfyUI how to recognize and load BindWeave checkpoints.
    """

    unet_config = {
        "image_model": "wan2.1",
        "model_type": "bindweave",
        "in_dim": 36,  # I2V variant
    }

    sampling_settings = {
        "shift": 8.0,
    }

    unet_extra_config = {}
    latent_format = comfy.latent_formats.Wan21

    memory_usage_factor = 0.9

    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.memory_usage_factor = self.unet_config.get("dim", 2000) / 2222

    @classmethod
    def matches(cls, unet_config, state_dict=None):
        """Check if this is a BindWeave model"""
        if unet_config.get("image_model") != "wan2.1":
            return False

        if unet_config.get("model_type") != "bindweave":
            return False

        # Optional: Verify text_projection exists in state_dict as confirmation
        if state_dict is not None:
            has_text_proj = any('text_projection' in k for k in state_dict.keys())
            if not has_text_proj:
                if BINDWEAVE_DEBUG:
                    log.warning("[WanExperiments] model_type is 'bindweave' but no text_projection weights found")
                return False

        return True

    def get_model(self, state_dict, prefix="", device=None):
        if BINDWEAVE_DEBUG:
            log.info(f"[WanExperiments] Creating WAN21_Bindweave model (device: {device})")
        out = WAN21_Bindweave(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        import comfy.text_encoders.sd3_clip
        import comfy.text_encoders.wan
        import comfy.supported_models_base

        pref = self.text_encoder_key_prefix[0]
        t5_detect = comfy.text_encoders.sd3_clip.t5_xxl_detect(state_dict, "{}umt5xxl.transformer.".format(pref))
        return comfy.supported_models_base.ClipTarget(
            comfy.text_encoders.wan.WanT5Tokenizer,
            comfy.text_encoders.wan.te(**t5_detect)
        )


class WAN21_HuMo_I2V(comfy.model_base.WAN21_HuMo):
    """
    Add I2V support for HuMo
    """

    def extra_conds(self, **kwargs):
        """Override to respect concat_latent_image instead of always creating zeros."""
        out = super(comfy.model_base.WAN21_HuMo, self).extra_conds(**kwargs)
        noise = kwargs.get("noise", None)
        audio_embed = kwargs.get("audio_embed", None)

        if audio_embed is not None:
            out['audio_embed'] = comfy.conds.CONDRegular(audio_embed)

        concat_latent_image = kwargs.get("concat_latent_image", None)

        if "c_concat" not in out:  # 1.7B model
            reference_latents = kwargs.get("reference_latents", None)
            if reference_latents is not None:
                out['reference_latent'] = comfy.conds.CONDRegular(self.process_latent_in(reference_latents[-1]))
        else:
            # THE FIX: Only create zeros if user didn't provide concat_latent_image
            if concat_latent_image is None:
                noise_shape = list(noise.shape)
                noise_shape[1] += 4
                concat_latent = torch.zeros(noise_shape, device=noise.device, dtype=noise.dtype)
                zero_vae_values_first = torch.tensor([0.8660, -0.4326, -0.0017, -0.4884, -0.5283, 0.9207, -0.9896, 0.4433, -0.5543, -0.0113, 0.5753, -0.6000, -0.8346, -0.3497, -0.1926, -0.6938]).view(1, 16, 1, 1, 1)
                zero_vae_values_second = torch.tensor([1.0869, -1.2370, 0.0206, -0.4357, -0.6411, 2.0307, -1.5972, 1.2659, -0.8595, -0.4654, 0.9638, -1.6330, -1.4310, -0.1098, -0.3856, -1.4583]).view(1, 16, 1, 1, 1)
                zero_vae_values = torch.tensor([0.8642, -1.8583, 0.1577, 0.1350, -0.3641, 2.5863, -1.9670, 1.6065, -1.0475, -0.8678, 1.1734, -1.8138, -1.5933, -0.7721, -0.3289, -1.3745]).view(1, 16, 1, 1, 1)
                concat_latent[:, 4:] = zero_vae_values
                concat_latent[:, 4:, :1] = zero_vae_values_first
                concat_latent[:, 4:, 1:2] = zero_vae_values_second
                out['c_concat'] = comfy.conds.CONDNoiseShape(concat_latent)

            reference_latents = kwargs.get("reference_latents", None)
            if reference_latents is not None:
                ref_latent = self.process_latent_in(reference_latents[-1])
                ref_latent_shape = list(ref_latent.shape)
                ref_latent_shape[1] += 4 + ref_latent_shape[1]
                ref_latent_full = torch.zeros(ref_latent_shape, device=ref_latent.device, dtype=ref_latent.dtype)
                ref_latent_full[:, 20:] = ref_latent
                ref_latent_full[:, 16:20] = 1.0
                out['reference_latent'] = comfy.conds.CONDRegular(ref_latent_full)

        return out


def register_wan_experiments_models():
    """
    This must be called at import time (in __init__.py).
    """
    import comfy.model_detection

    # Add BindWeave detection
    original_detect = comfy.model_detection.detect_unet_config

    def detect_unet_config_with_bindweave(state_dict, key_prefix, metadata=None):
        """Enhanced detection that checks for BindWeave models."""
        dit_config = original_detect(state_dict, key_prefix, metadata)

        if dit_config is not None:
            state_dict_keys = state_dict.keys()
            text_proj_key = '{}text_projection.0.weight'.format(key_prefix)

            if text_proj_key in state_dict_keys:
                dit_config["model_type"] = "bindweave"
                if BINDWEAVE_DEBUG:
                    log.info(f"[WanExperiments] Detected BindWeave model (found {text_proj_key})")

        return dit_config

    comfy.model_detection.detect_unet_config = detect_unet_config_with_bindweave

    # Register BindWeave config
    comfy.supported_models.models.insert(0, WAN21_BindweaveConfig)

    # Patch HuMo to use I2V-enabled version
    for i, model_config in enumerate(comfy.supported_models.models):
        if model_config.__name__ == 'WAN21_HuMo':
            original_get_model = model_config.get_model

            def patched_get_model(self, state_dict, prefix="", device=None):
                return WAN21_HuMo_I2V(self, device=device)

            model_config.get_model = patched_get_model
            log.info("[WanExperiments] Enabled I2V support for HuMo models")
            break

    log.info("[WanExperiments] Registered model enhancements: BindWeave, HuMo I2V")
