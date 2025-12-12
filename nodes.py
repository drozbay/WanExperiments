import nodes
import comfy.utils
import comfy.model_management
import comfy.clip_vision
from comfy_api.latest import io
import node_helpers
import torch

_KORNIA_MODULE = None

class WanEx_I2VCustomEmbeds:
    """Direct control over concat_latent_image and concat_mask for Wan I2V models. Supports custom temporal masks and advanced conditioning workflows."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT", {
                    "tooltip": "Optional CLIP vision conditioning for I2V models that support it."
                }),
                "concat_latent_image": ("LATENT", {
                    "tooltip": "Direct latent tensor input [batch, channels, temporal, height/8, width/8]. "
                            "Channels typically 16 (base I2V) or 32 (dual-channel). "
                            "If not provided, will auto-generate from start_image."
                }),
                "start_image": ("IMAGE", {
                    "tooltip": "Fallback: If concat_latent_image not provided, this image will be encoded to latent. "
                            "Useful for quick testing or when you don't need custom tensors."
                }),
                "concat_mask": ("MASK", {
                    "tooltip": "Direct mask tensor input. Will be reshaped to [1, 4, temporal, height/8, width/8]. "
                            "Values: 0.0 (use provided) to 1.0 (generate new). "
                            "If not provided, will auto-generate binary mask from start_image presence."
                }),
                "concat_mask_index": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Where to insert mask channels. 0 = mask first (default), >0 = insert at that channel index. "
                            "Used for dual-channel I2V variants."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING", "LATENT", "MASK", "MASK")
    RETURN_NAMES = ("positive", "negative", "latent", "debug_info", "concat_latent_preview", "concat_mask_preview", "mask_channels_preview")
    FUNCTION = "encode"
    CATEGORY = "WanExperiments"

    def encode(self, positive, negative, vae, width, height, length, batch_size,
            clip_vision_output=None, concat_latent_image=None,
            start_image=None, concat_mask=None, concat_mask_index=0,
            ):

        # Calculate expected dimensions
        latent_height = height // 8
        latent_width = width // 8
        temporal_latent = ((length - 1) // 4) + 1

        debug_info = []
        processing_mode = "unknown"

        # Process concat_latent_image
        final_concat_latent = None

        if concat_latent_image is not None:
            if start_image is not None:
                debug_info.append("Warning: Using concat_latent_image and ignoring start_image.")

            processing_mode = "direct_latent"
            if isinstance(concat_latent_image, dict) and "samples" in concat_latent_image:
                latent_samples = concat_latent_image["samples"]
            else:
                raise ValueError("concat_latent_image must be a LATENT dict with 'samples' key")

            # Validate shape
            if latent_samples.ndim != 5:
                raise ValueError(
                    f"concat_latent_image must be 5D tensor [batch, channels, temporal, height, width], "
                    f"got shape {latent_samples.shape} with {latent_samples.ndim} dimensions"
                )

            b, c, t, h, w = latent_samples.shape

            if h != latent_height or w != latent_width:
                raise ValueError(
                    f"concat_latent_image spatial dimensions mismatch: "
                    f"Expected {latent_height}x{latent_width} (from {height}x{width} pixels / 8), "
                    f"but got {h}x{w}"
                )

            # Validate temporal dimension (allow some flexibility)
            if t != temporal_latent:
                debug_info.append(
                    f"Warning: concat_latent_image temporal dimension {t} != expected {temporal_latent}. "
                    f"This may be intentional for custom workflows."
                )

            final_concat_latent = latent_samples
            debug_info.append(f"Using direct latent input: shape {latent_samples.shape}")
            debug_info.append(f"Channels: {c} (16=base I2V, 32=dual-channel)")

        elif start_image is not None:
            processing_mode = "auto_from_image"
            start_image_resized = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

            # Create full image tensor filled with 0.5 (neutral gray)
            image = torch.ones(
                (length, height, width, start_image_resized.shape[-1]),
                device=start_image_resized.device,
                dtype=start_image_resized.dtype
            ) * 0.5

            # Copy the actual image frames
            image[:start_image_resized.shape[0]] = start_image_resized

            # Encode to latent (take only RGB channels)
            final_concat_latent = vae.encode(image[:, :, :, :3])

            debug_info.append(f"Auto-encoded from start_image: {start_image_resized.shape[0]} frames")
            debug_info.append(f"Generated latent shape: {final_concat_latent.shape}")

        else:
            # No latent or image provided - create zero latent
            processing_mode = "zero_latent"
            final_concat_latent = torch.zeros(
                [batch_size, 16, temporal_latent, latent_height, latent_width],
                device=comfy.model_management.intermediate_device()
            )
            debug_info.append(f"No input provided - created zero latent: {final_concat_latent.shape}")

        # Process concat_mask
        # Output format: [1, 4, T_latent, H, W] where each of the 4 channels represents one pixel frame per latent frame
        final_concat_mask = None

        if concat_mask is not None:
            original_shape = concat_mask.shape
            debug_info.append(f"Input mask shape: {original_shape}")

            # Normalize to [B, T, H, W] format
            if concat_mask.ndim == 3:
                concat_mask = concat_mask.unsqueeze(0)
                debug_info.append(f"Detected ComfyUI mask format [temporal, h, w] → [1, {original_shape[0]}, {original_shape[1]}, {original_shape[2]}]")
            elif concat_mask.ndim == 4:
                debug_info.append(f"Detected 4D mask format [batch, temporal, h, w]")
            elif concat_mask.ndim == 5:
                concat_mask = torch.mean(concat_mask, dim=1)
                debug_info.append(f"Detected 5D mask format - averaged channels")
            else:
                raise ValueError(
                    f"concat_mask must be 3D, 4D, or 5D tensor, got {concat_mask.ndim}D with shape {concat_mask.shape}"
                )

            b_m, t_m, h_m, w_m = concat_mask.shape

            # Detect if mask is at pixel or latent temporal resolution
            pixel_frames = (temporal_latent - 1) * 4 + 1
            is_pixel_temporal = (t_m == pixel_frames or t_m == length)

            if is_pixel_temporal:
                debug_info.append(f"Detected pixel-space temporal resolution: {t_m} frames → converting to latent-space")

                if h_m != latent_height or w_m != latent_width:
                    concat_mask = comfy.utils.common_upscale(
                        concat_mask.view(-1, h_m, w_m).unsqueeze(1),
                        latent_width, latent_height, "bilinear", "center"
                    ).squeeze(1).view(b_m, t_m, latent_height, latent_width)
                    debug_info.append(f"Resized mask spatially: {h_m}x{w_m} → {latent_height}x{latent_width}")

                # Convert pixel temporal to latent temporal with per-frame channel mapping
                start_mask_repeated = concat_mask[:, 0:1].repeat(1, 4, 1, 1)
                mask_middle = concat_mask[:, 1:]
                concat_mask = torch.cat([start_mask_repeated, mask_middle], dim=1)

                num_groups = concat_mask.shape[1] // 4
                concat_mask = concat_mask[:, :num_groups * 4]
                concat_mask = concat_mask.view(b_m, num_groups, 4, latent_height, latent_width)
                concat_mask = concat_mask.transpose(1, 2)

                final_concat_mask = concat_mask

                debug_info.append(f"Converted pixel-space mask: {t_m} frames → {num_groups} latent frames with 4 channels")
                debug_info.append(f"Each of 4 channels represents one pixel frame per latent frame")

            else:
                debug_info.append(f"Detected latent-space temporal resolution: {t_m} frames")

                if h_m != latent_height or w_m != latent_width:
                    concat_mask = comfy.utils.common_upscale(
                        concat_mask.view(-1, h_m, w_m).unsqueeze(1),
                        latent_width, latent_height, "bilinear", "center"
                    ).squeeze(1).view(b_m, t_m, latent_height, latent_width)
                    debug_info.append(f"Resized mask spatially: {h_m}x{w_m} → {latent_height}x{latent_width}")

                concat_mask = concat_mask.unsqueeze(1).repeat(1, 4, 1, 1, 1)
                final_concat_mask = concat_mask

                debug_info.append(f"Expanded to 4 channels (all channels identical)")

            debug_info.append(f"Final mask shape: {final_concat_mask.shape}")

        elif start_image is not None and processing_mode == "auto_from_image":
            pixel_frames = (temporal_latent - 1) * 4 + 1
            mask = torch.ones(
                (1, pixel_frames, latent_height, latent_width),
                device=start_image.device,
                dtype=start_image.dtype
            )

            frames_with_image = min(start_image.shape[0], pixel_frames)
            mask[:, :frames_with_image] = 0.0

            debug_info.append(f"Auto-generated pixel-space mask: {frames_with_image}/{pixel_frames} frames from image")

            # Convert to latent temporal format: [1, 4, T_latent, H, W]
            start_mask_repeated = mask[:, 0:1].repeat(1, 4, 1, 1)
            mask_middle = mask[:, 1:]
            mask = torch.cat([start_mask_repeated, mask_middle], dim=1)

            num_groups = mask.shape[1] // 4
            mask = mask[:, :num_groups * 4]
            mask = mask.view(1, num_groups, 4, latent_height, latent_width)
            mask = mask.transpose(1, 2)

            final_concat_mask = mask

            debug_info.append(f"Converted to {num_groups} latent frames with 4 channels per frame")

        else:
            # No mask provided - create default mask (all generate)
            # Format: [1, 4, T_latent, H, W]
            final_concat_mask = torch.ones(
                (1, 4, temporal_latent, latent_height, latent_width),
                device=comfy.model_management.intermediate_device()
            )
            debug_info.append(f"No mask provided - created ones mask (all generate): {final_concat_mask.shape}")

        # Apply to conditioning
        conditioning_dict = {"concat_latent_image": final_concat_latent}

        # Handle concat_mask_index for dual-channel variants
        if concat_mask_index != 0:
            conditioning_dict["concat_mask_index"] = concat_mask_index
            debug_info.append(f"Using concat_mask_index: {concat_mask_index}")

        # Add mask to conditioning dict
        conditioning_dict["concat_mask"] = final_concat_mask

        # Apply to positive and negative conditioning
        positive = node_helpers.conditioning_set_values(positive, conditioning_dict)
        negative = node_helpers.conditioning_set_values(negative, conditioning_dict)

        # Add CLIP vision if provided
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})
            debug_info.append("Added CLIP vision output to conditioning")

        # Create output latent
        latent = torch.zeros(
            [batch_size, 16, temporal_latent, latent_height, latent_width],
            device=comfy.model_management.intermediate_device()
        )

        out_latent = {"samples": latent}

        # Compile debug info
        debug_summary = [
            f"=== WanEx I2VCustomEmbeds ===",
            f"Mode: {processing_mode} | Output: {latent.shape} | Concat Latent: {final_concat_latent.shape} | Concat Mask: {final_concat_mask.shape}",
            "",
        ] + debug_info

        debug_text = "\n".join(debug_summary)

        concat_latent_preview = {"samples": final_concat_latent}
        mask_preview = final_concat_mask[0, 0].float()
        mask_channels_preview = final_concat_mask[0].permute(1, 0, 2, 3).reshape(-1, final_concat_mask.shape[3], final_concat_mask.shape[4])

        return (positive, negative, out_latent, debug_text, concat_latent_preview, mask_preview, mask_channels_preview)
    

class WanEx_BindweaveSubjectToVid:
    """BindWeave subject-to-video conditioning with support for up to 4 reference images, CLIP vision, and QwenVL embeddings."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "vae": ("VAE", ),
                    "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                    "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                    "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
        },
        "optional": {
                    "i2v_images": ("IMAGE",),
                    "ref_images": ("IMAGE",),
                    "i2v_masks": ("MASK",),
                    "ref_masks": ("MASK",),
                    "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                    "qwen_conditioning_pos": ("CONDITIONING", {
                        "tooltip": "QwenVL CONDITIONING from native TextEncodeQwenImageEditPlus"
                    }),
                    "qwen_conditioning_neg": ("CONDITIONING", {
                        "tooltip": "QwenVL CONDITIONING for negative"
                    }),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING", "LATENT", "MASK", "MASK")
    RETURN_NAMES = ("positive", "negative", "latent", "debug_info", "concat_latent_preview", "concat_mask_preview", "mask_channels_preview")
    FUNCTION = "encode"
    CATEGORY = "WanExperiments"

    def encode(self, positive, negative, vae, width, height, length, batch_size,
            clip_padding_strategy="zeros", ref_mask_handling="mask_refs",
            i2v_images=None, ref_images=None, i2v_masks=None, ref_masks=None, clip_vision_output=None,
            qwen_conditioning_pos=None, qwen_conditioning_neg=None,
            ):

        # ========== STEP 1: INPUT PROCESSING & VALIDATION ==========
        latent_height = height // 8
        latent_width = width // 8
        temporal_latent = ((length - 1) // 4) + 1

        debug_info = []
        debug_info.append(f"Target dimensions: {width}x{height}, {length} frames -> {temporal_latent} latent frames")
        debug_info.append(f"Latent spatial: {latent_height}x{latent_width}")

        # Validate reference images
        num_references = 0
        if ref_images is not None:
            num_references = ref_images.shape[0]
            if num_references > 4:
                raise ValueError(f"BindWeave supports maximum 4 reference images, got {num_references}")
            debug_info.append(f"Reference images provided: {num_references}")
        else:
            debug_info.append("No reference images provided - will pad with 4 zero tensors")

        # ========== STEP 2: ENCODE REFERENCE IMAGES INDIVIDUALLY ==========
        reference_latents = []

        if ref_images is not None:
            debug_info.append("Encoding each reference image separately as 1-frame video:")
            for i in range(num_references):
                ref_img = ref_images[i:i+1]
                ref_resized = comfy.utils.common_upscale(
                    ref_img.movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)

                ref_latent = vae.encode(ref_resized[:, :, :, :3])
                reference_latents.append(ref_latent)
                debug_info.append(f"  Reference {i}: shape {ref_latent.shape}")

        # ========== STEP 3: PAD TO EXACTLY 4 REFERENCE SLOTS ==========
        num_padding = 4 - num_references
        if num_padding > 0:
            debug_info.append(f"Padding with {num_padding} zero tensors to reach 4 references")
            for i in range(num_padding):
                zero_ref = torch.zeros(
                    [1, 16, 1, latent_height, latent_width],
                    device=comfy.model_management.intermediate_device(),
                    dtype=torch.float32
                )
                reference_latents.append(zero_ref)
                debug_info.append(f"  Padding slot {num_references + i}: zero tensor {zero_ref.shape}")

        # ========== STEP 4: PROCESS START_IMAGE (Following WanEx I2VCustomEmbeds Pattern) ==========
        start_latent = None
        frames_with_start_image = 0

        if i2v_images is not None:
            debug_info.append("Processing start_image:")

            # Resize and pad the image to match length
            start_image_resized = comfy.utils.common_upscale(
                i2v_images[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

            # Create full image tensor filled with 0.5 (neutral gray)
            image = torch.ones(
                (length, height, width, start_image_resized.shape[-1]),
                device=start_image_resized.device,
                dtype=start_image_resized.dtype
            ) * 0.5

            # Copy the actual image frames
            image[:start_image_resized.shape[0]] = start_image_resized

            # Encode to latent (take only RGB channels)
            start_latent = vae.encode(image[:, :, :, :3])

            frames_with_start_image = start_image_resized.shape[0]
            debug_info.append(f"  Encoded {frames_with_start_image} frames from start_image")
            debug_info.append(f"  Start latent shape: {start_latent.shape}")
        else:
            # No start_image, create an image batch of 0.5 gray frames at the decoded length and encode
            start_image_gray = torch.ones(
                (length, height, width, 3),
                device=comfy.model_management.intermediate_device(),
                dtype=torch.float32
            ) * 0.5
            start_latent = vae.encode(start_image_gray)
            debug_info.append(f"No start_image - created gray image: {start_image_gray.shape}")

        # ========== STEP 5: PREPARE REFERENCE LATENTS FOR BATCHING ==========
        debug_info.append("Preparing reference latents for batching:")

        reference_latents_batched = []
        for i, ref_lat in enumerate(reference_latents):
            if ref_lat.shape[0] != batch_size:
                # Repeat to match batch size
                ref_lat = ref_lat.repeat(batch_size, 1, 1, 1, 1)
            reference_latents_batched.append(ref_lat)
            debug_info.append(f"  Reference {i} (batched): {ref_lat.shape}")

        # ========== STEP 6: MASK HANDLING ==========
        # Output format: [1, 4, T_latent, H, W] where each of the 4 channels represents one pixel frame per latent frame
        final_concat_mask = None

        if i2v_masks is not None:
            debug_info.append("Processing provided mask:")
            original_mask_shape = i2v_masks.shape
            debug_info.append(f"  Input mask shape: {original_mask_shape}")

            # Normalize to [B, T, H, W] format
            if i2v_masks.ndim == 3:
                i2v_masks = i2v_masks.unsqueeze(0)
            elif i2v_masks.ndim == 4:
                debug_info.append(f"  Detected 4D mask")
            elif i2v_masks.ndim == 5:
                i2v_masks = torch.mean(i2v_masks, dim=1)
                debug_info.append(f"  Averaged 5D mask channels")
            else:
                raise ValueError(f"i2v_masks must be 3D, 4D, or 5D, got {i2v_masks.ndim}D")

            b_m, t_m, h_m, w_m = i2v_masks.shape

            # Detect if mask is at pixel or latent temporal resolution
            pixel_frames = (temporal_latent - 1) * 4 + 1
            is_pixel_temporal = (t_m == pixel_frames or t_m == length)

            if is_pixel_temporal:
                debug_info.append(f"  Detected pixel-space temporal: {t_m} frames → converting to latent-space")

                if h_m != latent_height or w_m != latent_width:
                    i2v_masks = comfy.utils.common_upscale(
                        i2v_masks.view(-1, h_m, w_m).unsqueeze(1),
                        latent_width, latent_height, "bilinear", "center"
                    ).squeeze(1).view(b_m, t_m, latent_height, latent_width)

                # Convert pixel temporal to latent temporal with per-frame channel mapping
                start_mask_repeated = i2v_masks[:, 0:1].repeat(1, 4, 1, 1)
                mask_middle = i2v_masks[:, 1:]
                i2v_masks = torch.cat([start_mask_repeated, mask_middle], dim=1)

                num_groups = i2v_masks.shape[1] // 4
                i2v_masks = i2v_masks[:, :num_groups * 4]
                i2v_masks = i2v_masks.view(b_m, num_groups, 4, latent_height, latent_width)
                i2v_masks = i2v_masks.transpose(1, 2)

                final_concat_mask = i2v_masks
                debug_info.append(f"  Converted: {t_m} frames → {num_groups} latent frames with 4 channels")

            else:
                debug_info.append(f"  Detected latent-space temporal: {t_m} frames")

                if h_m != latent_height or w_m != latent_width:
                    i2v_masks = comfy.utils.common_upscale(
                        i2v_masks.view(-1, h_m, w_m).unsqueeze(1),
                        latent_width, latent_height, "bilinear", "center"
                    ).squeeze(1).view(b_m, t_m, latent_height, latent_width)

                i2v_masks = i2v_masks.unsqueeze(1).repeat(1, 4, 1, 1, 1)
                final_concat_mask = i2v_masks

            debug_info.append(f"  Final mask shape: {final_concat_mask.shape}")

        elif i2v_images is not None:
            # Auto-generate mask at pixel temporal resolution
            pixel_frames = (temporal_latent - 1) * 4 + 1
            mask = torch.ones(
                (1, pixel_frames, latent_height, latent_width),
                device=i2v_images.device,
                dtype=i2v_images.dtype
            )

            mask[:, :frames_with_start_image] = 0.0

            # Convert to latent temporal format: [1, 4, T_latent, H, W]
            start_mask_repeated = mask[:, 0:1].repeat(1, 4, 1, 1)
            mask_middle = mask[:, 1:]
            mask = torch.cat([start_mask_repeated, mask_middle], dim=1)

            num_groups = mask.shape[1] // 4
            mask = mask[:, :num_groups * 4]
            mask = mask.view(1, num_groups, 4, latent_height, latent_width)
            mask = mask.transpose(1, 2)

            final_concat_mask = mask
            debug_info.append(f"Auto-generated mask: {frames_with_start_image} pixel frames → {num_groups} latent frames")

        else:
            final_concat_mask = torch.ones(
                (1, 4, temporal_latent, latent_height, latent_width),
                device=comfy.model_management.intermediate_device()
            )
            debug_info.append(f"Auto-generated ones mask (all generate): {final_concat_mask.shape}")

        # ========== STEP 7: APPLY TO CONDITIONING ==========

        # BindWeave Architecture:
        # Channel layout: [noise (ch 0-15), mask (ch 16-19), concat_latent_image (ch 20-35)]
        # - 4 reference frames PREPENDED to the I2V conditioning in concat_latent_image
        # - concat_latent_image: [ref0, ref1, ref2, ref3, start_image frames] (25 frames, 16 channels)
        # - concat_mask: matches concat_latent_image temporal dimension (25 frames, 4 channels)
        # - Noise padding is handled internally by WAN21_Bindweave._apply_model

        # Concatenate references + start_image for concat_latent_image (I2V conditioning)
        reference_concat = torch.cat(reference_latents_batched, dim=2)
        full_concat_latent = torch.cat([reference_concat, start_latent], dim=2)
        debug_info.append(f"concat_latent_image (refs + start): {full_concat_latent.shape}")

        # Create mask to cover all frames (4 refs + start frames = total)
        # Mask semantics:
        # Default (mask_refs):
        #   - Reference frames WITH data → mask = 0.0
        #   - Reference frames EMPTY (zero-padded) → mask = 1.0
        #   - Start image frames WITH data → mask = 0.0
        #   - Frames to GENERATE → mask = 1.0

        if ref_masks is not None:
            # Use provided reference masks
            debug_info.append(f"Using provided ref_masks: shape={ref_masks.shape}")

            # ref_masks should be [batch, frames] where batch <= 4
            # Normalize to [1, 4, T, H, W] format
            if ref_masks.ndim == 2:
                # [batch, spatial] -> assume batch is temporal frames, spatial needs to be split
                # This is likely [T, H*W] - need to reshape
                num_ref_masks = ref_masks.shape[0]
                ref_masks = ref_masks.view(num_ref_masks, 1, latent_height, latent_width)
                ref_masks = ref_masks.unsqueeze(0).unsqueeze(0)  # [1, 1, T, H, W]
            elif ref_masks.ndim == 3:
                # [batch, H, W] -> [1, 1, batch, H, W]
                ref_masks = ref_masks.unsqueeze(0).unsqueeze(0)
            elif ref_masks.ndim == 4:
                # [batch, 1, H, W] -> [1, 1, batch, H, W]
                ref_masks = ref_masks.unsqueeze(0)

            # Now ref_masks should be [1, C, T, H, W]
            num_ref_masks = ref_masks.shape[2]

            # Resize spatial dimensions if needed
            if ref_masks.shape[-2] != latent_height or ref_masks.shape[-1] != latent_width:
                ref_masks = comfy.utils.common_upscale(
                    ref_masks.view(-1, ref_masks.shape[-2], ref_masks.shape[-1]).unsqueeze(1),
                    latent_width, latent_height, "bilinear", "center"
                ).squeeze(1).view(1, ref_masks.shape[1], num_ref_masks, latent_height, latent_width)
                debug_info.append(f"  Resized ref_masks spatial to {latent_height}x{latent_width}")

            # Expand to 4 channels if needed
            if ref_masks.shape[1] == 1:
                ref_masks = ref_masks.repeat(1, 4, 1, 1, 1)
            elif ref_masks.shape[1] != 4:
                ref_masks = torch.mean(ref_masks, dim=1, keepdim=True).repeat(1, 4, 1, 1, 1)

            # Pad temporal dimension to 4 frames with ones (empty slots) if needed
            reference_mask = torch.ones(
                (1, 4, 4, latent_height, latent_width),
                device=ref_masks.device,
                dtype=ref_masks.dtype
            )

            # Copy provided masks (up to 4 frames)
            frames_to_copy = min(num_ref_masks, 4)
            reference_mask[:, :, :frames_to_copy, :, :] = ref_masks[:, :, :frames_to_copy, :, :]
            debug_info.append(f"Reference mask: Using {frames_to_copy} provided masks, {4-frames_to_copy} ones-padded (empty)")

        else:
            # Auto-generate reference mask based on num_references
            reference_mask = torch.ones(
                (1, 4, 4, latent_height, latent_width),
                device=final_concat_mask.device,
                dtype=final_concat_mask.dtype
            )

            # Set mask to 0.0 for reference frames that have actual images
            if num_references > 0:
                reference_mask[:, :, :num_references, :, :] = 0.0
                debug_info.append(f"Reference mask: {num_references} frames set to 0.0 (use), {4-num_references} frames set to 1.0 (empty)")
            else:
                debug_info.append("Reference mask: all 4 frames set to 1.0 (no references)")

        # Apply mask inversion if requested
        if ref_mask_handling == "invert_mask_refs":
            reference_mask = 1.0 - reference_mask
            debug_info.append("Applied reference mask inversion (1.0 - mask)")

        # Concatenate reference mask with start_image mask
        full_mask = torch.cat([reference_mask, final_concat_mask], dim=2)
        debug_info.append(f"concat_mask (refs + start): {full_mask.shape}")

        conditioning_dict = {
            "concat_latent_image": full_concat_latent,     # 4 refs + 21 start = 25 frames (ch 20-35)
            "concat_mask": full_mask,                       # 25 frames (ch 16-19)
            # Note: Noise padding handled internally by model
        }

        # Apply to positive and negative conditioning
        positive = node_helpers.conditioning_set_values(positive, conditioning_dict)
        negative = node_helpers.conditioning_set_values(negative, conditioning_dict)

        debug_info.append("Applied BindWeave conditioning: concat_latent_image (25 frames), concat_mask (25 frames)")

        # Add QwenVL embeddings if provided
        final_qwen_pos = None
        if qwen_conditioning_pos is not None:
            final_qwen_pos = qwen_conditioning_pos[0][0]
            debug_info.append(f"QwenVL (pos): {final_qwen_pos.shape}")

        final_qwen_neg = None
        if qwen_conditioning_neg is not None:
            final_qwen_neg = qwen_conditioning_neg[0][0]
            debug_info.append(f"QwenVL (neg): {final_qwen_neg.shape}")

        # Apply to conditioning
        if final_qwen_pos is not None:
            positive = node_helpers.conditioning_set_values(positive, {"add_text_emb": final_qwen_pos})

        if final_qwen_neg is not None:
            negative = node_helpers.conditioning_set_values(negative, {"add_text_emb": final_qwen_neg})

        # Add CLIP vision if provided
        if clip_vision_output is not None:
            if hasattr(clip_vision_output, 'last_hidden_state'):
                clip_embeds = clip_vision_output.last_hidden_state
                debug_info.append(f"Extracted CLIP last_hidden_state: {clip_embeds.shape}")

                B, T, C = clip_embeds.shape
                tokens_per_image = 257
                target_len = 4 * tokens_per_image  # 4 references * 257 tokens each = 1028

                # Check if this is batched CLIP embeddings (multiple images encoded together)
                if T == tokens_per_image:
                    # Batched: [B, 257, C] where B is number of reference images
                    num_ref_images = min(B, 4)  # Cap at 4 reference frames

                    # Take up to 4 images from the batch
                    clip_embeds_to_use = clip_embeds[:num_ref_images, :, :]  # [num_ref, 257, C]

                    # Flatten to [1, num_ref*257, C]
                    sequential_embeds = clip_embeds_to_use.reshape(1, num_ref_images * tokens_per_image, C)

                    # Pad remaining reference slots (or don't pad if strategy is no_padding)
                    if num_ref_images < 4 and clip_padding_strategy != "no_padding":
                        remaining_tokens = (4 - num_ref_images) * tokens_per_image

                        # Create padding based on strategy
                        if clip_padding_strategy == "zeros":
                            pad = torch.zeros(1, remaining_tokens, C, device=clip_embeds.device, dtype=clip_embeds.dtype)
                            pad_desc = "zero-padded"
                        elif clip_padding_strategy == "mean":
                            mean_embed = sequential_embeds.mean(dim=1, keepdim=True)  # [1, 1, C]
                            pad = mean_embed.repeat(1, remaining_tokens, 1)
                            pad_desc = "mean-padded"
                        elif clip_padding_strategy == "repeat_last":
                            last_token = sequential_embeds[:, -1:, :]  # [1, 1, C]
                            pad = last_token.repeat(1, remaining_tokens, 1)
                            pad_desc = "repeat-last-padded"

                        final_embeds = torch.cat([sequential_embeds, pad], dim=1)
                        debug_info.append(f"CLIP: {num_ref_images} images → {4-num_ref_images} frames {pad_desc}")
                    else:
                        final_embeds = sequential_embeds
                        if clip_padding_strategy == "no_padding":
                            debug_info.append(f"CLIP: {num_ref_images} images, no padding ({num_ref_images * tokens_per_image} tokens)")
                        else:
                            debug_info.append(f"CLIP: 4 images, all reference frames")

                elif T < target_len:
                    if clip_padding_strategy == "no_padding":
                        final_embeds = clip_embeds
                        debug_info.append(f"CLIP: {T} tokens, no padding")
                    elif clip_padding_strategy == "zeros":
                        pad = torch.zeros(B, target_len - T, C, device=clip_embeds.device, dtype=clip_embeds.dtype)
                        final_embeds = torch.cat([clip_embeds, pad], dim=1)
                        debug_info.append(f"CLIP: {T} → {target_len} tokens (zeros)")
                    elif clip_padding_strategy == "mean":
                        mean_embed = clip_embeds.mean(dim=1, keepdim=True)
                        pad = mean_embed.repeat(1, target_len - T, 1)
                        final_embeds = torch.cat([clip_embeds, pad], dim=1)
                        debug_info.append(f"CLIP: {T} → {target_len} tokens (mean)")
                    elif clip_padding_strategy == "repeat_last":
                        last_token = clip_embeds[:, -1:, :]
                        pad = last_token.repeat(1, target_len - T, 1)
                        final_embeds = torch.cat([clip_embeds, pad], dim=1)
                        debug_info.append(f"CLIP: {T} → {target_len} tokens (repeat_last)")

                elif T > target_len:
                    # Truncate if too long
                    final_embeds = clip_embeds[:, :target_len, :]
                    debug_info.append(f"Truncated CLIP embeddings from {T} to {target_len} tokens")

                else:
                    # Already correct size
                    final_embeds = clip_embeds
                    debug_info.append(f"CLIP embeddings already correct size: {T} tokens")

                # Create a new Output object with processed embeddings
                clip_output = comfy.clip_vision.Output()
                clip_output["last_hidden_state"] = final_embeds
                clip_output["image_embeds"] = clip_vision_output["image_embeds"] if B == 1 else clip_vision_output["image_embeds"][:1]
                clip_output["penultimate_hidden_states"] = final_embeds
                debug_info.append(f"Set penultimate_hidden_states: {final_embeds.shape}")

                clip_output["mm_projected"] = getattr(clip_vision_output, "mm_projected", None)
                if hasattr(clip_vision_output, "all_hidden_states"):
                    clip_output["all_hidden_states"] = clip_vision_output["all_hidden_states"]

            else:
                # Fallback: pass through unchanged if structure is unexpected
                debug_info.append("Warning: Unexpected CLIP vision output format, passing through unchanged")
                clip_output = clip_vision_output

            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_output})
            debug_info.append("Added CLIP vision output to conditioning")

        # ========== STEP 8: CREATE OUTPUT LATENT (Target size, WITHOUT reference frames) ==========
        latent = torch.zeros(
            [batch_size, 16, temporal_latent, latent_height, latent_width],
            device=comfy.model_management.intermediate_device()
        )

        out_latent = {"samples": latent}

        # Compile debug info
        debug_summary = [
            f"=== WanEx BindweaveSubjectToVid ===",
            f"Output: {latent.shape} | Concat Latent: {full_concat_latent.shape} | Concat Mask: {full_mask.shape}",
            f"References: {num_references} provided, {num_padding} padded (4 total)",
            f"",
        ] + debug_info

        debug_text = "\n".join(debug_summary)

        # Create preview outputs (like WanVideoWrapper's WanVideoAddBindweaveEmbeds)
        concat_latent_preview = {"samples": full_concat_latent}
        concat_mask_preview = full_mask[0, 0].float()
        mask_channels_preview = full_mask[0].permute(1, 0, 2, 3).reshape(-1, full_mask.shape[3], full_mask.shape[4])

        return (positive, negative, out_latent, debug_text, concat_latent_preview, concat_mask_preview, mask_channels_preview)

class WanEx_QwenVLTextConditioning:
    """Encode QwenVL text without template wrapping. Returns CONDITIONING type."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP",),
            "prompt": ("STRING", {"default": "", "multiline": True}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "WanExperiments"

    def encode(self, clip, prompt):
        tokens = clip.tokenize(prompt, llama_template="{}")
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return (conditioning,)


class WanEx_ImageEmbedsPreview:
    """Preview image_embeds from WanVideoWrapper nodes without modification. Outputs latent and mask for inspection."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_embeds": ("WANVIDIMAGE_EMBEDS",),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "LATENT", "MASK", "MASK")
    RETURN_NAMES = ("image_embeds", "latent_preview", "mask_preview", "mask_channels_preview")
    FUNCTION = "preview"
    CATEGORY = "WanExperiments"

    def preview(self, image_embeds):
        latent = image_embeds.get("image_embeds", None)
        mask = image_embeds.get("mask", None)

        latent_preview = None
        mask_preview = None
        mask_channels_preview = None

        if latent is not None:
            latent_preview = {"samples": latent.unsqueeze(0) if latent.ndim == 4 else latent}

        if mask is not None:
            mask_preview = mask[0].float()
            mask_channels_preview = mask.permute(1, 0, 2, 3).reshape(-1, mask.shape[2], mask.shape[3]).float()

        return (image_embeds, latent_preview, mask_preview, mask_channels_preview)


class WanEx_ConditioningEmbedsPreview:
    """Extract and preview concat_latent_image and concat_mask from CONDITIONING. Useful for debugging conditioning data."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "MASK", "MASK")
    RETURN_NAMES = ("conditioning", "concat_latent_preview", "concat_mask_preview", "mask_channels_preview")
    FUNCTION = "preview"
    CATEGORY = "WanExperiments"

    def preview(self, conditioning):
        concat_latent = None
        mask_preview = None
        mask_channels_preview = None

        if conditioning and len(conditioning) > 0:
            cond_dict = conditioning[0][1]

            concat_latent_image = cond_dict.get("concat_latent_image", None)
            concat_mask = cond_dict.get("concat_mask", None)

            if concat_latent_image is not None:
                concat_latent = {"samples": concat_latent_image}

            if concat_mask is not None:
                if concat_mask.ndim == 5:
                    mask_preview = concat_mask[0, 0].float()
                    mask_channels_preview = concat_mask[0].permute(1, 0, 2, 3).reshape(-1, concat_mask.shape[3], concat_mask.shape[4]).float()
                elif concat_mask.ndim == 4:
                    mask_preview = concat_mask[0].float()
                    mask_channels_preview = concat_mask.permute(1, 0, 2, 3).reshape(-1, concat_mask.shape[2], concat_mask.shape[3]).float()

        return (conditioning, concat_latent, mask_preview, mask_channels_preview)

# Derived from ComfyUI-PainterI2V motion amplitude scaling math - https://github.com/princepainter/ComfyUI-PainterI2V
class WanEx_PainterMotionAmplitude:
    """
    Applies PainterI2V motion amplitude scaling to existing conditioning.
    Works with conditioning that has concat_latent_image already set (e.g., from WanImageToVideo or WanEx I2VCustomEmbeds).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "motion_amplitude": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05}),
                "base_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "mean_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "base_frame": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
            },
            "optional": {
                "motion_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "execute"
    CATEGORY = "WanExperiments"
    DISPLAY_NAME = "WanEx PainterMotionAmplitude"

    def execute(
        self,
        positive,
        negative,
        motion_amplitude=1.15,
        base_strength=1.0,
        mean_strength=1.0,
        base_frame=0,
        motion_mask=None,
    ):
        # Extract concat_latent_image from positive conditioning
        if len(positive) == 0 or "concat_latent_image" not in positive[0][1]:
            raise RuntimeError("WanEx PainterMotionAmplitude: conditioning must have 'concat_latent_image' set (e.g., from WanImageToVideo node)")

        concat_latent_image = positive[0][1]["concat_latent_image"].clone()
        num_frames = concat_latent_image.shape[2]

        # Validate base_frame
        if base_frame < 0 or base_frame >= num_frames:
            raise RuntimeError(f"WanEx PainterMotionAmplitude: base_frame {base_frame} is out of range (0 to {num_frames - 1})")

        # Determine mask for scaling
        # We support two modes:
        # 1. Full-frame masks: average spatially, threshold, apply scaling to entire frame
        # 2. Partial masks: apply scaling only to masked regions (spatial blending)

        spatial_mask = None  # Will be set if we need per-pixel blending

        if motion_mask is not None:
            # Use provided motion_mask
            # Threshold to binary
            motion_mask = (motion_mask > 0.5).float()

            # Reshape to match latent dimensions [1, 1, T, H, W]
            if motion_mask.dim() == 3:
                # [T, H, W] -> [1, 1, T, H, W]
                spatial_mask = motion_mask.unsqueeze(0).unsqueeze(0)
            elif motion_mask.dim() == 4:
                # [B, T, H, W] or [1, T, H, W] -> [1, 1, T, H, W]
                if motion_mask.shape[0] == 1:
                    spatial_mask = motion_mask.unsqueeze(1)
                else:
                    spatial_mask = motion_mask.unsqueeze(0)
            else:
                spatial_mask = motion_mask.view(1, 1, -1, 1, 1)

            # Resize spatial dims to match latent if needed
            latent_h, latent_w = concat_latent_image.shape[3], concat_latent_image.shape[4]
            if spatial_mask.shape[3] != latent_h or spatial_mask.shape[4] != latent_w:
                spatial_mask = torch.nn.functional.interpolate(
                    spatial_mask.view(-1, 1, spatial_mask.shape[3], spatial_mask.shape[4]),
                    size=(latent_h, latent_w),
                    mode='nearest'
                ).view(1, 1, -1, latent_h, latent_w)

            # Adjust temporal dimension
            if spatial_mask.shape[2] < num_frames:
                # Pad with ones (scale remaining frames)
                pad_frames = num_frames - spatial_mask.shape[2]
                padding = torch.ones(1, 1, pad_frames, latent_h, latent_w, device=spatial_mask.device, dtype=spatial_mask.dtype)
                spatial_mask = torch.cat([spatial_mask, padding], dim=2)
            elif spatial_mask.shape[2] > num_frames:
                spatial_mask = spatial_mask[:, :, :num_frames]

            # Check if mask is full-frame or partial per frame
            frame_mask = spatial_mask.mean(dim=(0, 1, 3, 4))  # [T]

        elif "concat_mask" in positive[0][1]:
            # Use concat_mask from conditioning
            concat_mask = positive[0][1]["concat_mask"]
            # Threshold to binary
            concat_mask = (concat_mask > 0.5).float()
            # concat_mask shape: [1, C, T, H, W] - average across channels for spatial mask
            spatial_mask = concat_mask.mean(dim=1, keepdim=True)  # [1, 1, T, H, W]
            frame_mask = spatial_mask.mean(dim=(0, 1, 3, 4))  # [T]
        else:
            # No mask available - assume all frames except base_frame are noise
            frame_mask = torch.ones(num_frames, device=concat_latent_image.device, dtype=concat_latent_image.dtype)
            frame_mask[base_frame] = 0.0
            spatial_mask = None

        # Identify frames that have any masked pixels (need scaling)
        # A frame needs processing if its average mask > 0 (has some pixels to scale)
        frames_to_process = (frame_mask > 0.01).nonzero(as_tuple=True)[0]

        if len(frames_to_process) == 0:
            # No frames to scale, return unchanged
            return (positive, negative)

        # Skip if no scaling needed
        if motion_amplitude == 1.0 and base_strength == 1.0 and mean_strength == 1.0:
            return (positive, negative)

        # Get base frame latent
        base_latent = concat_latent_image[:, :, base_frame:base_frame+1]

        # Apply motion amplitude scaling to each frame that needs it
        for idx in frames_to_process:
            idx = idx.item()
            if idx == base_frame:
                continue  # Don't modify the base frame

            noise_latent = concat_latent_image[:, :, idx:idx+1]

            # Compute residual and its mean
            diff = noise_latent - base_latent
            diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
            diff_centered = diff - diff_mean

            # Scale: modulated base + scaled centered residual + scaled mean
            scaled = (base_latent * base_strength) + (diff_centered * motion_amplitude) + (diff_mean * mean_strength)
            scaled = torch.clamp(scaled, -6, 6)

            # Apply spatially if we have a partial mask
            if spatial_mask is not None:
                frame_spatial_mask = spatial_mask[:, :, idx:idx+1]
                # Check if this frame has a partial mask (not all 0 or all 1)
                mask_mean = frame_spatial_mask.mean()
                if mask_mean > 0.01 and mask_mean < 0.99:
                    # Partial mask - blend between original and scaled
                    scaled = noise_latent * (1 - frame_spatial_mask) + scaled * frame_spatial_mask

            concat_latent_image[:, :, idx:idx+1] = scaled

        # Update conditioning
        def update_conditioning(cond, new_latent):
            new_cond = []
            for c_tensor, c_dict in cond:
                new_dict = c_dict.copy()
                new_dict["concat_latent_image"] = new_latent
                new_cond.append([c_tensor, new_dict])
            return new_cond

        pos_out = update_conditioning(positive, concat_latent_image)
        neg_out = update_conditioning(negative, concat_latent_image)

        return (pos_out, neg_out)
    
class WanEx_HuMoImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanEx_HuMoImageToVideo",
            category="WanExperiments",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=97, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.AudioEncoderOutput.Input("audio_encoder_output", optional=True),
                io.Image.Input("ref_images", optional=True),
                io.Image.Input("start_images", optional=True),
                io.Image.Input("end_images", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, ref_images=None, audio_encoder_output=None, start_images=None, end_images=None) -> io.NodeOutput:
        latent_t = ((length - 1) // 4) + 1
        latent_height = height // 8
        latent_width = width // 8
        latent = torch.zeros([batch_size, 16, latent_t, latent_height, latent_width], device=comfy.model_management.intermediate_device())

        if ref_images is not None:
            ref_images = comfy.utils.common_upscale(ref_images[:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            num_refs = ref_images.shape[0]
            ref_latent = vae.encode(ref_images[:1, :, :, :3])
            # Append the other reference latents
            for i in range(num_refs - 1):
                ref_latent = torch.cat((ref_latent, vae.encode(ref_images[i + 1:i + 2, :, :, :3])), dim=2)
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)
        else:
            zero_latent = torch.zeros([batch_size, 16, 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [zero_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [zero_latent]}, append=True)

        if audio_encoder_output is not None:
            audio_emb = torch.stack(audio_encoder_output["encoded_audio_all_layers"], dim=2)
            audio_len = audio_encoder_output["audio_samples"] // 640
            audio_emb = audio_emb[:, :audio_len * 2]

            from comfy_extras.nodes_wan import linear_interpolation, get_audio_emb_window

            feat0 = linear_interpolation(audio_emb[:, :, 0: 8].mean(dim=2), 50, 25)
            feat1 = linear_interpolation(audio_emb[:, :, 8: 16].mean(dim=2), 50, 25)
            feat2 = linear_interpolation(audio_emb[:, :, 16: 24].mean(dim=2), 50, 25)
            feat3 = linear_interpolation(audio_emb[:, :, 24: 32].mean(dim=2), 50, 25)
            feat4 = linear_interpolation(audio_emb[:, :, 32], 50, 25)
            audio_emb = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]  # [T, 5, 1280]
            audio_emb, _ = get_audio_emb_window(audio_emb, length, frame0_idx=0)

            audio_emb = audio_emb.unsqueeze(0)
            audio_emb_neg = torch.zeros_like(audio_emb)
            positive = node_helpers.conditioning_set_values(positive, {"audio_embed": audio_emb})
            negative = node_helpers.conditioning_set_values(negative, {"audio_embed": audio_emb_neg})
        else:
            # If no audio embedding, leave as none, model code handles it
            pass

        pixel_frames = (latent_t - 1) * 4 + 1
        have_start = start_images is not None
        have_end = end_images is not None

        if have_start or have_end:
            # Prefill neutral gray frames once, then overlay start/end data
            image = torch.ones(
                (length, height, width, 3),
                device=comfy.model_management.intermediate_device(),
                dtype=torch.float32,
            ) * 0.5

            mask = torch.ones(
                (1, pixel_frames, latent_height, latent_width),
                device=image.device,
                dtype=image.dtype,
            )

            if have_start:
                start_resized = comfy.utils.common_upscale(
                    start_images[:length].movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
                image[:start_resized.shape[0]] = start_resized[:, :, :, :3]
                start_frames = min(start_images.shape[0], pixel_frames)
                mask[:, :start_frames] = 0.0

            if have_end:
                end_resized = comfy.utils.common_upscale(
                    end_images[:length].movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
                tail = end_resized.shape[0]
                if tail > 0:
                    image[-tail:] = end_resized[:, :, :, :3]
                    end_frames = min(end_images.shape[0], pixel_frames)
                    mask[:, -end_frames:] = 0.0

            concat_latent_image = vae.encode(image)

            start_mask_repeated = mask[:, 0:1].repeat(1, 4, 1, 1)
            mask_middle = mask[:, 1:]
            mask = torch.cat([start_mask_repeated, mask_middle], dim=1)
            num_groups = mask.shape[1] // 4
            mask = mask[:, :num_groups * 4]
            mask = mask.view(1, num_groups, 4, latent_height, latent_width)
            mask = mask.transpose(1, 2)

            conditioning_updates = {
                "concat_latent_image": concat_latent_image,
                "concat_mask": mask,
            }
            positive = node_helpers.conditioning_set_values(positive, conditioning_updates)
            negative = node_helpers.conditioning_set_values(negative, conditioning_updates)

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)


# Based on WanContextWindowsManual in ComfyUI core
class WanEx_ContextWindowsAdvanced:
    """
    Advanced context windows node for WAN models with additional features:
    - cond_retain_index_list: Preserve specific frame indices (e.g., initial image) in conditioning across all windows
    - split_conds_to_windows: Split multiple conditionings to different windows based on region
    """

    @classmethod
    def INPUT_TYPES(s):
        import comfy.context_windows
        return {
            "required": {
                "model": ("MODEL",),
                "context_length": ("INT", {
                    "default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4,
                    "tooltip": "The length of the context window in frames (will be converted to latent space)."
                }),
                "context_overlap": ("INT", {
                    "default": 29, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1,
                    "tooltip": "The overlap between context windows in frames."
                }),
                "context_schedule": (["standard_static", "standard_uniform", "looped_uniform", "batched"], {
                    "default": "standard_static",
                    "tooltip": "The scheduling strategy for context windows."
                }),
                "context_stride": ("INT", {
                    "default": 1, "min": 1, "max": 16,
                    "tooltip": "The stride of the context window; only applicable to uniform schedules."
                }),
                "closed_loop": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether to close the context window loop; only applicable to looped schedules."
                }),
                "fuse_method": (["pyramid", "flat", "relative", "overlap-linear"], {
                    "default": "pyramid",
                    "tooltip": "The method to use to fuse/blend overlapping context windows."
                }),
                "freenoise": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether to apply FreeNoise noise shuffling, improves window blending."
                }),
            },
            "optional": {
                "cond_retain_index_list": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated list of latent frame indices to retain in conditioning for each window. "
                              "For example, '0' will preserve the initial/start image conditioning in every window. "
                              "Use '0,1' to retain first two frames, etc."
                }),
                "split_conds_to_windows": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether to split multiple conditionings (created by ConditioningCombine) to each window "
                              "based on region index. Useful for applying different prompts to different parts of the video."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_context_windows"
    CATEGORY = "WanExperiments"

    def apply_context_windows(self, model, context_length, context_overlap, context_schedule, context_stride,
                               closed_loop, fuse_method, freenoise,
                               cond_retain_index_list="", split_conds_to_windows=False):
        import comfy.context_windows

        # Convert frame counts to latent space (WAN uses 4:1 temporal compression)
        latent_context_length = max(((context_length - 1) // 4) + 1, 1)
        latent_context_overlap = max(((context_overlap - 1) // 4) + 1, 0)

        # Clone the model
        model = model.clone()

        # Create context handler with all features enabled
        model.model_options["context_handler"] = comfy.context_windows.IndexListContextHandler(
            context_schedule=comfy.context_windows.get_matching_context_schedule(context_schedule),
            fuse_method=comfy.context_windows.get_matching_fuse_method(fuse_method),
            context_length=latent_context_length,
            context_overlap=latent_context_overlap,
            context_stride=context_stride,
            closed_loop=closed_loop,
            dim=2,  # WAN models use dim=2 for temporal
            freenoise=freenoise,
            cond_retain_index_list=cond_retain_index_list,
            split_conds_to_windows=split_conds_to_windows
        )

        # make memory usage calculation only take into account the context window latents
        comfy.context_windows.create_prepare_sampling_wrapper(model)
        if freenoise:
            comfy.context_windows.create_sampler_sample_wrapper(model)

        return (model,)

def _ensure_kornia_available():
    """Lazy import Kornia so optional installs can raise a guided error."""
    global _KORNIA_MODULE
    if _KORNIA_MODULE is None:
        try:
            import kornia  # pylint: disable=import-error
            _KORNIA_MODULE = kornia
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "WanEx_VideoColorMatch requires the optional dependency 'kornia'. "
                "Install it with `pip install kornia` to use this node."
            ) from exc
    return _KORNIA_MODULE


def _resolve_frame_index(num_frames, requested_index):
    """Clamp or wrap a requested frame index for reference batches."""
    if num_frames <= 1:
        return 0
    resolved = requested_index
    if resolved < 0:
        resolved = num_frames + resolved
    resolved = max(0, min(num_frames - 1, resolved))
    return resolved

# Modified from ImageColorMatch node in ComfyUI Essentials node pack (https://github.com/cubiq/ComfyUI_essentials)
class WanEx_VideoColorMatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference": ("IMAGE",),
                "target_index": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Index of the frame in the batch to use for computing color matching metrics."
                }),
                "reference_index": ("INT", {
                    "default": -1, "min": -1, "max": 10000, "step": 1,
                    "tooltip": "Frame index to sample from the reference batch (-1 selects the last frame)."
                }),
                "color_space": (["LAB", "YCbCr", "RGB", "LUV", "YUV", "XYZ"],),
                "factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "device": (["auto", "cpu", "gpu"],),
                "batch_size": ("INT", {
                    "default": 0, "min": 0, "max": 1024, "step": 1,
                    "tooltip": "Processing batch size for memory efficiency. 0 = process all at once."
                }),
            },
            "optional": {
                "reference_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "WanExperiments"
    DESCRIPTION = "Applies uniform color matching across a video batch using a single target frame for metric computation."

    def execute(self, images, reference, target_index, reference_index, color_space, factor, device, batch_size, reference_mask=None):
        _ensure_kornia_available()

        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = 'cpu'

        # Validate target_index
        num_frames = images.shape[0]
        if target_index >= num_frames:
            target_index = num_frames - 1

        # Permute to [B, C, H, W] format
        images_perm = images.permute([0, 3, 1, 2])
        reference_perm = reference.permute([0, 3, 1, 2]).to(device)
        ref_frame_count = reference_perm.shape[0]
        ref_idx = _resolve_frame_index(ref_frame_count, reference_index)
        reference_perm = reference_perm[ref_idx:ref_idx+1]

        # Process reference_mask if provided
        if reference_mask is not None:
            assert reference_mask.ndim == 3, f"Expected reference_mask to have 3 dimensions, but got {reference_mask.ndim}"
            assert reference_mask.shape[0] == ref_frame_count, f"Frame count mismatch: reference_mask has {reference_mask.shape[0]} frames, but reference has {ref_frame_count}"

            reference_mask = reference_mask[ref_idx:ref_idx+1]

            reference_mask = reference_mask.unsqueeze(1).to(device)
            reference_mask = (reference_mask > 0.5).float()

            if reference_mask.shape[2:] != reference_perm.shape[2:]:
                reference_mask = comfy.utils.common_upscale(
                    reference_mask,
                    reference_perm.shape[3], reference_perm.shape[2],
                    upscale_method='bicubic',
                    crop='center'
                )

        if batch_size == 0 or batch_size > num_frames:
            batch_size = num_frames

        # Convert reference to target color space
        ref_converted = self._convert_to_color_space(reference_perm, color_space)
        reference_mean, reference_std = self._compute_mean_std(ref_converted, reference_mask)

        # Get the target frame and compute its statistics
        target_frame = images_perm[target_index:target_index+1].to(device)
        target_converted = self._convert_to_color_space(target_frame, color_space)
        target_mean, target_std = self._compute_mean_std(target_converted)

        # Process images in batches using the SAME target statistics for all frames
        image_batches = torch.split(images_perm, batch_size, dim=0)
        output = []

        for img_batch in image_batches:
            img_batch = img_batch.to(device)

            # Convert to color space
            img_converted = self._convert_to_color_space(img_batch, color_space)

            # Apply color matching using the target frame's statistics (uniform across batch)
            # Formula: (image - target_mean) / target_std * reference_std + reference_mean
            matched = torch.nan_to_num((img_converted - target_mean) / target_std) * torch.nan_to_num(reference_std) + reference_mean
            matched = factor * matched + (1 - factor) * img_converted

            # Convert back to RGB
            matched = self._convert_from_color_space(matched, color_space)

            out = matched.permute([0, 2, 3, 1]).clamp(0, 1).to(comfy.model_management.intermediate_device())
            output.append(out)

        output = torch.cat(output, dim=0)
        return (output,)

    def _convert_to_color_space(self, tensor, color_space):
        kornia = _ensure_kornia_available()
        if color_space == "LAB":
            return kornia.color.rgb_to_lab(tensor)
        elif color_space == "YCbCr":
            return kornia.color.rgb_to_ycbcr(tensor)
        elif color_space == "LUV":
            return kornia.color.rgb_to_luv(tensor)
        elif color_space == "YUV":
            return kornia.color.rgb_to_yuv(tensor)
        elif color_space == "XYZ":
            return kornia.color.rgb_to_xyz(tensor)
        return tensor  # RGB

    def _convert_from_color_space(self, tensor, color_space):
        kornia = _ensure_kornia_available()
        if color_space == "LAB":
            return kornia.color.lab_to_rgb(tensor)
        elif color_space == "YCbCr":
            return kornia.color.ycbcr_to_rgb(tensor)
        elif color_space == "LUV":
            return kornia.color.luv_to_rgb(tensor)
        elif color_space == "YUV":
            return kornia.color.yuv_to_rgb(tensor)
        elif color_space == "XYZ":
            return kornia.color.xyz_to_rgb(tensor)
        return tensor  # RGB

    def _compute_mean_std(self, tensor, mask=None):
        if mask is not None:
            masked_tensor = tensor * mask
            mask_sum = mask.sum(dim=[2, 3], keepdim=True)
            mask_sum = torch.clamp(mask_sum, min=1e-6)
            mean = torch.nan_to_num(masked_tensor.sum(dim=[2, 3], keepdim=True) / mask_sum)
            std = torch.sqrt(torch.nan_to_num(((masked_tensor - mean) ** 2 * mask).sum(dim=[2, 3], keepdim=True) / mask_sum))
        else:
            mean = tensor.mean(dim=[2, 3], keepdim=True)
            std = tensor.std(dim=[2, 3], keepdim=True)
        return mean, std


class WanEx_VideoContrastMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference": ("IMAGE",),
                "target_index": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Frame index in the batch that defines the contrast correction mapping."
                }),
                "reference_index": ("INT", {
                    "default": -1, "min": -1, "max": 10000, "step": 1,
                    "tooltip": "Frame index to sample from the reference batch (-1 selects the last frame)."
                }),
                "contrast_space": (["LAB", "YCbCr", "YUV"], {
                    "default": "LAB",
                    "tooltip": "Color space used to extract the luminance channel for contrast matching."
                }),
                "technique": ([
                    "global_mean_std",
                    "percentile_levels",
                    "histogram"
                ], {
                    "default": "global_mean_std",
                    "tooltip": "Select the contrast/levels matching algorithm."
                }),
                "match_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0.0) and fully matched result (1.0)."
                }),
                "device": (["auto", "cpu", "gpu"],),
                "batch_size": ("INT", {
                    "default": 0, "min": 0, "max": 1024, "step": 1,
                    "tooltip": "Processing batch size for memory efficiency. 0 = process all frames at once."
                }),
            },
            "optional": {
                "reference_mask": ("MASK", {
                    "tooltip": "Optional mask to limit which pixels in the reference contribute to statistics."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "WanExperiments"
    DESCRIPTION = "Matches video contrast/levels to a reference using a chosen luminance technique and propagates the correction across frames."

    def execute(self, images, reference, target_index, reference_index, contrast_space, technique, match_strength, device, batch_size, reference_mask=None):
        _ensure_kornia_available()

        if device == "gpu":
            device = comfy.model_management.get_torch_device()
        elif device == "auto":
            device = comfy.model_management.intermediate_device()
        else:
            device = "cpu"

        num_frames = images.shape[0]
        if target_index >= num_frames:
            target_index = num_frames - 1

        images_perm = images.permute(0, 3, 1, 2)
        reference_perm = reference.permute(0, 3, 1, 2).to(device)
        ref_frame_count = reference_perm.shape[0]
        ref_idx = _resolve_frame_index(ref_frame_count, reference_index)
        reference_perm = reference_perm[ref_idx:ref_idx+1]

        ref_mask_tensor = None
        if reference_mask is not None:
            assert reference_mask.ndim == 3, "reference_mask must be [frames, height, width]"
            assert reference_mask.shape[0] == ref_frame_count, "reference_mask frame count must match reference"
            reference_mask = reference_mask[ref_idx:ref_idx+1]
            ref_mask_tensor = reference_mask.unsqueeze(1).to(device)
            ref_mask_tensor = (ref_mask_tensor > 0.5).float()
            if ref_mask_tensor.shape[2:] != reference_perm.shape[2:]:
                ref_mask_tensor = comfy.utils.common_upscale(
                    ref_mask_tensor,
                    reference_perm.shape[3], reference_perm.shape[2],
                    upscale_method="bicubic",
                    crop="center"
                )

        target_frame = images_perm[target_index:target_index + 1].to(device)

        reference_color = self._convert_to_color_space(reference_perm, contrast_space)
        target_color = self._convert_to_color_space(target_frame, contrast_space)

        reference_lum = self._extract_luminance(reference_color, contrast_space)
        target_lum = self._extract_luminance(target_color, contrast_space)

        params = self._build_correction_params(
            reference_lum,
            target_lum,
            ref_mask_tensor,
            technique
        )

        if batch_size == 0 or batch_size > num_frames:
            batch_size = num_frames

        output_batches = []
        for img_batch in torch.split(images_perm, batch_size, dim=0):
            img_batch = img_batch.to(device)
            batch_color = self._convert_to_color_space(img_batch, contrast_space)
            batch_lum = self._extract_luminance(batch_color, contrast_space)
            matched_lum = self._apply_luminance_correction(batch_lum, params, technique)
            batch_color = self._inject_luminance(batch_color, matched_lum, contrast_space)
            corrected = self._convert_from_color_space(batch_color, contrast_space)
            blended = (match_strength * corrected) + ((1.0 - match_strength) * img_batch)
            blended = blended.permute(0, 2, 3, 1).clamp(0, 1)
            output_batches.append(blended.to(comfy.model_management.intermediate_device()))

        output = torch.cat(output_batches, dim=0)
        return (output,)

    def _convert_to_color_space(self, tensor, color_space):
        kornia = _ensure_kornia_available()
        if color_space == "LAB":
            return kornia.color.rgb_to_lab(tensor)
        elif color_space == "YCbCr":
            return kornia.color.rgb_to_ycbcr(tensor)
        elif color_space == "YUV":
            return kornia.color.rgb_to_yuv(tensor)
        return tensor

    def _convert_from_color_space(self, tensor, color_space):
        kornia = _ensure_kornia_available()
        if color_space == "LAB":
            return kornia.color.lab_to_rgb(tensor)
        elif color_space == "YCbCr":
            return kornia.color.ycbcr_to_rgb(tensor)
        elif color_space == "YUV":
            return kornia.color.yuv_to_rgb(tensor)
        return tensor

    def _extract_luminance(self, tensor, color_space):
        if color_space in ("LAB", "YCbCr", "YUV"):
            return tensor[:, :1, :, :]
        # Fallback to perceptual luminance for already-RGB tensors
        return (0.2126 * tensor[:, 0:1] + 0.7152 * tensor[:, 1:2] + 0.0722 * tensor[:, 2:3])

    def _inject_luminance(self, tensor, luminance, color_space):
        if color_space in ("LAB", "YCbCr", "YUV"):
            tensor = tensor.clone()
            tensor[:, :1, :, :] = luminance
            return tensor
        # Approximate injection by scaling RGB channels to match luminance ratio
        base_lum = self._extract_luminance(tensor, "RGB")
        ratio = torch.nan_to_num(luminance / torch.clamp(base_lum, min=1e-4))
        return torch.clamp(tensor * ratio, 0.0, 1.0)

    def _build_correction_params(self, reference_lum, target_lum, reference_mask, technique):
        if technique == "global_mean_std":
            ref_mean, ref_std = self._masked_mean_std(reference_lum, reference_mask)
            tgt_mean, tgt_std = self._masked_mean_std(target_lum)
            return {
                "ref_mean": ref_mean,
                "ref_std": torch.clamp(ref_std, min=1e-4),
                "tgt_mean": tgt_mean,
                "tgt_std": torch.clamp(tgt_std, min=1e-4)
            }
        elif technique == "percentile_levels":
            ref_low, ref_high = self._masked_percentiles(reference_lum, reference_mask)
            tgt_low, tgt_high = self._masked_percentiles(target_lum)
            return {
                "ref_low": ref_low,
                "ref_high": torch.maximum(ref_high, ref_low + 1e-4),
                "tgt_low": tgt_low,
                "tgt_high": torch.maximum(tgt_high, tgt_low + 1e-4)
            }
        elif technique == "histogram":
            mapping = self._build_histogram_mapping(target_lum, reference_lum, reference_mask)
            return {"mapping": mapping}
        else:
            raise ValueError(f"Unsupported technique: {technique}")

    def _apply_luminance_correction(self, tensor, params, technique):
        if technique == "global_mean_std":
            corrected = (tensor - params["tgt_mean"]) / params["tgt_std"]
            corrected = corrected * params["ref_std"] + params["ref_mean"]
            return torch.clamp(corrected, 0.0, 1.0)
        elif technique == "percentile_levels":
            normalized = (tensor - params["tgt_low"]) / (params["tgt_high"] - params["tgt_low"])
            normalized = torch.clamp(normalized, 0.0, 1.0)
            scaled = normalized * (params["ref_high"] - params["ref_low"]) + params["ref_low"]
            return torch.clamp(scaled, 0.0, 1.0)
        elif technique == "histogram":
            return self._apply_histogram_mapping(tensor, params["mapping"])
        else:
            return tensor

    def _masked_mean_std(self, tensor, mask=None):
        if mask is not None:
            masked = tensor * mask
            denom = torch.clamp(mask.sum(dim=[2, 3], keepdim=True), min=1e-6)
            mean = masked.sum(dim=[2, 3], keepdim=True) / denom
            var = ((masked - mean) ** 2 * mask).sum(dim=[2, 3], keepdim=True) / denom
            std = torch.sqrt(torch.clamp(var, min=1e-6))
        else:
            mean = tensor.mean(dim=[2, 3], keepdim=True)
            std = tensor.std(dim=[2, 3], keepdim=True)
        return mean, std

    def _masked_percentiles(self, tensor, mask=None, low=0.02, high=0.98):
        flat = tensor.flatten()
        if mask is not None:
            mask_flat = mask.flatten()
            flat = flat[mask_flat > 0.5]
        if flat.numel() == 0:
            flat = tensor.flatten()
        flat = torch.sort(flat)[0]
        low_idx = max(int((flat.numel() - 1) * low), 0)
        high_idx = max(int((flat.numel() - 1) * high), 0)
        return flat[low_idx], flat[high_idx]

    def _build_histogram_mapping(self, target_lum, reference_lum, reference_mask=None, bins=256):
        target_flat = target_lum.flatten()
        reference_flat = reference_lum.flatten()
        if reference_mask is not None:
            mask_flat = reference_mask.flatten()
            reference_flat = reference_flat[mask_flat > 0.5]
        if reference_flat.numel() == 0:
            reference_flat = reference_lum.flatten()
        target_cdf = self._cdf_from_tensor(target_flat, bins)
        reference_cdf = self._cdf_from_tensor(reference_flat, bins)
        mapping = torch.zeros(bins, device=target_lum.device)
        ref_idx = 0
        for bin_idx in range(bins):
            src_val = target_cdf[bin_idx]
            while ref_idx < bins - 1 and reference_cdf[ref_idx] < src_val:
                ref_idx += 1
            mapping[bin_idx] = ref_idx / (bins - 1)
        return mapping

    def _cdf_from_tensor(self, tensor, bins):
        hist = torch.histc(tensor, bins=bins, min=0.0, max=1.0)
        cdf = torch.cumsum(hist, dim=0)
        total = torch.clamp(cdf[-1], min=1e-6)
        return cdf / total

    def _apply_histogram_mapping(self, tensor, mapping):
        bins = mapping.shape[0]
        idx = torch.clamp((tensor * (bins - 1)).long(), 0, bins - 1)
        matched = mapping[idx]
        return torch.clamp(matched, 0.0, 1.0)


class WanEx_ContextWindowsPreview:
    """
    Preview how context windows will be laid out for a given configuration.
    Prints window layout to console and returns preview text.
    Pass-through for model and conditioning inputs.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "total_frames": ("INT", {
                    "default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4,
                    "tooltip": "Total video frames (will be converted to latent frames)."
                }),
                "context_length": ("INT", {
                    "default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4,
                    "tooltip": "Context window size in frames."
                }),
                "context_overlap": ("INT", {
                    "default": 30, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 4,
                    "tooltip": "Overlap between windows in frames."
                }),
                "context_schedule": (["standard_static", "batched"], {
                    "default": "standard_static",
                    "tooltip": "Window scheduling strategy."
                }),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Pass-through model input."}),
                "positive": ("CONDITIONING", {"tooltip": "Pass-through. Count used for split preview."}),
                "negative": ("CONDITIONING", {"tooltip": "Pass-through."}),
                "split_conds_to_windows": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Preview how conditioning would be split across windows."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "preview_text")
    FUNCTION = "preview_windows"
    CATEGORY = "WanExperiments"
    OUTPUT_NODE = True

    def _calculate_windows_static(self, num_frames, context_length, context_overlap):
        """Replicate standard_static window calculation."""
        if num_frames <= context_length:
            return [list(range(num_frames))]

        windows = []
        delta = context_length - context_overlap

        for start_idx in range(0, num_frames, delta):
            ending = start_idx + context_length
            if ending >= num_frames:
                # Shift back to maintain context_length
                final_start_idx = start_idx - (ending - num_frames)
                windows.append(list(range(final_start_idx, final_start_idx + context_length)))
                break
            windows.append(list(range(start_idx, start_idx + context_length)))

        return windows

    def _calculate_windows_batched(self, num_frames, context_length):
        """Replicate batched window calculation."""
        if num_frames <= context_length:
            return [list(range(num_frames))]

        windows = []
        for start_idx in range(0, num_frames, context_length):
            windows.append(list(range(start_idx, min(start_idx + context_length, num_frames))))
        return windows

    def _get_region_index(self, index_list, total_frames, num_regions):
        """Calculate which conditioning region a window maps to."""
        center_ratio = (min(index_list) + max(index_list)) / (2 * total_frames)
        region_idx = int(center_ratio * num_regions)
        return min(max(region_idx, 0), num_regions - 1)

    def preview_windows(self, total_frames, context_length, context_overlap, context_schedule,
                        model=None, positive=None, negative=None, split_conds_to_windows=False):

        # Convert to latent frames (WAN 4:1 compression)
        latent_total = max(((total_frames - 1) // 4) + 1, 1)
        latent_context = max(((context_length - 1) // 4) + 1, 1)
        latent_overlap = max(((context_overlap - 1) // 4) + 1, 0)

        # Calculate windows
        if context_schedule == "standard_static":
            windows = self._calculate_windows_static(latent_total, latent_context, latent_overlap)
        elif context_schedule == "batched":
            windows = self._calculate_windows_batched(latent_total, latent_context)
        else:
            windows = self._calculate_windows_static(latent_total, latent_context, latent_overlap)

        # Build preview text
        lines = []
        lines.append("=" * 50)
        lines.append("Context Windows Preview")
        lines.append("=" * 50)
        lines.append(f"Video frames: {total_frames} → Latent frames: {latent_total}")
        lines.append(f"Context: {context_length} frames → {latent_context} latent")
        lines.append(f"Overlap: {context_overlap} frames → {latent_overlap} latent")
        lines.append(f"Schedule: {context_schedule}")

        if context_schedule == "standard_static":
            delta = latent_context - latent_overlap
            lines.append(f"Step size (delta): {delta} latent frames")

        lines.append("")
        lines.append("-" * 50)

        prev_window = None
        for i, window in enumerate(windows):
            win_start = min(window)
            win_end = max(window)
            win_len = len(window)

            overlap_info = ""
            if prev_window is not None:
                overlap_frames = set(prev_window) & set(window)
                if overlap_frames:
                    overlap_info = f" | overlap: {len(overlap_frames)} frames [{min(overlap_frames)}-{max(overlap_frames)}]"

            lines.append(f"Window {i+1}: frames [{win_start}-{win_end}] ({win_len} frames){overlap_info}")
            prev_window = window

        lines.append("-" * 50)
        lines.append(f"Total: {len(windows)} context window(s)")

        # Conditioning split preview
        if split_conds_to_windows and positive is not None:
            num_conds = len(positive)
            if num_conds > 1:
                lines.append("")
                lines.append("=" * 50)
                lines.append("Conditioning Split Preview")
                lines.append("=" * 50)
                lines.append(f"Positive conditionings: {num_conds}")
                lines.append("")

                for i, window in enumerate(windows):
                    center_ratio = (min(window) + max(window)) / (2 * latent_total)
                    region_idx = self._get_region_index(window, latent_total, num_conds)
                    lines.append(f"  Window {i+1} (center: {center_ratio:.2f}) → conditioning [{region_idx}]")
            else:
                lines.append("")
                lines.append("(split_conds_to_windows enabled but only 1 conditioning provided)")

        lines.append("=" * 50)

        preview_text = "\n".join(lines)

        # Print to console
        print(preview_text)

        return (model, positive, negative, preview_text)


NODE_CLASS_MAPPINGS = {
    "WanEx_I2VCustomEmbeds": WanEx_I2VCustomEmbeds,
    "WanEx_BindweaveSubjectToVid": WanEx_BindweaveSubjectToVid,
    "WanEx_QwenVLTextConditioning": WanEx_QwenVLTextConditioning,
    "WanEx_ImageEmbedsPreview": WanEx_ImageEmbedsPreview,
    "WanEx_ConditioningEmbedsPreview": WanEx_ConditioningEmbedsPreview,
    "WanEx_PainterMotionAmplitude": WanEx_PainterMotionAmplitude,
    "WanEx_HuMoImageToVideo": WanEx_HuMoImageToVideo,
    "WanEx_ContextWindowsAdvanced": WanEx_ContextWindowsAdvanced,
    "WanEx_ContextWindowsPreview": WanEx_ContextWindowsPreview,
    "WanEx_VideoColorMatch": WanEx_VideoColorMatch,
    "WanEx_VideoContrastMatch": WanEx_VideoContrastMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanEx_I2VCustomEmbeds": "WanEx I2VCustomEmbeds",
    "WanEx_BindweaveSubjectToVid": "WanEx BindweaveSubjectToVid",
    "WanEx_QwenVLTextConditioning": "WanEx QwenVLTextConditioning",
    "WanEx_ImageEmbedsPreview": "WanEx ImageEmbedsPreview",
    "WanEx_ConditioningEmbedsPreview": "WanEx ConditioningEmbedsPreview",
    "WanEx_PainterMotionAmplitude": "WanEx PainterMotionAmplitude",
    "WanEx_HuMoImageToVideo": "WanEx HuMoImageToVideo",
    "WanEx_ContextWindowsAdvanced": "WanEx Context Windows",
    "WanEx_ContextWindowsPreview": "WanEx Context Windows Preview",
    "WanEx_VideoColorMatch": "WanEx Video Color Match",
    "WanEx_VideoContrastMatch": "WanEx Video Contrast Match",
}
