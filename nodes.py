import nodes
import comfy.utils
import comfy.model_management
import comfy.clip_vision
import node_helpers
import torch

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

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING", "LATENT", "MASK")
    RETURN_NAMES = ("positive", "negative", "latent", "debug_info", "concat_latent_preview", "concat_mask_preview")
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
            mask = torch.zeros(
                (1, pixel_frames, latent_height, latent_width),
                device=start_image.device,
                dtype=start_image.dtype
            )

            frames_with_image = min(start_image.shape[0], pixel_frames)
            mask[:, :frames_with_image] = 1.0

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
        mask_preview = final_concat_mask.squeeze(0).permute(1, 0, 2, 3).reshape(-1, final_concat_mask.shape[3], final_concat_mask.shape[4])

        return (positive, negative, out_latent, debug_text, concat_latent_preview, mask_preview)
    

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

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING", "LATENT", "MASK")
    RETURN_NAMES = ("positive", "negative", "latent", "debug_info", "concat_latent_preview", "concat_mask_preview")
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

            pixel_frames = (temporal_latent - 1) * 4 + 1
            is_pixel_temporal = (t_m == pixel_frames or t_m == length)

            if is_pixel_temporal:
                debug_info.append(f"  Detected pixel-space temporal: {t_m} frames → converting to latent-space")

                if h_m != latent_height or w_m != latent_width:
                    i2v_masks = comfy.utils.common_upscale(
                        i2v_masks.view(-1, h_m, w_m).unsqueeze(1),
                        latent_width, latent_height, "bilinear", "center"
                    ).squeeze(1).view(b_m, t_m, latent_height, latent_width)

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
            pixel_frames = (temporal_latent - 1) * 4 + 1
            mask = torch.zeros(
                (1, pixel_frames, latent_height, latent_width),
                device=i2v_images.device,
                dtype=i2v_images.dtype
            )

            mask[:, :frames_with_start_image] = 1.0

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
        # full_mask shape: [1, 4, T, H, W] - take first batch, first channel for preview
        concat_mask_preview = full_mask[0, 0].float()  # [T, H, W]

        return (positive, negative, out_latent, debug_text, concat_latent_preview, concat_mask_preview)

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


NODE_CLASS_MAPPINGS = {
    "WanEx_I2VCustomEmbeds": WanEx_I2VCustomEmbeds,
    "WanEx_BindweaveSubjectToVid": WanEx_BindweaveSubjectToVid,
    "WanEx_QwenVLTextConditioning": WanEx_QwenVLTextConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanEx_I2VCustomEmbeds": "WanEx I2VCustomEmbeds",
    "WanEx_BindweaveSubjectToVid": "WanEx BindweaveSubjectToVid",
    "WanEx_QwenVLTextConditioning": "WanEx QwenVLTextConditioning",
}
