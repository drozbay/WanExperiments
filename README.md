# WanExperiments Nodes (very experimental!)

These are quick partially vibe-coded nodes that I use for testing things for Wan in ComfyUI. Expect bugs, weird behavior, and random changes.

## Nodes
- **WanEx_I2VCustomEmbeds** – feed your own latents/masks into Wan I2V runs. A more advanced version of WanImageToVideo.
- **WanEx_BindweaveSubjectToVid** – Basic implementation of BindWeave model natively in ComfyUI with refs, masks, Qwen, and CLIP.
- **WanEx_QwenVLTextConditioning** – Plain QwenVL text-only encoder specifically for matching the QwenVL negative conditioning of Bindweave.
- **WanEx_ImageEmbedsPreview** – Inspect ComfyUI-WanVideoWrapper image_embeds latents and masks.
- **WanEx_ConditioningEmbedsPreview** – Inspect embedded latents and masks in native Conditioning pipes.
- **WanEx_PainterMotionAmplitude** – PainterI2V math (from [ComfyUI-PainterI2V](https://github.com/princepainter/ComfyUI-PainterI2V)), but just the math that does the noise modulation for I2V embeddings. Can be used on any Conditioning lines that have I2V latents applied, after nodes like WanEx_I2VCustomEmbeds or WanImageToVideo. Also allows you to use masks to mask the noise application.
- **WanEx_HuMoImageToVideo** – Extends capabilities compared to WanHuMoImageToVideo. Allows for providing start/end images in the node directly for convenience (could also use the I2VCustomEmbeds node for more control) and critically allows for providing a batch of reference images instead of only one.

License: GNU GPLv3
PainterI2V code licensed under MIT License (c) princepainter