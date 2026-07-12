# FLUX.2 Klein 9B (distilled) — Model Rules

Distilled model: **cfg MUST be 1.0**, negative conditioning MUST be zeroed (ConditioningZeroOut).
`{NEGATIVE}` is NOT usable. Default steps 6 (range 4-10). Natural-language prompts.

## Allowed node types
UnetLoaderGGUF, CLIPLoader, VAELoader, LoraLoader, CLIPTextEncode, ConditioningZeroOut,
CFGGuider, Flux2Scheduler, KSamplerSelect, RandomNoise, EmptyFlux2LatentImage,
SamplerCustomAdvanced, VAEDecode, SaveImage,
LoadImage, ImageScaleToTotalPixels, VAEEncode, ReferenceLatent (for image reference / editing).

## Verified base graph (text-to-image)
```json
{
  "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "flux-2-klein-9b-Q6_K.gguf"}},
  "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_8b_fp8mixed.safetensors", "type": "flux2", "device": "default"}},
  "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
  "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "{PROMPT}"}},
  "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
  "7": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "cfg": 1.0}},
  "8": {"class_type": "Flux2Scheduler", "inputs": {"steps": "{STEPS}", "width": "{WIDTH}", "height": "{HEIGHT}"}},
  "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
  "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": "{SEED}"}},
  "11": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": "{WIDTH}", "height": "{HEIGHT}", "batch_size": 1}},
  "12": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["7", 0], "sampler": ["9", 0], "sigmas": ["8", 0], "latent_image": ["11", 0]}},
  "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
  "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "{PREFIX}"}}
}
```

## Extensions
- **LoRA**: insert LoraLoader between loaders and consumers:
  `{"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["2", 0], "lora_name": "<file>", "strength_model": 1.0, "strength_clip": 1.0}}`
  then use its outputs (0=MODEL into CFGGuider.model, 1=CLIP into CLIPTextEncode.clip).
- **Reference image (image editing)**: per reference:
  LoadImage(image="{REF_IMAGE}") -> ImageScaleToTotalPixels(upscale_method="lanczos", megapixels=1.0)
  -> VAEEncode(pixels, vae ["3",0]) -> ReferenceLatent(conditioning=<positive cond>, latent=<encoded>).
  Chain ReferenceLatent AFTER node "5" and feed its output into BOTH CFGGuider.positive and the
  ConditioningZeroOut input (so the zeroed negative also carries the reference).
