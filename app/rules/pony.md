# CyberRealistic Pony (SDXL) — Model Rules

SDXL checkpoint: cfg 5.0-7.0 (default 5.7), steps 15-30 (default 17). Prompts are
comma-separated Danbooru tags. Positive should start with `score_9, score_8_up, score_7_up`.
Real negative prompt supported; quality negative starts with `score_6, score_5, score_4`.

## Allowed node types
CheckpointLoaderSimple, CLIPSetLastLayer, LoraLoader, CLIPTextEncode, EmptyLatentImage,
KSampler, VAEDecode, VAEEncode, SaveImage, LoadImage, ImageScaleToTotalPixels, LatentUpscale.

## Verified base graph (text-to-image)
```json
{
  "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "cyberrealisticPony_v150.safetensors"}},
  "2": {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["1", 1], "stop_at_clip_layer": -7}},
  "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "score_9, score_8_up, score_7_up, photorealistic, {PROMPT}"}},
  "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "score_6, score_5, score_4, worst quality, low quality, blurry, deformed, bad anatomy, watermark, {NEGATIVE}"}},
  "11": {"class_type": "EmptyLatentImage", "inputs": {"width": "{WIDTH}", "height": "{HEIGHT}", "batch_size": 1}},
  "12": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["11", 0], "seed": "{SEED}", "steps": "{STEPS}", "cfg": 5.7, "sampler_name": "euler", "scheduler": "beta", "denoise": 1.0}},
  "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}},
  "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "{PREFIX}"}}
}
```

## Extensions
- **img2img**: LoadImage(image="{REF_IMAGE}") -> VAEEncode(pixels, vae ["1",2]) replaces node "11";
  KSampler denoise 0.4-0.7.
- **LoRA (SDXL only)**: LoraLoader with model ["1",0] and clip ["2",0]; outputs feed KSampler.model
  and both CLIPTextEncode.clip inputs.
