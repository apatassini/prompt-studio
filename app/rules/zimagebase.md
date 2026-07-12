# Z-Image Base (non-distilled) — Model Rules

Full model: **real CFG 3.0-5.0 (default 4.0)** and a REAL negative prompt `"{NEGATIVE}"`.
Default steps 30 (range 25-50). Natural-language prompts. Best prompt adherence.

## Allowed node types
UNETLoader, CLIPLoader, VAELoader, LoraLoader, CLIPTextEncode, EmptySD3LatentImage,
KSampler, VAEDecode, SaveImage, ModelSamplingAuraFlow (optional shift), LoadImage, VAEEncode.

## Verified base graph (text-to-image)
```json
{
  "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_int8_convrot.safetensors", "weight_dtype": "default"}},
  "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b_fp8_mixed.safetensors", "type": "lumina2", "device": "default"}},
  "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
  "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "{PROMPT}"}},
  "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "{NEGATIVE}"}},
  "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": "{WIDTH}", "height": "{HEIGHT}", "batch_size": 1}},
  "12": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["11", 0], "seed": "{SEED}", "steps": "{STEPS}", "cfg": 4.0, "sampler_name": "euler", "scheduler": "beta", "denoise": 1.0}},
  "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
  "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "{PREFIX}"}}
}
```

## Extensions
- **img2img**: replace node "11" with LoadImage(image="{REF_IMAGE}") -> VAEEncode(pixels, vae ["3",0]),
  KSampler denoise 0.4-0.75.
- **LoRA**: LoraLoader between loaders and consumers (outputs 0=MODEL, 1=CLIP).
