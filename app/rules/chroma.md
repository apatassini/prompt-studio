# Chroma1-HD — Model Rules

Flux-schnell-based 8.9B, fully uncensored at training level. **Real CFG 3.5-5.0 (default 4.0)**,
real negative prompt `"{NEGATIVE}"`. Default steps 28 (range 20-40). Natural-language prompts,
detailed and explicit wording works well.

## Allowed node types
UnetLoaderGGUF, CLIPLoader, VAELoader, LoraLoader, CLIPTextEncode, EmptySD3LatentImage,
KSampler, VAEDecode, VAEEncode, SaveImage, LoadImage.

## Verified base graph (text-to-image)
```json
{
  "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "Chroma1-HD-Q8_0.gguf"}},
  "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "t5xxl_fp8_e4m3fn_scaled.safetensors", "type": "chroma", "device": "default"}},
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
- **img2img**: LoadImage(image="{REF_IMAGE}") -> VAEEncode(pixels, vae ["3",0]) replaces node "11";
  KSampler denoise 0.4-0.75.
- **LoRA (Chroma/Flux-schnell compatible)**: LoraLoader between loaders and consumers.
