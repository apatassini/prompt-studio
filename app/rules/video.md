# LTX 2.3 — Image-to-Video (motion) rules

Questo target genera un VIDEO a partire da un'immagine (primo frame) + un prompt di **movimento**.
Il modello è LTX 2.3 "distilled": gira a pochi step. Il checkpoint contiene già la VAE (video+audio).

## Come scrivere il prompt (movimento, non descrizione statica)
A differenza delle foto, qui il prompt deve descrivere **cosa succede nel tempo**:
- **Azione del soggetto**: cosa fa, come si muove (es. "she slowly turns her head toward the camera and smiles").
- **Movimento di camera**: "slow dolly-in", "gentle handheld pan left", "static locked-off shot", "orbit around the subject".
- **Dinamica della scena**: vento tra i capelli, luce che cambia, elementi che si muovono sullo sfondo.
- **Ritmo**: di default VELOCITÀ NATURALE. ATTENZIONE: parole come "slowly", "gently",
  "subtle" spingono il modello verso l'effetto RALLENTATORE (tendenza nota di LTX i2v) —
  usale solo se il rallenti è voluto. Per movimenti vivaci: "walks briskly", "turns quickly",
  "energetic". In alternativa c'è l'opzione "Velocità" in UI (conditioning frame-rate).
- Frase unica e continua, in inglese, presente. Evita elenchi di aggettivi statici (quelli valgono per la foto di partenza, non per il video).
- Non ripetere la descrizione fisica del soggetto: quella è già nel primo frame. Concentrati sul MOVIMENTO.

### Esempi di buon prompt di movimento
- "The woman slowly turns her head toward the camera, hair gently moving in the breeze, soft golden light, slow cinematic dolly-in."
- "Static locked-off shot; the man blinks and takes a slow breath, subtle facial expression change, faint steam rising from the coffee cup."
- "Gentle handheld camera drifts left as she walks forward through the crowd, natural motion blur, shallow depth of field."

## Impostazioni consigliate (16GB VRAM, distilled) — agg. 2026-07-11
- Risoluzione di lavoro: **1152×640** (~5 min per 4s) — l'upscale finale lo fa il software
  esterno dell'utente (Topaz), NON il workflow 2K interno (accantonato per scelta).
- **24 fps fissi** (scelta utente). Le opzioni 12/15 esistono in UI ma non sono lo standard.
- Frame: multipli 8n+1 automatici dalla durata; step 8-15 (distilled).
- Primo frame `{REF_IMAGE}`: conditioning strength **0.85** (era 0.7: causava "color pop" —
  frame 0 caldo dalla foto poi palette sbiadita, Δ colore 63 tra frame 0 e 12).
- Prompt video: con "prompt letterale" il testo va al modello VERBATIM (salta l'LLM che
  abbelliva con billowing/cinematic e faceva inventare elementi di scena).

## Nodi disponibili (custom nodes: ComfyUI-LTXVideo, KJNodes, RES4LYF; + nativi)
CheckpointLoaderSimple, LTXAVTextEncoderLoader, GemmaAPITextEncode, LTXVConditioning,
LTXVImgToVideoConditionOnly, EmptyLTXVLatentVideo, LTXVEmptyLatentAudio, LTXVConcatAVLatent,
LTXVScheduler, KSamplerSelect, RandomNoise, SamplerCustomAdvanced, ClownSampler_Beta (RES4LYF),
LTXVSeparateAVLatent, LTXVTiledVAEDecode, LTXVAudioVAEDecode, LTXVAudioVAELoader,
CreateVideo, SaveVideo, LoadImage, ResizeImageMaskNode.

Modelli: checkpoint `ltx-2.3-22b-distilled-1.1.safetensors` (checkpoints/), encoder
`gemma_3_12B_it_fp4_mixed.safetensors` (text_encoders/), upscaler
`ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (latent_upscale_models/).

## Placeholder del workflow
`{PROMPT}` (movimento), `{SEED}`, `{PREFIX}` (obbligatori), `{REF_IMAGE}` (primo frame),
`{WIDTH}`, `{HEIGHT}`, `{STEPS}`.

## Workflow disponibili (verificati end-to-end)
- `video_ltx_i2v.json` — i2v base: `{REF_IMAGE}` = primo frame. 28 nodi.
- `video_ltx_i2v_fine.json` — i2v con **inizio+fine**: `{REF_IMAGE}` = primo frame,
  `{REF_IMAGE2}` = ultimo frame (LTXVAddGuideAdvanced frame_idx=-1 strength 1.0 +
  LTXVCropGuides prima del decode). Usalo per la coerenza del personaggio: l'identita'
  viene "ancorata" a entrambi gli estremi. 31 nodi.
- `video_ltx_i2v_likeness.json` — i2v con **Likeness Guide** (nodi 10S): il volto del
  primo frame (`{REF_IMAGE}`) viene rilevato (LTXFaceDetector) e iniettato nel
  conditioning come riferimento silenzioso (silent_reference + bbox_softfade) per tutto
  il video. Contro il "character drift" nei movimenti ampi. 30 nodi.
- `video_ltx_i2v_2k.json` — i2v con **upscale 2K**: prima passata normale, poi upscale
  latente x2 (LTXVLatentUpsamplerTiled) + ri-ancoraggio primo frame + refine 3 step con
  LTXTiledSampler (audio pass-1 congelato via SolidMask 0). Output = 2x {WIDTH}x{HEIGHT}
  (es. 768x448 -> 1536x896). Piu' lento; il tiling evita l'hue-shift e regge in 16GB. 41 nodi.

## Note tecniche (imparate dal sorgente, non cambiare senza motivo)
- Il guide dell'ultimo frame va applicato sul latente SOLO-video, PRIMA di LTXVConcatAVLatent
  (il core solleva errore sui latenti AV combinati).
- LTXVAddGuide APPENDE frame-guida al latente: dopo il sampling servono LTXVCropGuides
  (sul latente video separato) o il decode include il frame guida.
- frame_idx negativo = contato dalla fine; per video 9+ frame dev'essere divisibile per 8
  (il nodo arrotonda da solo).
- La strength di LTXVAddGuideAdvanced DEVE restare 1.0: valori diversi attivano il percorso
  "guide attention mask" del core che crasha col wrapper STG del MultimodalGuider
  (RuntimeError 4368 vs 4704 token). 1.0 = frame ancorato esatto.
- L'ULTIMO FRAME deve essere una continuazione PLAUSIBILE della stessa scena (stesso sfondo,
  posa raggiungibile col movimento nel tempo dato): un target "impossibile" (es. foto
  specchiata = sfondo invertito) produce un arrivo in dissolvenza/ghosting sugli ultimi
  frame. Uso ideale: frame estratto da un video precedente della stessa scena (concatenare
  clip) o variante generata della stessa inquadratura.
- reference_mask_mode della LikenessGuide: bbox_softfade. NON usare whole_frame (provato:
  palette che oscilla/flicker Δ25-30 a meta' clip).
- Il checkpoint distilled-1.1 incorpora la LoRA di distillazione: NON aggiungere
  ltx-2.3-22b-distilled-lora (peggiora).
- SaveVideo riporta l'output sotto la chiave "images" nello history (non "videos").
