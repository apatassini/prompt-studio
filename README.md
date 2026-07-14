# Prompt Studio

A local web app that turns plain-language descriptions into **coherent photo sets and image-to-video clips**, on top of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [llama.cpp](https://github.com/ggml-org/llama.cpp). Everything runs on your own machine.

You write a *setting* for each tab; a local LLM writes and self-repairs the image prompts in English; ComfyUI renders the photos (or animates them into video); an optional local vision model scores the results so you can filter the good ones. The UI is **bilingual (English / Italian)** with a toggle in the header.

> **Made by [Persoft](https://persoft.it)** — custom software & AI solutions. Contact: [info@persoft.it](mailto:info@persoft.it) · <https://persoft.it>

<p align="center">
  <img src="docs/example-hiking.jpg" width="42%" alt="AI-generated hiking photo produced with Prompt Studio">
  &nbsp;&nbsp;
  <img src="docs/example-camp.jpg" width="42%" alt="AI-generated camping photo produced with Prompt Studio">
</p>
<p align="center"><sub>Two chapters of a chronological "photo-director" story, generated locally with Prompt Studio (FLUX.2 Klein + a character LoRA). The person shown is an AI-generated persona.</sub></p>

> ⚠️ **Adult content.** Prompt Studio can be used to generate explicit imagery. Use it responsibly and legally. You are solely responsible for what you generate and for complying with the license of every model you download (see *Third-party components* below). Do not use it to create illegal content or non-consensual imagery of real people.

---

## Features

- **Multi-tab prompt generation** — one *setting* per tab, an LLM writes N English prompts; forced elements in `[square brackets]` are guaranteed in every image.
- **Photo & Video modes** — photos, or image-to-video (LTX 2.3): *Director* (start/end photos generated for you), *Photo + text*, *Two photos*.
- **Persona configurator** — build a character (16 traits + presets); the description is enforced in every photo.
- **Photo Director (story)** — turn N photos into chronological chapters of one story.
- **AI quality judge** — a local vision model (Qwen3-VL) scores each photo 1–10 (anatomy, physics, scene coherence, prompt fidelity), with an optional adversarial second pass and your own custom criteria.
- **Style chips, LoRA support, custom ComfyUI workflows**, saved characters, right-click regenerate / load-to-editor / animate.
- **Built-in model manager** — downloads engines and models with progress, resume, and a first-run setup guide.
- **Bilingual UI (EN/IT)** — default English, one-click toggle, remembers your choice.
- **Fully local** — nothing leaves your PC except the model downloads.

## Requirements

- Windows 10/11 (64-bit)
- NVIDIA GPU with ≥12 GB VRAM (16 GB recommended) + a recent driver
- 16 GB RAM (40+ GB to keep the Qwen agent resident)
- ~25 GB for the install + 9–22 GB per AI model
- [ComfyUI (portable)](https://github.com/comfyanonymous/ComfyUI) and [llama.cpp](https://github.com/ggml-org/llama.cpp) — installed automatically by the built-in downloader

## Install

**Easy path (recommended).** From the [**latest release**](../../releases/latest), download **`bootstrap.zip`** (~11 MB — mini-Python + the app + launchers), extract it anywhere, and run `ComfyUI_windows_portable/Avvia_Prompt_Studio.bat`. On first launch the **Models** window opens and downloads the 4 essentials (image engine, text engine, an image model, the prompt writer), with progress and resume. See [`installer/LEGGIMI.txt`](installer/LEGGIMI.txt).
_(The release also ships `app.zip` — just the UI files — for updating an existing install; but the app updates itself, see below.)_

**From source (developers).** This repository contains the source only (no binaries or models). Place the [`app/`](app) folder into your ComfyUI portable tree as `ComfyUI_windows_portable/LLM/app`, copy the [`launchers/`](launchers) `.bat` files next to `ComfyUI_windows_portable`, then run `Avvia_Prompt_Studio.bat`. The app serves the UI at <http://127.0.0.1:8500> and drives ComfyUI over its HTTP API.

## Usage

- **Start:** `Avvia_Prompt_Studio.bat` → opens <http://127.0.0.1:8500>
- **Stop:** `Arresta_Prompt_Studio.bat`

## Auto-update

Prompt Studio can update itself from this GitHub repository's **Releases**.

- Open the **Models** window → **Updates** section. Enter your repo as `owner/name` and press **Save**.
- The app checks for a newer release on startup and when you open that window. If one is found, an **Update** badge appears in the header. Click **Update now** and the app downloads the release, backs up the current files, replaces the code (your settings, saved characters and job history are **preserved**), and restarts on its own — the page reloads when it is back.
- Enable **Auto-update on startup** to apply new versions automatically at launch (skipped while jobs are running).

**Releasing a new version (maintainers).** Bump `APP_VERSION` in [`app/server.py`](app/server.py), commit, then create a **GitHub Release** with a tag like `v1.1.0`. The app compares the running `APP_VERSION` with the latest release tag and offers the update. It uses the release's auto-generated source archive by default (no manual asset needed); optionally attach an `app.zip` asset (the app source at the archive root) to override it.

## Architecture

| Path | What it is |
|------|-----------|
| [`app/server.py`](app/server.py) | Backend HTTP server (Python standard library only). A **client** of ComfyUI's HTTP/WebSocket API and of llama.cpp — it does not import any ComfyUI module. |
| [`app/static/index.html`](app/static/index.html) | Single-file bilingual UI (EN/IT i18n). |
| [`app/rules/*.md`](app/rules) | Per-model guidance for the prompt-writing LLM. |
| [`app/workflows/*.json`](app/workflows) | ComfyUI graphs (API format) with `{PROMPT}`/`{SEED}`/`{WIDTH}`… placeholders. |

## Privacy

This repository intentionally contains **source code only**. It excludes all personal data and user content via [`.gitignore`](.gitignore): saved characters/presets, job history, generated media, LoRAs, downloaded models, and the ComfyUI runtime. Nothing you generate is ever committed.

## Third-party components & model licenses

Prompt Studio **orchestrates** third-party software and **downloads** third-party models at install time; it does not redistribute them. Each is governed by **its own license**, and you must comply with all of them:

- **ComfyUI** — GPL-3.0. Prompt Studio talks to it only over its network API (separate process), so it is not a derivative work of ComfyUI.
- **llama.cpp** — MIT.
- **AI models** (e.g. FLUX.2 Klein, Z-Image, Chroma, LTX-2.3, Qwen, Mistral-Nemo) — each has **its own license**, and **some are non-commercial or otherwise restricted**. Read and respect the license of every model before use, especially for any commercial purpose.

## Author

Designed and developed by **[Persoft](https://persoft.it)** — custom software and AI solutions.

- 🌐 Website: <https://persoft.it>
- ✉️ Contact: [info@persoft.it](mailto:info@persoft.it)

If Prompt Studio is useful to you, a ⭐ on the repository is appreciated. For custom development, integrations or consulting, get in touch.

## License

Licensed under the **GNU General Public License v3.0** — see [`LICENSE`](LICENSE).

---

## In breve (Italiano)

**Prompt Studio** è un'app web **locale** che trasforma descrizioni in linguaggio naturale in **set di foto coerenti e clip video** (image-to-video), appoggiandosi a ComfyUI e llama.cpp. Scrivi un'ambientazione per scheda, un LLM locale scrive/ripara i prompt in inglese, ComfyUI genera, e un giudice IA opzionale dà un voto alle foto. **Interfaccia bilingue IT/EN** con interruttore nell'header (default inglese). Tutto gira in locale. Avvio: `Avvia_Prompt_Studio.bat` → <http://127.0.0.1:8500>.

> ⚠️ L'app può generare contenuti espliciti: usala in modo lecito e responsabile. Rispetta la licenza di ogni modello che scarichi (alcuni sono non commerciali). Licenza del codice: **GPL-3.0**.
