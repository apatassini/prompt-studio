# -*- coding: utf-8 -*-
"""
Prompt Studio - genera prompt con LLM locale (llama.cpp) e immagini con ComfyUI.
Multi-tab: ogni scheda ha la sua descrizione; i prompt si generano tutti insieme,
poi la creazione delle foto si avvia subito o si programma a un orario.
Solo libreria standard Python. Avvio: python server.py  ->  http://127.0.0.1:8500
"""
import base64
import hashlib
import io
import json
import os
import re
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.parse
import urllib.error
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ------------------------------------------------------------------ config
APP_PORT = 8500
LLAMA_PORT = 8600
COMFY_URL = "http://127.0.0.1:8188"

# Versione dell'app: confrontata col tag dell'ultima Release su GitHub per l'auto-aggiornamento.
# Per pubblicare una nuova versione: alza APP_VERSION, committa, crea una Release con tag "vX.Y.Z".
APP_VERSION = "1.0.0"
# Repository GitHub da cui scaricare gli aggiornamenti ("utente/repo"). Override in config.json
# ("github_repo") o dalla finestra "Modelli" nell'interfaccia. Vuoto = auto-aggiornamento disattivato.
GITHUB_REPO_DEFAULT = ""
LLAMA_URL = f"http://127.0.0.1:{LLAMA_PORT}"

# Percorsi relativi alla posizione di questo file: <root>\LLM\app\server.py
# Tutta la cartella <root> (ComfyUI_windows_portable) e' spostabile/portabile.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_DIR = os.path.dirname(APP_DIR)
PORTABLE_ROOT = os.path.dirname(LLM_DIR)
LLAMA_SERVER_EXE = os.path.join(LLM_DIR, "llamacpp", "llama-server.exe")
MODELS_DIR = os.path.join(LLM_DIR, "models")
PREFERRED_MODEL = "Mistral-Nemo-Instruct-2407-abliterated.Q5_K_M.gguf"
NEMO_TEMPLATE = os.path.join(MODELS_DIR, "mistral-nemo.jinja")


def list_models():
    out = []
    if os.path.isdir(MODELS_DIR):
        for fn in sorted(os.listdir(MODELS_DIR)):
            fp = os.path.join(MODELS_DIR, fn)
            if not fn.lower().endswith(".gguf"):
                continue
            # esclude file troppo piccoli o ancora in download (modificati di recente)
            if os.path.getsize(fp) < 1_000_000_000 or time.time() - os.path.getmtime(fp) < 120:
                continue
            out.append(fn)
    return out


def default_model():
    models = list_models()
    if PREFERRED_MODEL in models:
        return PREFERRED_MODEL
    return models[0] if models else None
COMFY_ROOT = os.path.join(PORTABLE_ROOT, "ComfyUI")
COMFY_OUTPUT_DIR = os.path.join(COMFY_ROOT, "output")
LORA_DIR = os.path.join(COMFY_ROOT, "models", "loras")
STATIC_DIR = os.path.join(APP_DIR, "static")
DEFAULT_DEST = os.path.join(LLM_DIR, "output")

UNET_NAME = "flux-2-klein-9b-Q6_K.gguf"
CLIP_NAME = "qwen_3_8b_fp8mixed.safetensors"
VAE_NAME = "flux2-vae.safetensors"

# Modelli foto selezionabili per scheda (impostazioni ricavate dai workflow verificati)
IMAGE_MODELS = {
    "klein":      {"label": "FLUX.2 Klein 9B (LoRA ok)",  "steps": 6},
    "zimage":     {"label": "Z-Image Turbo (veloce)",      "steps": 11},
    "zimagebase": {"label": "Z-Image Base (piu' fedele)",  "steps": 30},
    "pony":       {"label": "CyberRealistic Pony (SDXL)",  "steps": 17},
    "chroma":     {"label": "Chroma1-HD (esplicito)",      "steps": 28},
}
CHROMA_NEGATIVE = "blurry, low quality, distorted, deformed, bad anatomy, watermark, text"
ZIMAGE_BASE_NEGATIVE = "blurry, low quality, distorted, deformed, bad anatomy, watermark, text"
# Anatomia corretta: iniettata in ogni foto quando l'opzione e' attiva (default).
# Positivo (per tutti i modelli, anche Klein che ignora il negative) + negativo (per chi lo usa).
ANATOMY_POSITIVE = ("anatomically correct, natural realistic body proportions, well-formed hands "
                    "with five fingers, correct number of limbs, coherent natural pose")
ANATOMY_NEGATIVE = ("deformed, disfigured, bad anatomy, malformed, mutated, extra limbs, missing "
                    "limbs, extra arms, extra legs, extra fingers, missing fingers, fused fingers, "
                    "mutated hands, bad hands, distorted body, unnatural pose, disproportionate body")
RULES_DIR = os.path.join(APP_DIR, "rules")
WORKFLOWS_DIR = os.path.join(APP_DIR, "workflows")
AGENT_MODEL_PATTERN = "qwen3.6"   # se presente in models/, l'agente usa questo
EVAL_MODEL_PATTERN = "qwen3-vl"   # VLM dedicato alla VALUTAZIONE immagini (giudice), separato dall'agente

# ------------------------------------------------------------------ catalogo modelli scaricabili
# Percorsi relativi a PORTABLE_ROOT. Usato dalla sezione "Modelli" della UI (installer leggero).
_HF = "https://huggingface.co"
MODEL_CATALOG = {
    "klein": {
        "label": "FLUX.2 Klein 9B", "kind": "Immagini", "size_gb": 16.3,
        "desc": "Veloce (6 step), supporta le LoRA. Il tuttofare.",
        "files": [
            ("ComfyUI/models/diffusion_models/flux-2-klein-9b-Q6_K.gguf",
             _HF + "/unsloth/FLUX.2-klein-9B-GGUF/resolve/main/flux-2-klein-9b-Q6_K.gguf"),
            ("ComfyUI/models/text_encoders/qwen_3_8b_fp8mixed.safetensors",
             _HF + "/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors"),
            ("ComfyUI/models/vae/flux2-vae.safetensors",
             _HF + "/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/vae/flux2-vae.safetensors"),
        ]},
    "zimagebase": {
        "label": "Z-Image Base", "kind": "Immagini", "size_gb": 11.3,
        "desc": "Il piu' fedele al prompt: CFG e negative reali (30 step).",
        "files": [
            ("ComfyUI/models/diffusion_models/z_image_int8_convrot.safetensors",
             _HF + "/Comfy-Org/z_image/resolve/main/split_files/diffusion_models/z_image_int8_convrot.safetensors"),
            ("ComfyUI/models/text_encoders/qwen_3_4b_fp8_mixed.safetensors",
             _HF + "/Comfy-Org/z_image/resolve/main/split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors"),
            ("ComfyUI/models/vae/ae.safetensors",
             _HF + "/Comfy-Org/z_image/resolve/main/split_files/vae/ae.safetensors"),
        ]},
    "chroma": {
        "label": "Chroma1-HD", "kind": "Immagini", "size_gb": 14.4,
        "desc": "Decensurato all'origine, il migliore per contenuti espliciti (28 step).",
        "files": [
            ("ComfyUI/models/diffusion_models/Chroma1-HD-Q8_0.gguf",
             _HF + "/silveroxides/Chroma1-HD-GGUF/resolve/main/Chroma1-HD-Q8_0.gguf"),
            ("ComfyUI/models/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors",
             _HF + "/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors"),
            ("ComfyUI/models/vae/ae.safetensors",
             _HF + "/Comfy-Org/z_image/resolve/main/split_files/vae/ae.safetensors"),
        ]},
    "llm_nemo": {
        "label": "Mistral Nemo 12B (senza filtri)", "kind": "LLM", "size_gb": 8.7,
        "desc": "Scrive i prompt delle foto, anche espliciti. Consigliato.",
        "files": [
            ("LLM/models/Mistral-Nemo-Instruct-2407-abliterated.Q5_K_M.gguf",
             _HF + "/mradermacher/Mistral-Nemo-Instruct-2407-abliterated-GGUF/resolve/main/Mistral-Nemo-Instruct-2407-abliterated.Q5_K_M.gguf"),
        ]},
    "llm_qwen": {
        "label": "Qwen3.6-35B MoE (agente)", "kind": "LLM", "size_gb": 21.6,
        "desc": "Il cervello dell'agente workflow: vede le immagini e programma. Serve molta RAM (40GB+).",
        "files": [
            ("LLM/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
             _HF + "/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
            ("LLM/models/Qwen3.6-35B-A3B-mmproj-F16.gguf",
             _HF + "/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/mmproj-F16.gguf"),
        ]},
    "llm_eval": {
        "label": "Qwen3-VL 8B (giudice foto)", "kind": "LLM", "size_gb": 5.8,
        "desc": "Il modello che VALUTA le foto generate: vede l'immagine, assegna un voto, "
                "filtra gli scarti e fa il contraddittorio. Serve per la valutazione automatica.",
        "files": [
            ("LLM/models/Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
             _HF + "/unsloth/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3-VL-8B-Instruct-Q4_K_M.gguf"),
            ("LLM/models/Qwen3-VL-8B-Instruct-mmproj-F16.gguf",
             _HF + "/unsloth/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-F16.gguf"),
        ]},
    # --- LTX 2.3 video (image-to-video). Il checkpoint "distilled" e' autonomo: include
    #     la VAE (video+audio) e gira a pochi step. Encoder Gemma e upscaler separati.
    "ltx_checkpoint": {
        "label": "LTX 2.3 Video (distilled)", "kind": "Video", "size_gb": 43.0,
        "desc": "Il motore video LTX 2.3 (image-to-video). Contiene la VAE. Gira a pochi step. "
                "Su 16GB VRAM va in offload nella RAM (lento in locale, veloce su GPU cloud).",
        "files": [
            ("ComfyUI/models/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors",
             _HF + "/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-1.1.safetensors"),
        ]},
    "ltx_encoder": {
        "label": "LTX 2.3 encoder Gemma (fp4)", "kind": "Video", "size_gb": 8.8,
        "desc": "Il text encoder Gemma 3 12B (fp4) che LTX usa per capire il prompt di movimento.",
        "files": [
            ("ComfyUI/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
             _HF + "/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"),
        ]},
    "ltx_upscaler": {
        "label": "LTX 2.3 upscaler spaziale x2", "kind": "Video", "size_gb": 0.93,
        "desc": "Upscaler latente x2 per portare il video a risoluzione piu' alta (tiled, salva VRAM).",
        "files": [
            ("ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
             _HF + "/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
        ]},
}
# file richiesti dai modelli non a catalogo (per il flag "installato" nella UI)
IMAGE_MODEL_FILES = {
    "klein": [f[0] for f in MODEL_CATALOG["klein"]["files"]],
    "zimagebase": [f[0] for f in MODEL_CATALOG["zimagebase"]["files"]],
    "chroma": [f[0] for f in MODEL_CATALOG["chroma"]["files"]],
    "zimage": ["ComfyUI/models/diffusion_models/z_image_turbo_bf16.safetensors",
               "ComfyUI/models/text_encoders/qwen_3_4b_fp8_mixed.safetensors",
               "ComfyUI/models/vae/ae.safetensors"],
    "pony": ["ComfyUI/models/checkpoints/cyberrealisticPony_v150.safetensors"],
}

# Componenti di programma scaricabili (per l'installer bootstrap: interfaccia subito,
# runtime ComfyUI/PyTorch e motore LLM scaricati da fonti ufficiali alla prima esecuzione)
COMPONENT_CATALOG = {
    "runtime": {
        "label": "ComfyUI + PyTorch (runtime immagini)", "kind": "Programma", "size_gb": 2.0,
        "desc": "Il motore che genera le immagini (release ufficiale v0.27.0, GitHub). "
                "Obbligatorio. Al termine riavvia Prompt Studio dall'icona.",
        "check_files": ["python_embeded/python.exe", "ComfyUI/main.py",
                        "ComfyUI/custom_nodes/ComfyUI-GGUF/nodes.py"],
        "steps": [
            ("dl", "_dl/comfyui_portable.7z",
             "https://github.com/comfyanonymous/ComfyUI/releases/download/v0.27.0/ComfyUI_windows_portable_nvidia.7z"),
            ("extract_parent", "_dl/comfyui_portable.7z"),
            ("dl", "_dl/gguf_node.zip",
             "https://github.com/city96/ComfyUI-GGUF/archive/refs/heads/main.zip"),
            ("extract_to", "_dl/gguf_node.zip", "ComfyUI/custom_nodes"),
            ("rename", "ComfyUI/custom_nodes/ComfyUI-GGUF-main", "ComfyUI/custom_nodes/ComfyUI-GGUF"),
            ("pip", ["gguf", "websocket-client"]),
            ("strip_templates",),   # via i ~366MB di video/immagini demo della galleria
            ("del", "_dl/comfyui_portable.7z"),
            ("del", "_dl/gguf_node.zip"),
            ("start_comfy",),
        ]},
    "llm_engine": {
        "label": "Motore LLM (llama.cpp)", "kind": "Programma", "size_gb": 0.6,
        "desc": "Fa girare i modelli di testo (prompt e agente). Obbligatorio per i prompt automatici.",
        "check_files": ["LLM/llamacpp/llama-server.exe", "LLM/llamacpp/cudart64_13.dll"],
        "steps": [
            ("dl", "_dl/llama.zip",
             "https://github.com/ggml-org/llama.cpp/releases/download/b9928/llama-b9928-bin-win-cuda-13.3-x64.zip"),
            ("dl", "_dl/cudart.zip",
             "https://github.com/ggml-org/llama.cpp/releases/download/b9928/cudart-llama-bin-win-cuda-13.3-x64.zip"),
            ("extract_to", "_dl/llama.zip", "LLM/llamacpp"),
            ("extract_to", "_dl/cudart.zip", "LLM/llamacpp"),
            ("del", "_dl/llama.zip"),
            ("del", "_dl/cudart.zip"),
        ]},
    "video_nodes": {
        "label": "Nodi video (LTX-Video + KJNodes + RES4LYF)", "kind": "Programma", "size_gb": 0.3,
        "desc": "I nodi ComfyUI necessari per generare video con LTX 2.3. Al termine riavvia ComfyUI.",
        "check_files": ["ComfyUI/custom_nodes/ComfyUI-LTXVideo/__init__.py",
                        "ComfyUI/custom_nodes/ComfyUI-KJNodes/__init__.py",
                        "ComfyUI/custom_nodes/RES4LYF/__init__.py"],
        "steps": [
            ("dl", "_dl/ltxvideo.zip",
             "https://github.com/Lightricks/ComfyUI-LTXVideo/archive/refs/heads/master.zip"),
            ("extract_to", "_dl/ltxvideo.zip", "ComfyUI/custom_nodes"),
            ("rename", "ComfyUI/custom_nodes/ComfyUI-LTXVideo-master",
             "ComfyUI/custom_nodes/ComfyUI-LTXVideo"),
            ("pip_req", "ComfyUI/custom_nodes/ComfyUI-LTXVideo/requirements.txt"),
            ("dl", "_dl/kjnodes.zip",
             "https://github.com/kijai/ComfyUI-KJNodes/archive/refs/heads/main.zip"),
            ("extract_to", "_dl/kjnodes.zip", "ComfyUI/custom_nodes"),
            ("rename", "ComfyUI/custom_nodes/ComfyUI-KJNodes-main",
             "ComfyUI/custom_nodes/ComfyUI-KJNodes"),
            ("pip_req", "ComfyUI/custom_nodes/ComfyUI-KJNodes/requirements.txt"),
            ("dl", "_dl/res4lyf.zip",
             "https://github.com/ClownsharkBatwing/RES4LYF/archive/refs/heads/main.zip"),
            ("extract_to", "_dl/res4lyf.zip", "ComfyUI/custom_nodes"),
            ("rename", "ComfyUI/custom_nodes/RES4LYF-main", "ComfyUI/custom_nodes/RES4LYF"),
            ("pip_req", "ComfyUI/custom_nodes/RES4LYF/requirements.txt"),
            ("del", "_dl/ltxvideo.zip"),
            ("del", "_dl/kjnodes.zip"),
            ("del", "_dl/res4lyf.zip"),
            ("start_comfy",),
        ]},
}

DOWNLOADS = {}   # key catalogo -> stato download
downloads_lock = threading.Lock()

# Ruolo di ogni voce per la guida di primo avvio: gruppo, ordine, se fa parte del
# "kit minimo per iniziare". I gruppi guidano l'utente: base (motori obbligatori),
# image (almeno un modello immagini), text (chi scrive i prompt), optional.
CATALOG_ROLE = {
    "runtime":    {"group": "base",     "order": 1, "starter": True},
    "llm_engine": {"group": "base",     "order": 2, "starter": True},
    "klein":      {"group": "image",    "order": 3, "starter": True},
    "zimagebase": {"group": "image",    "order": 4, "starter": False},
    "chroma":     {"group": "image",    "order": 5, "starter": False},
    "llm_nemo":   {"group": "text",     "order": 6, "starter": True},
    "video_nodes":    {"group": "video",    "order": 7,  "starter": False},
    "ltx_checkpoint": {"group": "video",    "order": 8,  "starter": False},
    "ltx_encoder":    {"group": "video",    "order": 9,  "starter": False},
    "ltx_upscaler":   {"group": "video",    "order": 10, "starter": False},
    "llm_eval":   {"group": "optional", "order": 11, "starter": False},
    "llm_qwen":   {"group": "optional", "order": 12, "starter": False},
}
# kit minimo per creare la prima foto: motori + un modello immagini + scrittore prompt
STARTER_SET = ["runtime", "llm_engine", "klein", "llm_nemo"]


def any_image_model_installed():
    return any(image_model_installed(k) for k in IMAGE_MODEL_FILES)


def essentials_missing():
    """Cosa manca per poter creare foto. Lista di chiavi 'logiche' ancora da installare."""
    need = []
    if not component_installed("runtime"):
        need.append("runtime")
    if not component_installed("llm_engine"):
        need.append("llm_engine")
    if not any_image_model_installed():
        need.append("image")
    if not model_installed("llm_nemo"):
        need.append("llm_nemo")
    return need


def files_installed(rel_files):
    return all(os.path.exists(os.path.join(PORTABLE_ROOT, r)) for r in rel_files)


def model_installed(key):
    return files_installed([f[0] for f in MODEL_CATALOG[key]["files"]])


def image_model_installed(key):
    rels = IMAGE_MODEL_FILES.get(key)
    return files_installed(rels) if rels else True


def _download_file(st, url, dest):
    """Scarica un file con progresso/velocita' in st, con ripresa da .part."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    resume = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"User-Agent": "PromptStudio/1.0"}
    if resume:
        headers["Range"] = f"bytes={resume}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        total = resume + int(r.headers.get("Content-Length") or 0)
        st["total"] = total
        st["done"] = resume
        t_last, b_last = time.time(), resume
        with open(part, "ab" if resume else "wb") as f:
            while True:
                if st["cancel"]:
                    raise RuntimeError("annullato")
                chunk = r.read(512 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                st["done"] += len(chunk)
                now = time.time()
                if now - t_last >= 1.0:
                    st["speed"] = (st["done"] - b_last) / (now - t_last)
                    t_last, b_last = now, st["done"]
    os.replace(part, dest)
    # retrodata il file cosi' il filtro anti-download-incompleti non lo nasconde
    old = time.time() - 600
    os.utime(dest, (old, old))


def download_model_worker(key):
    st = DOWNLOADS[key]
    files = MODEL_CATALOG[key]["files"]
    try:
        for i, (rel, url) in enumerate(files):
            dest = os.path.join(PORTABLE_ROOT, rel.replace("/", os.sep))
            st["file"] = os.path.basename(rel)
            st["file_n"] = i + 1
            st["files_total"] = len(files)
            if os.path.exists(dest):
                continue
            _download_file(st, url, dest)
        st["status"] = "done"
    except Exception as e:
        st["status"] = "cancelled" if st.get("cancel") else "error"
        st["error"] = str(e)[:300]


def component_installed(key):
    return files_installed(COMPONENT_CATALOG[key]["check_files"])


def component_worker(key):
    """Esegue i passi di installazione di un componente (download, estrazione, pip...)."""
    st = DOWNLOADS[key]
    steps = COMPONENT_CATALOG[key]["steps"]
    dls = [s for s in steps if s[0] == "dl"]
    try:
        dl_n = 0
        for step in steps:
            if st["cancel"]:
                raise RuntimeError("annullato")
            op = step[0]
            if op == "dl":
                dl_n += 1
                _, rel, url = step
                dest = os.path.join(PORTABLE_ROOT, rel.replace("/", os.sep))
                st["file"] = os.path.basename(rel)
                st["file_n"] = dl_n
                st["files_total"] = len(dls)
                st["phase"] = "scarico"
                if not os.path.exists(dest):
                    _download_file(st, url, dest)
            elif op in ("extract_parent", "extract_to"):
                st["phase"] = "estraggo (alcuni minuti)"
                archive = os.path.join(PORTABLE_ROOT, step[1].replace("/", os.sep))
                if op == "extract_parent":
                    dest_dir = os.path.dirname(PORTABLE_ROOT)
                else:
                    dest_dir = os.path.join(PORTABLE_ROOT, step[2].replace("/", os.sep))
                    os.makedirs(dest_dir, exist_ok=True)
                r = subprocess.run(["tar", "-xf", archive, "-C", dest_dir],
                                   capture_output=True, text=True, timeout=3600,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                if r.returncode != 0:
                    raise RuntimeError("estrazione fallita: " + (r.stderr or "")[:200])
            elif op == "rename":
                src = os.path.join(PORTABLE_ROOT, step[1].replace("/", os.sep))
                dst = os.path.join(PORTABLE_ROOT, step[2].replace("/", os.sep))
                if os.path.exists(src) and not os.path.exists(dst):
                    os.replace(src, dst)
            elif op == "pip":
                st["phase"] = "installo dipendenze"
                py = os.path.join(PORTABLE_ROOT, "python_embeded", "python.exe")
                subprocess.run([py, "-m", "pip", "install", "--no-warn-script-location"] + step[1],
                               capture_output=True, timeout=900,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            elif op == "pip_req":
                # installa le dipendenze di un custom node dal suo requirements.txt (se presente)
                st["phase"] = "installo dipendenze dei nodi"
                py = os.path.join(PORTABLE_ROOT, "python_embeded", "python.exe")
                req = os.path.join(PORTABLE_ROOT, step[1].replace("/", os.sep))
                if os.path.exists(req):
                    subprocess.run([py, "-m", "pip", "install", "--no-warn-script-location",
                                    "-r", req], capture_output=True, timeout=1800,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
            elif op == "del":
                try:
                    os.remove(os.path.join(PORTABLE_ROOT, step[1].replace("/", os.sep)))
                except OSError:
                    pass
            elif op == "strip_templates":
                st["phase"] = "rimuovo i template demo"
                import glob as _glob
                sp = os.path.join(PORTABLE_ROOT, "python_embeded", "Lib", "site-packages")
                for d in _glob.glob(os.path.join(sp, "comfyui_workflow_templates_media_*", "templates")):
                    shutil.rmtree(d, ignore_errors=True)
            elif op == "start_comfy":
                st["phase"] = "avvio ComfyUI"
                try:
                    http_json(COMFY_URL + "/system_stats", timeout=3)
                except Exception:
                    py = os.path.join(PORTABLE_ROOT, "python_embeded", "python.exe")
                    logf = open(os.path.join(PORTABLE_ROOT, "comfyui.log"), "w")
                    subprocess.Popen([py, "-s", os.path.join("ComfyUI", "main.py"),
                                      "--windows-standalone-build"],
                                     cwd=PORTABLE_ROOT, stdout=logf, stderr=logf,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
        st["status"] = "done"
        st["phase"] = "completato"
    except Exception as e:
        st["status"] = "cancelled" if st.get("cancel") else "error"
        st["error"] = str(e)[:300]


PRESETS_FILE = os.path.join(APP_DIR, "presets.json")
presets_lock = threading.Lock()


def load_presets():
    try:
        with open(PRESETS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_presets(presets):
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, ensure_ascii=False)


def make_thumb(path):
    """Miniatura JPEG base64 (max 256px) da incorporare nel preset."""
    from PIL import Image
    img = Image.open(path)
    img.thumbnail((256, 256))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# Chip "stile fotografico": vincoli forzati nei prompt, organizzati in gruppi.
# prompt = frammento inglese iniettato; check = parola per non duplicare se l'LLM l'ha gia' scritto;
# neg = termini aggiunti al negative prompt (solo modelli con negative: Z-Image Base, Chroma, Pony)
STYLE_OPTIONS = {
    # --- Realismo ---
    "hyper":     {"group": "Realismo", "label": "Iperrealismo", "tip": "Foto iperrealistica: RAW, grana sottile, dettagli reali",
                  "prompt": "hyperrealistic RAW photograph, photorealistic", "check": "realistic",
                  "neg": "cartoon, anime, illustration, painting, 3d render, cgi"},
    "skin":      {"group": "Realismo", "label": "Pelle vera", "tip": "Contro l'effetto plastica/lisciato: pori, texture irregolare, nei, pelle non ritoccata (forzatura alta, messa a inizio prompt)",
                  "prompt": "extremely detailed real human skin with clearly visible pores and natural uneven texture, "
                            "slight skin tone variations and redness, a few small moles and faint freckles, fine expression "
                            "lines, natural skin oils and subtle imperfections, completely unretouched documentary-style skin",
                  "check": "pores", "prepend": True,
                  "neg": "smooth skin, plastic skin, airbrushed skin, waxy skin, porcelain skin, doll-like skin, "
                         "beauty filter, retouched skin, flawless skin, blurred skin, cgi skin, perfect skin, "
                         "heavy makeup, skin smoothing"},
    "nocartoon": {"group": "Realismo", "label": "No cartoon", "tip": "Impedisce stile cartone/anime/illustrazione: foto vera",
                  "prompt": "a real photograph taken with a professional camera, absolutely not a cartoon, not anime, not an illustration, not a 3D render", "check": "not a cartoon",
                  "neg": "cartoon, anime, illustration, drawing, comic, 3d render"},
    "candid":    {"group": "Realismo", "label": "Foto amatoriale", "tip": "Scatto rubato e non in posa: il trucco piu' efficace per il realismo",
                  "prompt": "candid amateur photography, unedited, natural unposed moment, slightly imperfect framing", "check": "amateur",
                  "neg": "professional retouching, perfect studio photo"},
    "grain":     {"group": "Realismo", "label": "Grana ISO alta", "tip": "Grana visibile da ISO alto: rompe la perfezione digitale",
                  "prompt": "visible film grain, high ISO photography, natural sensor noise", "check": "grain", "neg": ""},
    # --- Focale ---
    "f24":       {"group": "Focale", "label": "24mm", "tip": "Grandangolo: ambiente ampio, prospettiva dinamica",
                  "prompt": "shot on a 24mm wide-angle lens", "check": "24mm", "neg": ""},
    "f35":       {"group": "Focale", "label": "35mm", "tip": "Reportage: contesto e soggetto in equilibrio",
                  "prompt": "shot on a 35mm reportage lens", "check": "35mm", "neg": ""},
    "f50":       {"group": "Focale", "label": "50mm", "tip": "Standard: prospettiva naturale come l'occhio umano",
                  "prompt": "shot on a 50mm lens, natural perspective", "check": "50mm", "neg": ""},
    "f85":       {"group": "Focale", "label": "85mm", "tip": "Ritratto: soggetto valorizzato, sfondo compresso",
                  "prompt": "shot on an 85mm portrait lens", "check": "85mm", "neg": ""},
    "f135":      {"group": "Focale", "label": "135mm tele", "tip": "Teleobiettivo: forte compressione, soggetto isolato",
                  "prompt": "shot on a 135mm telephoto lens, compressed perspective", "check": "135mm", "neg": ""},
    "macro":     {"group": "Focale", "label": "Macro", "tip": "Primissimi piani con texture finissime",
                  "prompt": "macro lens extreme close-up, ultra fine texture detail", "check": "macro", "neg": ""},
    "pano":      {"group": "Focale", "label": "Panoramica", "tip": "Inquadratura larghissima e scenografica",
                  "prompt": "ultra wide panoramic composition, expansive scene", "check": "panoram", "neg": ""},
    # --- Diaframma ---
    "f12":       {"group": "Diaframma", "label": "f/1.2", "tip": "Apertura estrema: piano di fuoco sottilissimo, sfondo dissolto",
                  "prompt": "aperture f/1.2, razor-thin depth of field, dreamy background dissolution", "check": "f/1.2", "neg": ""},
    "f14":       {"group": "Diaframma", "label": "f/1.4", "tip": "Molto aperto: soggetto isolato, bokeh morbidissimo",
                  "prompt": "aperture f/1.4, very shallow depth of field, soft bokeh", "check": "f/1.4", "neg": ""},
    "bokeh":     {"group": "Diaframma", "label": "f/1.8", "tip": "Aperto: soggetto a fuoco, sfondo sfocato cremoso",
                  "prompt": "aperture f/1.8, creamy bokeh, shallow depth of field", "check": "f/1.8", "neg": ""},
    "f8":        {"group": "Diaframma", "label": "f/8 tutto a fuoco", "tip": "Chiuso: massima profondita' di campo, tutto nitido",
                  "prompt": "aperture f/8, deep depth of field, everything in sharp focus", "check": "f/8", "neg": ""},
    # --- Luce ---
    "golden":    {"group": "Luce", "label": "Ora dorata", "tip": "Luce calda radente da tramonto",
                  "prompt": "golden hour warm sunlight", "check": "golden hour", "neg": ""},
    "window":    {"group": "Luce", "label": "Finestra", "tip": "Luce morbida naturale da finestra, interni",
                  "prompt": "soft natural window light, indoor ambience", "check": "window light", "neg": ""},
    "studio":    {"group": "Luce", "label": "Studio softbox", "tip": "Luce professionale da studio con softbox",
                  "prompt": "professional studio lighting with large softbox", "check": "softbox", "neg": ""},
    "flash":     {"group": "Luce", "label": "Flash diretto", "tip": "Flash frontale duro, stile paparazzi/anni 2000 — molto fotografico",
                  "prompt": "direct on-camera flash photography, hard shadows, slightly overexposed highlights", "check": "flash", "neg": ""},
    "neon":      {"group": "Luce", "label": "Neon notturno", "tip": "Notte urbana illuminata da insegne al neon",
                  "prompt": "night scene lit by colorful neon signs, glowing reflections", "check": "neon", "neg": ""},
    # --- Stile ---
    "sharp":     {"group": "Stile", "label": "Dettagli nitidi", "tip": "Massima nitidezza e micro-dettaglio sul soggetto",
                  "prompt": "tack sharp focus, extremely detailed", "check": "sharp focus", "neg": "blurry, soft focus"},
    "film":      {"group": "Pellicola", "label": "Kodak Portra", "tip": "Estetica analogica Kodak Portra 400: colori morbidi",
                  "prompt": "analog film photography look, Kodak Portra 400", "check": "portra", "neg": ""},
    "vintage":   {"group": "Epoca", "label": "Anni '70", "tip": "Foto anni '70: colori sbiaditi, atmosfera retro",
                  "prompt": "1970s vintage photograph, faded warm colors, retro atmosphere", "check": "1970s", "neg": ""},
    "bw":        {"group": "Stile", "label": "Bianco e nero", "tip": "Fotografia in bianco e nero ad alto contrasto",
                  "prompt": "black and white photography, rich tonal contrast", "check": "black and white", "neg": ""},
    "editorial": {"group": "Stile", "label": "Editoriale moda", "tip": "Stile rivista di moda: composizione curata, look da copertina",
                  "prompt": "high fashion editorial photography, magazine cover quality", "check": "editorial", "neg": ""},
    "cinema":    {"group": "Stile", "label": "Cinematografico", "tip": "Fotogramma di film: colori teal&orange, formato anamorfico",
                  "prompt": "cinematic film still, anamorphic look, teal and orange color grading", "check": "cinematic", "neg": ""},
    # --- Realismo (extra) ---
    "phone":     {"group": "Realismo", "label": "Foto smartphone", "tip": "Scatto casuale da telefono: look quotidiano e credibile",
                  "prompt": "casual smartphone snapshot, shot on a phone camera, everyday realistic look", "check": "smartphone", "neg": ""},
    # --- Inquadratura ---
    "closeup":   {"group": "Inquadratura", "label": "Primo piano", "tip": "Viso e spalle riempiono l'inquadratura",
                  "prompt": "close-up portrait framing the face and shoulders", "check": "close-up", "neg": ""},
    "extremecu": {"group": "Inquadratura", "label": "Primissimo piano", "tip": "Solo il viso, occhi perfettamente a fuoco",
                  "prompt": "extreme close-up on the face, eyes in sharp focus", "check": "extreme close-up", "neg": ""},
    "halfbody":  {"group": "Inquadratura", "label": "Mezzo busto", "tip": "Inquadratura dalla vita in su",
                  "prompt": "medium shot framed from the waist up", "check": "waist up", "neg": ""},
    "fullbody":  {"group": "Inquadratura", "label": "Figura intera", "tip": "Tutta la figura visibile, dalla testa ai piedi",
                  "prompt": "full body shot, the entire figure visible from head to toe", "check": "full body",
                  "neg": "close-up, cropped body, cut off legs"},
    "profile":   {"group": "Inquadratura", "label": "Di profilo", "tip": "Soggetto visto di lato",
                  "prompt": "side profile view of the subject", "check": "profile", "neg": ""},
    "frombehind":{"group": "Inquadratura", "label": "Di spalle", "tip": "Soggetto ripreso da dietro",
                  "prompt": "view from behind, subject seen from the back", "check": "from behind", "neg": ""},
    "threeq":    {"group": "Inquadratura", "label": "Tre quarti", "tip": "Angolo a tre quarti, il piu' fotogenico",
                  "prompt": "three-quarter angle view of the subject", "check": "three-quarter", "neg": ""},
    "highangle": {"group": "Inquadratura", "label": "Dall'alto", "tip": "Camera alta che guarda in giu'",
                  "prompt": "high angle shot looking down at the subject", "check": "high angle", "neg": ""},
    "lowangle":  {"group": "Inquadratura", "label": "Dal basso", "tip": "Camera bassa che guarda in su: prospettiva imponente",
                  "prompt": "low angle shot looking up at the subject, imposing perspective", "check": "low angle", "neg": ""},
    "ground":    {"group": "Inquadratura", "label": "Rasoterra", "tip": "Obiettivo quasi a terra",
                  "prompt": "ground level shot, camera almost touching the ground", "check": "ground level", "neg": ""},
    "drone":     {"group": "Inquadratura", "label": "Vista drone", "tip": "Ripresa aerea dall'alto",
                  "prompt": "aerial drone shot from high above", "check": "drone", "neg": ""},
    "ots":       {"group": "Inquadratura", "label": "Sopra la spalla", "tip": "Ripresa da sopra la spalla di qualcuno",
                  "prompt": "over-the-shoulder shot", "check": "over-the-shoulder", "neg": ""},
    "mirror":    {"group": "Inquadratura", "label": "Specchio", "tip": "Soggetto fotografato riflesso in uno specchio",
                  "prompt": "subject photographed through a mirror reflection", "check": "mirror", "neg": ""},
    "selfie":    {"group": "Inquadratura", "label": "Selfie", "tip": "Autoscatto a braccio teso, leggera distorsione da smartphone",
                  "prompt": "selfie taken at arm's length, slight wide phone lens distortion", "check": "selfie", "neg": ""},
    # --- Posa ---
    "standing":  {"group": "Posa", "label": "In piedi", "tip": "Postura eretta e naturale",
                  "prompt": "standing upright, natural confident posture", "check": "standing", "neg": ""},
    "sitting":   {"group": "Posa", "label": "Seduta", "tip": "Posa seduta e rilassata",
                  "prompt": "sitting down, relaxed seated pose", "check": "sitting", "neg": ""},
    "lying":     {"group": "Posa", "label": "Sdraiata", "tip": "Posa distesa/reclinata",
                  "prompt": "lying down, reclining relaxed pose", "check": "lying", "neg": ""},
    "walking":   {"group": "Posa", "label": "Camminando", "tip": "Camminata naturale colta a meta' passo",
                  "prompt": "walking naturally, captured mid-stride", "check": "walking", "neg": ""},
    "running":   {"group": "Posa", "label": "Di corsa", "tip": "Corsa: movimento dinamico, capelli in movimento",
                  "prompt": "running, dynamic movement, hair in motion", "check": "running", "neg": ""},
    "jumping":   {"group": "Posa", "label": "Salto", "tip": "Salto a mezz'aria congelato dallo scatto",
                  "prompt": "jumping in mid-air, frozen action moment", "check": "jumping", "neg": ""},
    "leaning":   {"group": "Posa", "label": "Appoggiata", "tip": "Appoggiata con nonchalance a un muro o una superficie",
                  "prompt": "leaning casually against a wall", "check": "leaning", "neg": ""},
    "armscross": {"group": "Posa", "label": "Braccia incrociate", "tip": "Braccia incrociate sul petto",
                  "prompt": "arms crossed over the chest", "check": "arms crossed", "neg": ""},
    "pockets":   {"group": "Posa", "label": "Mani in tasca", "tip": "Mani in tasca, atteggiamento rilassato",
                  "prompt": "hands in pockets, relaxed stance", "check": "pockets", "neg": ""},
    "hairhand":  {"group": "Posa", "label": "Mano nei capelli", "tip": "Una mano che passa tra i capelli",
                  "prompt": "one hand running through the hair", "check": "running through", "neg": ""},
    "dancing":   {"group": "Posa", "label": "Ballando", "tip": "Colta a meta' di un movimento di danza",
                  "prompt": "dancing, caught mid-movement", "check": "dancing", "neg": ""},
    "lookback":  {"group": "Posa", "label": "Sguardo indietro", "tip": "Si allontana e si volta a guardare oltre la spalla",
                  "prompt": "walking away and glancing back over the shoulder", "check": "glancing back", "neg": ""},
    "crouch":    {"group": "Posa", "label": "Accovacciata", "tip": "Posa bassa e raccolta",
                  "prompt": "crouching low, compact pose", "check": "crouching", "neg": ""},
    # --- Espressione ---
    "smilesoft": {"group": "Espressione", "label": "Sorriso lieve", "tip": "Sorriso appena accennato, espressione rilassata",
                  "prompt": "gentle soft smile, relaxed natural expression", "check": "soft smile", "neg": ""},
    "smilebig":  {"group": "Espressione", "label": "Sorriso aperto", "tip": "Sorriso ampio e sincero che mostra i denti",
                  "prompt": "wide genuine smile showing teeth", "check": "genuine smile", "neg": ""},
    "laugh":     {"group": "Espressione", "label": "Risata", "tip": "Risata spontanea a bocca aperta",
                  "prompt": "laughing out loud, spontaneous joyful laughter", "check": "laugh", "neg": ""},
    "tongue":    {"group": "Espressione", "label": "Linguaccia", "tip": "Lingua fuori in modo giocoso verso la camera",
                  "prompt": "playfully sticking the tongue out at the camera", "check": "tongue", "neg": ""},
    "goofy":     {"group": "Espressione", "label": "Boccaccia", "tip": "Smorfia buffa: guance gonfie, occhi storti",
                  "prompt": "making a silly goofy face at the camera, puffed cheeks, crossed eyes", "check": "goofy", "neg": ""},
    "serious":   {"group": "Espressione", "label": "Seria", "tip": "Espressione seria e composta, viso neutro",
                  "prompt": "serious composed expression, neutral face", "check": "serious", "neg": ""},
    "sad":       {"group": "Espressione", "label": "Malinconica", "tip": "Espressione malinconica, sguardo perso",
                  "prompt": "melancholic wistful expression, distant gaze", "check": "melanchol", "neg": ""},
    "angry":     {"group": "Espressione", "label": "Arrabbiata", "tip": "Espressione intensa e arrabbiata, sopracciglia corrugate",
                  "prompt": "angry intense expression, furrowed brow", "check": "angry", "neg": ""},
    "surprised": {"group": "Espressione", "label": "Sorpresa", "tip": "Occhi spalancati e bocca aperta per la sorpresa",
                  "prompt": "surprised expression, wide eyes, open mouth", "check": "surprised", "neg": ""},
    "scared":    {"group": "Espressione", "label": "Spaventata", "tip": "Paura: occhi sgranati, tensione sul viso",
                  "prompt": "frightened expression, tense wide-eyed fear", "check": "frighten", "neg": ""},
    "crying":    {"group": "Espressione", "label": "Pianto", "tip": "Lacrime visibili sulle guance",
                  "prompt": "crying, visible tears on the cheeks", "check": "tears", "neg": ""},
    "eyesclosed":{"group": "Espressione", "label": "Occhi chiusi", "tip": "Occhi chiusi, espressione serena",
                  "prompt": "eyes closed, serene peaceful expression", "check": "eyes closed", "neg": ""},
    "lookcam":   {"group": "Espressione", "label": "Sguardo in camera", "tip": "Guarda dritto nell'obiettivo, contatto visivo intenso",
                  "prompt": "looking directly into the camera, intense eye contact", "check": "eye contact", "neg": ""},
    "lookaway":  {"group": "Espressione", "label": "Sguardo altrove", "tip": "Guarda lontano dall'obiettivo, assorta",
                  "prompt": "looking away from the camera, lost in thought", "check": "looking away", "neg": ""},
    "wink":      {"group": "Espressione", "label": "Occhiolino", "tip": "Strizza l'occhio in modo giocoso",
                  "prompt": "playful wink at the camera", "check": "wink", "neg": ""},
    "pensive":   {"group": "Espressione", "label": "Pensierosa", "tip": "Espressione riflessiva, mano vicino al mento",
                  "prompt": "pensive thoughtful expression, hand near the chin", "check": "pensive", "neg": ""},
    "shy":       {"group": "Espressione", "label": "Timida", "tip": "Espressione timida, leggero rossore, sguardo basso",
                  "prompt": "shy bashful expression, slight blush, looking down", "check": "bashful", "neg": ""},
    "flirty":    {"group": "Espressione", "label": "Ammiccante", "tip": "Espressione maliziosa e provocante",
                  "prompt": "flirtatious teasing expression, playful gaze", "check": "flirt", "neg": ""},
    "scream":    {"group": "Espressione", "label": "Urlo", "tip": "Urlo a bocca spalancata, intensita' drammatica",
                  "prompt": "screaming with mouth wide open, dramatic intensity", "check": "scream", "neg": ""},
    "bored":     {"group": "Espressione", "label": "Annoiata", "tip": "Espressione annoiata, occhi a mezz'asta",
                  "prompt": "bored unimpressed expression, half-lidded eyes", "check": "bored", "neg": ""},
    # --- Luce (extra) ---
    "lowkey":    {"group": "Luce", "label": "Buia low-key", "tip": "Scena scura: ombre nere profonde, una sola luce fioca",
                  "prompt": "low-key photography, deep black shadows, single dim light source, dark moody scene",
                  "check": "low-key", "neg": "flat bright lighting, overexposed"},
    "shade":     {"group": "Luce", "label": "Ombrosa", "tip": "Soggetto in ombra aperta: luce tenue e ombre morbide",
                  "prompt": "subject in open shade, soft diffused shadows, muted light", "check": "open shade", "neg": ""},
    "dim":       {"group": "Luce", "label": "Penombra", "tip": "Scena appena illuminata, bagliore ambientale debole",
                  "prompt": "dim barely lit scene, faint ambient glow, heavy shadow", "check": "dim", "neg": ""},
    "backlit":   {"group": "Luce", "label": "Controluce", "tip": "Luce forte alle spalle: bordo luminoso e flare",
                  "prompt": "backlit subject, strong light from behind, glowing rim light, subtle lens flare",
                  "check": "backlit", "neg": ""},
    "noon":      {"group": "Luce", "label": "Sole a picco", "tip": "Mezzogiorno pieno: luce dura e ombre nette",
                  "prompt": "harsh direct midday sunlight, strong hard shadows", "check": "midday", "neg": ""},
    "sunrise":   {"group": "Luce", "label": "Alba", "tip": "Prima luce rosa del mattino, ombre lunghe e delicate",
                  "prompt": "soft pink sunrise light, early morning atmosphere, long gentle shadows", "check": "sunrise", "neg": ""},
    "bluehour":  {"group": "Luce", "label": "Ora blu", "tip": "Dopo il tramonto: ambiente blu profondo e freddo",
                  "prompt": "blue hour light after sunset, deep cool blue ambience", "check": "blue hour", "neg": ""},
    "moonlight": {"group": "Luce", "label": "Luce di luna", "tip": "Notte illuminata solo dalla luna pallida",
                  "prompt": "night scene lit only by pale moonlight", "check": "moonlight", "neg": ""},
    "candle":    {"group": "Luce", "label": "Candela", "tip": "Bagliore caldo e tremolante di candele, atmosfera intima",
                  "prompt": "warm flickering candlelight, intimate cozy glow", "check": "candlelight", "neg": ""},
    "firelight": {"group": "Luce", "label": "Fuoco/camino", "tip": "Luce arancione danzante di un camino o falo'",
                  "prompt": "warm firelight from a fireplace, dancing orange glow", "check": "firelight", "neg": ""},
    "streetlamp":{"group": "Luce", "label": "Lampione", "tip": "Cono di luce di un lampione nella notte",
                  "prompt": "night street lit by a sodium street lamp, warm pool of light", "check": "street lamp", "neg": ""},
    "tungsten":  {"group": "Luce", "label": "Locale notturno", "tip": "Luci miste da bar: tungsteno caldo + accenti colorati",
                  "prompt": "moody bar interior lighting, warm tungsten mixed with colored accent lights",
                  "check": "tungsten", "neg": ""},
    "fluo":      {"group": "Luce", "label": "Fluorescente", "tip": "Neon da ufficio freddo e verdastro, realismo spietato",
                  "prompt": "cold greenish fluorescent tube lighting, unflattering realistic light", "check": "fluorescent", "neg": ""},
    "screenlit": {"group": "Luce", "label": "Luce di schermo", "tip": "Viso illuminato solo dal bagliore freddo di uno schermo",
                  "prompt": "face lit only by the cold blue glow of a screen in a dark room", "check": "screen", "neg": ""},
    "chiaro":    {"group": "Luce", "label": "Chiaroscuro", "tip": "Luce drammatica alla Rembrandt: una sorgente, sfondo scuro",
                  "prompt": "dramatic chiaroscuro lighting, Rembrandt style, single light source, dark background",
                  "check": "chiaroscuro", "neg": ""},
    "sidelight": {"group": "Luce", "label": "Luce laterale", "tip": "Luce forte di lato che scolpisce viso e figura",
                  "prompt": "strong side lighting sculpting the face and body", "check": "side lighting", "neg": ""},
    "underlight":{"group": "Luce", "label": "Luce dal basso", "tip": "Illuminazione inquietante dal basso verso l'alto",
                  "prompt": "eerie underlighting from below the face", "check": "underlighting", "neg": ""},
    "silhouette":{"group": "Luce", "label": "Silhouette", "tip": "Figura completamente scura contro uno sfondo luminoso",
                  "prompt": "subject as a dark silhouette against a bright background", "check": "silhouette", "neg": ""},
    "spotlight": {"group": "Luce", "label": "Spotlight", "tip": "Unico fascio di luce teatrale che taglia il buio",
                  "prompt": "single theatrical spotlight beam cutting through darkness", "check": "spotlight", "neg": ""},
    # --- Meteo / Atmosfera ---
    "rain":      {"group": "Meteo", "label": "Pioggia", "tip": "Pioggia battente, superfici bagnate e riflettenti",
                  "prompt": "heavy rain falling, wet reflective surfaces, visible raindrops", "check": "rain", "neg": ""},
    "afterrain": {"group": "Meteo", "label": "Dopo la pioggia", "tip": "Tutto bagnato e lucido, pozzanghere che riflettono",
                  "prompt": "just after the rain, wet glistening ground, puddle reflections", "check": "after the rain", "neg": ""},
    "snow":      {"group": "Meteo", "label": "Neve", "tip": "Neve che cade dolcemente, paesaggio invernale",
                  "prompt": "snow falling softly, white winter landscape", "check": "snow", "neg": ""},
    "fog":       {"group": "Meteo", "label": "Nebbia", "tip": "Nebbia fitta, profondita' che sfuma nel bianco",
                  "prompt": "thick atmospheric fog, soft faded depth", "check": "fog", "neg": ""},
    "storm":     {"group": "Meteo", "label": "Temporale", "tip": "Cielo nero e drammatico, lampi in lontananza",
                  "prompt": "dark stormy sky, dramatic clouds, distant lightning", "check": "storm", "neg": ""},
    "sunny":     {"group": "Meteo", "label": "Sole pieno", "tip": "Giornata limpida e luminosa, cielo blu intenso",
                  "prompt": "bright clear sunny day, vivid blue sky", "check": "sunny", "neg": ""},
    "overcast":  {"group": "Meteo", "label": "Nuvoloso", "tip": "Cielo coperto: luce diffusa, morbida e uniforme",
                  "prompt": "overcast sky, soft even diffused daylight", "check": "overcast", "neg": ""},
    "windy":     {"group": "Meteo", "label": "Vento", "tip": "Vento forte: capelli e vestiti in movimento",
                  "prompt": "strong wind, hair and clothes blowing", "check": "wind", "neg": ""},
    "heathaze":  {"group": "Meteo", "label": "Afa estiva", "tip": "Caldo torrido, aria che trema all'orizzonte",
                  "prompt": "hot summer haze, shimmering heat in the air", "check": "haze", "neg": ""},
    "autumn":    {"group": "Meteo", "label": "Autunno", "tip": "Foglie dorate che cadono, atmosfera autunnale",
                  "prompt": "autumn scene with golden falling leaves", "check": "autumn", "neg": ""},
    "frost":     {"group": "Meteo", "label": "Gelo", "tip": "Aria gelida: fiato visibile, brina sulle superfici",
                  "prompt": "freezing winter air, visible breath, frost on surfaces", "check": "frost", "neg": ""},
    "dust":      {"group": "Meteo", "label": "Polvere/sabbia", "tip": "Vento polveroso, foschia di sabbia",
                  "prompt": "dusty wind, sand haze in the air", "check": "dust", "neg": ""},
    # --- Colore ---
    "warmtones": {"group": "Colore", "label": "Toni caldi", "tip": "Palette calda: ambra e oro",
                  "prompt": "warm color palette, amber and golden tones", "check": "warm color", "neg": ""},
    "coldtones": {"group": "Colore", "label": "Toni freddi", "tip": "Palette fredda: blu acciaio e teal",
                  "prompt": "cool color palette, steel blue and teal tones", "check": "cool color", "neg": ""},
    "pastel":    {"group": "Colore", "label": "Pastello", "tip": "Colori tenui e delicati",
                  "prompt": "soft pastel color palette, delicate hues", "check": "pastel", "neg": ""},
    "vivid":     {"group": "Colore", "label": "Saturi vividi", "tip": "Colori molto saturi e incisivi",
                  "prompt": "highly saturated vivid colors, punchy look", "check": "saturated", "neg": ""},
    "desat":     {"group": "Colore", "label": "Desaturato", "tip": "Colori spenti e smorzati, palette sottile",
                  "prompt": "desaturated muted colors, washed-out subtle palette", "check": "desaturated", "neg": ""},
    "sepia":     {"group": "Colore", "label": "Seppia", "tip": "Monocromo caldo marrone, sapore d'archivio",
                  "prompt": "sepia toned photograph, warm brown monochrome", "check": "sepia", "neg": ""},
    "earthy":    {"group": "Colore", "label": "Toni terra", "tip": "Marroni, ocra e verde oliva",
                  "prompt": "earthy color palette, browns, ochre and olive tones", "check": "earthy", "neg": ""},
    "highcon":   {"group": "Colore", "label": "Alto contrasto", "tip": "Neri profondi e luci brillanti",
                  "prompt": "high contrast look, deep blacks and bright highlights", "check": "high contrast", "neg": ""},
    "lowcon":    {"group": "Colore", "label": "Contrasto morbido", "tip": "Gamma tonale dolce e delicata",
                  "prompt": "low contrast soft look, gentle tonal range", "check": "low contrast", "neg": ""},
    # --- Stile (extra) ---
    "street":    {"group": "Stile", "label": "Street photo", "tip": "Street photography rubata, vita urbana quotidiana",
                  "prompt": "candid street photography, urban everyday life moment", "check": "street photography", "neg": ""},
    "docu":      {"group": "Stile", "label": "Documentario", "tip": "Realta' onesta e non costruita, da reportage",
                  "prompt": "documentary photography style, honest unstaged reality", "check": "documentary", "neg": ""},
    "glamour":   {"group": "Stile", "label": "Glamour", "tip": "Eleganza sensuale e curata da rivista",
                  "prompt": "glamour photography, polished sensual elegance", "check": "glamour", "neg": ""},
    "action":    {"group": "Stile", "label": "Azione sportiva", "tip": "Movimento veloce congelato, energia dinamica",
                  "prompt": "action sports photography, fast motion frozen mid-movement", "check": "action", "neg": ""},
    "minimal":   {"group": "Stile", "label": "Minimalista", "tip": "Composizione pulita, spazio negativo, sfondo semplice",
                  "prompt": "minimalist composition, negative space, clean simple background", "check": "minimalist", "neg": ""},
    "dreamy":    {"group": "Stile", "label": "Sognante", "tip": "Atmosfera eterea: bagliore morbido, romantica foschia",
                  "prompt": "dreamy ethereal atmosphere, soft glow, hazy romantic feel", "check": "dreamy", "neg": ""},
    "noir":      {"group": "Stile", "label": "Noir", "tip": "Film noir: ombre nette, luce da veneziana, mistero",
                  "prompt": "film noir style, dramatic shadows, venetian blind light, mystery mood", "check": "noir", "neg": ""},
    "surreal":   {"group": "Stile", "label": "Surreale", "tip": "Scena onirica con dettagli sottilmente impossibili",
                  "prompt": "surreal dreamlike scene, subtly impossible details", "check": "surreal", "neg": ""},
    # --- Pellicola (extra) ---
    "ekta":      {"group": "Pellicola", "label": "Ektachrome", "tip": "Diapositiva Kodak: toni puliti e freddi",
                  "prompt": "Kodak Ektachrome slide film look, clean cool tones", "check": "ektachrome", "neg": ""},
    "cinestill": {"group": "Pellicola", "label": "CineStill 800T", "tip": "Notturna al tungsteno con alone rosso sulle luci",
                  "prompt": "CineStill 800T look, tungsten night tones, red halation around lights", "check": "cinestill", "neg": ""},
    "velvia":    {"group": "Pellicola", "label": "Fuji Velvia", "tip": "Colori intensissimi da paesaggio su diapositiva",
                  "prompt": "Fuji Velvia film look, intense saturated colors", "check": "velvia", "neg": ""},
    "ilford":    {"group": "Pellicola", "label": "Ilford HP5 B/N", "tip": "Bianco e nero granuloso da reportage classico",
                  "prompt": "Ilford HP5 black and white film, gritty grain, classic reportage", "check": "ilford", "neg": ""},
    "polaroid":  {"group": "Pellicola", "label": "Polaroid", "tip": "Istantanea: fuoco morbido, colori leggermente sbiaditi",
                  "prompt": "instant Polaroid photo look, soft focus, slightly faded colors", "check": "polaroid", "neg": ""},
    "disposable":{"group": "Pellicola", "label": "Usa e getta", "tip": "Macchinetta usa e getta: flash duro, grana da festa anni '90",
                  "prompt": "disposable camera snapshot, harsh flash, grainy 90s party look", "check": "disposable", "neg": ""},
    "lomo":      {"group": "Pellicola", "label": "Lomografia", "tip": "Vignettatura pesante e colori sballati",
                  "prompt": "lomography look, heavy vignetting, shifted quirky colors", "check": "lomography", "neg": ""},
    "expired":   {"group": "Pellicola", "label": "Pellicola scaduta", "tip": "Dominanti imprevedibili e colori slavati",
                  "prompt": "expired film look, color shifts, unpredictable faded cast", "check": "expired", "neg": ""},
    "super8":    {"group": "Pellicola", "label": "Super 8", "tip": "Fotogramma da filmino di famiglia: grana calda",
                  "prompt": "vintage Super 8 film frame, soft grain, warm home-movie feel", "check": "super 8", "neg": ""},
    # --- Epoca (extra) ---
    "y20s":      {"group": "Epoca", "label": "Anni '20", "tip": "Foto d'epoca in monocromo, abiti e ambienti del periodo",
                  "prompt": "1920s photograph style, monochrome, period clothing and setting", "check": "1920s", "neg": ""},
    "y50s":      {"group": "Epoca", "label": "Anni '50", "tip": "Americana classica anni '50",
                  "prompt": "1950s photograph, classic americana period look", "check": "1950s", "neg": ""},
    "y60s":      {"group": "Epoca", "label": "Anni '60", "tip": "Colori era Kodachrome, stile anni '60",
                  "prompt": "1960s photograph, kodachrome era colors, period styling", "check": "1960s", "neg": ""},
    "y80s":      {"group": "Epoca", "label": "Anni '80", "tip": "Colori accesi e moda anni '80",
                  "prompt": "1980s photograph, bold colors, period fashion and vibe", "check": "1980s", "neg": ""},
    "y90s":      {"group": "Epoca", "label": "Anni '90", "tip": "Estetica snapshot analogica anni '90",
                  "prompt": "1990s snapshot aesthetic, casual analog era look", "check": "1990s", "neg": ""},
    "y2k":       {"group": "Epoca", "label": "Y2K 2000s", "tip": "Fotocamera digitale primi 2000: flash e moda y2k",
                  "prompt": "early 2000s digital camera photo, on-camera flash, y2k fashion", "check": "2000s", "neg": ""},
    "future":    {"group": "Epoca", "label": "Futuristico", "tip": "Ambientazione sci-fi avanzata ed elegante",
                  "prompt": "futuristic sci-fi setting, sleek advanced technology ambience", "check": "futuristic", "neg": ""},
    # --- Corpo (grooming/peluria: positivo per tutti, negativo per i modelli che lo usano) ---
    "shaved":    {"group": "Corpo", "label": "Depilata", "tip": "Corpo completamente depilato, pelle liscia e senza peli",
                  "prompt": "completely smooth hairless body, clean-shaven skin, no body hair anywhere",
                  "check": "hairless bod",
                  "neg": "body hair, armpit hair, underarm hair, leg hair, arm hair, pubic hair, stubble, hairy"},
    "armpits":   {"group": "Corpo", "label": "Ascelle depilate", "tip": "Ascelle lisce e depilate",
                  "prompt": "smooth shaved armpits, hairless underarms", "check": "shaved armpits",
                  "neg": "armpit hair, underarm hair, hairy armpits"},
    "legssmooth":{"group": "Corpo", "label": "Gambe depilate", "tip": "Gambe lisce e depilate",
                  "prompt": "smooth shaved legs, hairless legs", "check": "shaved legs",
                  "neg": "leg hair, hairy legs"},
    "bodyhair":  {"group": "Corpo", "label": "Peluria naturale", "tip": "Peluria del corpo naturale, non depilata",
                  "prompt": "natural untrimmed body hair", "check": "body hair", "neg": ""},
    "cleanface": {"group": "Corpo", "label": "Viso glabro", "tip": "Viso liscio e senza peli",
                  "prompt": "clean smooth face, no facial hair", "check": "no facial hair",
                  "neg": "facial hair, beard, mustache, stubble"},
    "tan":       {"group": "Corpo", "label": "Abbronzata", "tip": "Pelle abbronzata dorata",
                  "prompt": "sun-kissed golden tanned skin", "check": "tanned", "neg": ""},
    "fair":      {"group": "Corpo", "label": "Pelle chiara", "tip": "Carnagione chiara",
                  "prompt": "fair pale skin tone", "check": "fair skin", "neg": ""},
    "athletic":  {"group": "Corpo", "label": "Fisico atletico", "tip": "Corpo tonico e atletico",
                  "prompt": "toned athletic fit body", "check": "athletic", "neg": ""},
    "curvy":     {"group": "Corpo", "label": "Formosa", "tip": "Fisico morbido e formoso",
                  "prompt": "curvy soft feminine figure", "check": "curvy", "neg": ""},
    # --- Nudità (studio adulto privato) ---
    "nudefull":  {"group": "Nudità", "label": "Nuda", "tip": "Completamente nuda, nessun indumento",
                  "prompt": "completely nude, fully naked, no clothing at all",
                  "check": "completely nude",
                  "neg": "clothed, dressed, wearing clothes, lingerie, swimsuit, bikini, underwear"},
    "topless":   {"group": "Nudità", "label": "Topless", "tip": "Seno scoperto, solo la parte bassa",
                  "prompt": "topless, bare exposed breasts, wearing only bottoms",
                  "check": "topless", "neg": "bra, bikini top, covered breasts"},
    "bottomless":{"group": "Nudità", "label": "Senza slip", "tip": "Nuda dalla vita in giù",
                  "prompt": "bottomless, no panties, bare exposed hips and crotch",
                  "check": "bottomless", "neg": "panties, underwear, bikini bottom, covered"},
    "frontal":   {"group": "Nudità", "label": "Nudo integrale", "tip": "Nudo frontale completo, genitali in vista",
                  "prompt": "full frontal nudity, completely naked, genitals visible",
                  "check": "full frontal", "neg": "covered, censored, clothed, blurred"},
    "nipples":   {"group": "Nudità", "label": "Capezzoli in vista", "tip": "Seni nudi con capezzoli visibili",
                  "prompt": "bare breasts with visible erect nipples",
                  "check": "erect nipples", "neg": "covered nipples, pasties, bra"},
}

PONY_PREFIX = "score_9, score_8_up, score_7_up, photorealistic, "
PONY_NEGATIVE = ("score_6, score_5, score_4, worst quality, low quality, blurry, deformed, "
                 "mutated hands, extra limbs, missing fingers, bad anatomy, cartoon, "
                 "3d render, overexposed, artifacts, watermark")

# ------------------------------------------------------------------ configuratore PERSONA
# Sezioni ordinate; ogni opzione: label italiana, frammento inglese per il prompt, "kw" =
# parola-chiave usata dal repair per verificare che il dettaglio sia davvero nel prompt.
# Chiave opzione "" = non specificato (la sezione non entra nel prompt).
PERSONA_OPTIONS = [
    {"key": "bellezza", "label": "Bellezza", "options": [
        ("mozzafiato", "Mozzafiato", "breathtakingly beautiful", "breathtakingly"),
        ("moltobella", "Molto bella", "very beautiful", "beautiful"),
        ("carina", "Carina", "pretty, cute", "pretty"),
        ("normale", "Normale", "ordinary average-looking", "ordinary"),
        ("acquaesapone", "Acqua e sapone", "fresh-faced natural girl-next-door beauty, no makeup", "girl-next-door")]},
    {"key": "eta", "label": "Età", "options": [
        ("18", "18 anni", "18 year old", "18"), ("20", "20 anni", "20 year old", "20"),
        ("25", "25 anni", "25 year old", "25"), ("30", "30 anni", "30 year old", "30"),
        ("35", "35 anni", "35 year old", "35"), ("40", "40 anni", "40 year old", "40"),
        ("50", "50 anni", "50 year old", "50")]},
    {"key": "corporatura", "label": "Corporatura", "options": [
        ("magrissima", "Magrissima", "very slim skinny body", "skinny"),
        ("magra", "Magra", "slim slender body", "slim"),
        ("normale", "Normale", "average natural body", "average"),
        ("tonica", "Tonica/atletica", "toned athletic body", "toned"),
        ("morbida", "Un po' morbida", "soft slightly curvy body", "soft"),
        ("formosa", "Formosa", "curvy voluptuous figure", "curvy"),
        ("unpograssa", "Un po' grassa", "chubby plump body", "chubby"),
        ("robusta", "Robusta", "plus-size full-figured body", "plus-size")]},
    {"key": "altezza", "label": "Altezza", "options": [
        ("bassa", "Bassa/minuta", "short petite stature", "petite"),
        ("media", "Media", "average height", "height"),
        ("alta", "Alta", "tall stature", "tall"),
        ("statuaria", "Molto alta", "very tall statuesque figure", "statuesque")]},
    {"key": "pelle", "label": "Pelle", "options": [
        ("liscia", "Liscia e perfetta", "smooth flawless skin", "flawless"),
        ("giovane", "Giovane e fresca", "youthful fresh glowing skin", "youthful"),
        ("naturale", "Naturale (texture vera)", "natural realistic skin texture with visible pores", "pores"),
        ("abbronzata", "Abbronzata", "sun-kissed golden tanned skin", "tanned"),
        ("chiara", "Chiara", "fair pale skin", "fair"),
        ("olivastra", "Olivastra mediterranea", "olive mediterranean skin tone", "olive"),
        ("scura", "Scura", "dark brown skin", "dark")]},
    {"key": "lentiggini", "label": "Lentiggini", "options": [
        ("leggere", "Leggere", "light freckles across nose and cheeks", "freckles"),
        ("marcate", "Marcate", "prominent freckles all over face and shoulders", "freckles"),
        ("nessuna", "Nessuna", "clear even skin without freckles", "without")]},
    {"key": "imperfezioni", "label": "Imperfezioni pelle", "options": [
        ("nessuna", "Nessuna (pelle pulita)", "flawless clear skin, no blemishes, no moles", "blemishes"),
        ("neo", "Qualche neo", "a few small beauty marks and natural moles", "moles"),
        ("qualche", "Qualche macchia", "a few subtle skin blemishes and small imperfections", "imperfections"),
        ("naturali", "Naturali visibili", "natural visible skin imperfections, slightly uneven skin tone", "uneven")]},
    {"key": "capellicolore", "label": "Capelli — colore", "options": [
        ("neri", "Neri", "black hair", "black"),
        ("castanoscuro", "Castano scuro", "dark brown hair", "brown"),
        ("castani", "Castani", "chestnut brown hair", "chestnut"),
        ("castanochiaro", "Castano chiaro", "light brown hair", "brown"),
        ("biondi", "Biondi", "blonde hair", "blonde"),
        ("platino", "Biondo platino", "platinum blonde hair", "platinum"),
        ("rossi", "Rossi", "red hair", "red"),
        ("grigi", "Grigi/argento", "silver grey hair", "grey")]},
    {"key": "capellitipo", "label": "Capelli — tipo", "options": [
        ("lisci", "Lisci", "straight hair", "straight"),
        ("mossi", "Mossi", "wavy hair", "wavy"),
        ("ricci", "Ricci", "curly hair", "curly"),
        ("afro", "Crespi/afro", "coily afro-textured hair", "afro")]},
    {"key": "capellilunghezza", "label": "Capelli — lunghezza", "options": [
        ("corti", "Corti", "short hair", "short"),
        ("caschetto", "Caschetto", "bob-length hair", "bob-length"),
        ("medi", "Medi (spalle)", "shoulder-length hair", "shoulder-length"),
        ("lunghi", "Lunghi", "long hair", "long"),
        ("moltolunghi", "Molto lunghi", "very long waist-length hair", "waist-length")]},
    {"key": "occhi", "label": "Occhi", "options": [
        ("marroni", "Marroni", "brown eyes", "brown"),
        ("nocciola", "Nocciola", "hazel eyes", "hazel"),
        ("verdi", "Verdi", "green eyes", "green"),
        ("azzurri", "Azzurri", "blue eyes", "blue"),
        ("grigi", "Grigi", "grey eyes", "grey"),
        ("scuri", "Quasi neri", "deep dark eyes", "dark")]},
    {"key": "viso", "label": "Viso", "options": [
        ("ovale", "Ovale", "oval face", "oval"),
        ("rotondo", "Rotondo", "round soft face", "round"),
        ("spigoloso", "Spigoloso", "angular face with defined jawline", "jawline"),
        ("cuore", "A cuore", "heart-shaped face", "heart-shaped"),
        ("zigomi", "Zigomi alti", "high cheekbones", "cheekbones")]},
    {"key": "bocca", "label": "Bocca", "options": [
        ("sottili", "Labbra sottili", "thin lips", "thin"),
        ("normali", "Labbra normali", "natural lips", "lips"),
        ("carnose", "Labbra carnose", "full plump lips", "plump")]},
    {"key": "seno", "label": "Seno", "options": [
        ("piccolo", "Piccolo", "small breasts", "small"),
        ("medio", "Medio", "medium breasts", "medium"),
        ("grande", "Grande", "large breasts", "large"),
        ("moltogrande", "Molto grande", "very large heavy breasts", "heavy")]},
    {"key": "fianchi", "label": "Fianchi", "options": [
        ("stretti", "Stretti", "narrow hips", "narrow"),
        ("normali", "Normali", "natural hips", "hips"),
        ("larghi", "Larghi", "wide hips", "wide")]},
    {"key": "sedere", "label": "Sedere", "options": [
        ("piccolo", "Piccolo", "small butt", "butt"),
        ("sodo", "Sodo e rotondo", "firm round butt", "firm"),
        ("grande", "Grande", "large round butt", "large")]},
    {"key": "gambe", "label": "Gambe", "options": [
        ("snelle", "Snelle", "slender legs", "slender"),
        ("toniche", "Toniche", "toned legs", "toned"),
        ("lunghe", "Lunghe", "long legs", "long"),
        ("morbide", "Morbide", "soft thick thighs", "thighs")]},
]
# Preset bellezza: pacchetti di scelte pronti (l'utente li applica e poi ritocca).
PERSONA_PRESETS = {
    "topmodel": {"label": "Top model", "sel": {
        "bellezza": "mozzafiato", "corporatura": "magra", "altezza": "alta",
        "pelle": "liscia", "gambe": "lunghe", "viso": "zigomi", "bocca": "carnose"}},
    "ragazzanormale": {"label": "Ragazza normale", "sel": {
        "bellezza": "normale", "corporatura": "normale", "altezza": "media",
        "pelle": "naturale", "seno": "medio", "gambe": "snelle"}},
    "acquaesapone": {"label": "Acqua e sapone", "sel": {
        "bellezza": "acquaesapone", "corporatura": "normale", "pelle": "giovane",
        "lentiggini": "leggere", "bocca": "normali"}},
    "maggiorata": {"label": "Maggiorata", "sel": {
        "bellezza": "moltobella", "corporatura": "formosa", "seno": "grande",
        "fianchi": "larghi", "sedere": "grande", "bocca": "carnose"}},
    "fitness": {"label": "Fitness", "sel": {
        "bellezza": "moltobella", "corporatura": "tonica", "pelle": "abbronzata",
        "sedere": "sodo", "gambe": "toniche", "altezza": "media"}},
    "morbida": {"label": "Morbida naturale", "sel": {
        "bellezza": "carina", "corporatura": "morbida", "pelle": "giovane",
        "seno": "grande", "fianchi": "larghi", "lentiggini": "leggere"}},
}
_PERSONA_MAP = {s["key"]: {o[0]: o for o in s["options"]} for s in PERSONA_OPTIONS}


def persona_forced_conflicts(frags, raw_terms=None):
    """Sezioni della persona da NON comporre perche' gia' dettate dai [forzati] dell'utente:
    i [forzati] VINCONO sempre (es. [25 anni] batte l'eta' della persona, [bellissima] la
    bellezza) — altrimenti i due vincoli si contraddicono nel prompt (25 vs 30, visto sul
    campo). Guarda anche i termini ORIGINALI italiani: la traduzione puo' perdere pezzi."""
    txt = " ".join(f.get("text", "").lower() for f in (frags or []) if not f.get("directive"))
    txt += " " + " ".join(str(t).lower() for t in (raw_terms or []))
    skip = set()
    if re.search(r"\b(1[89]|[2-6]\d)\b[ -]*(year|years|anni|yo)", txt):
        skip.add("eta")
    if re.search(r"\b(beautiful|gorgeous|stunning|pretty|cute|bella|bellissima)\b", txt):
        skip.add("bellezza")
    return skip


def compose_persona(sel, skip=()):
    """(descrizione inglese, parole-chiave per il repair) dalle scelte del configuratore.
    Le sezioni non specificate (o in 'skip' perche' dettate dai [forzati]) non entrano nel
    prompt. Le keyword si limitano alle 6 piu' distintive (troppe = repair che litiga)."""
    if not isinstance(sel, dict):
        return "", []
    head, frags, kws = [], [], []
    hair = []
    for sec in PERSONA_OPTIONS:
        k = sec["key"]
        if k in skip:
            continue
        o = _PERSONA_MAP[k].get(str(sel.get(k) or ""))
        if not o:
            continue
        if k in ("bellezza", "eta"):       # aprono la frase: "a very beautiful 25 year old woman"
            head.append(o[2])
            kws.append(o[3])
            continue
        if k in ("capellilunghezza", "capellitipo", "capellicolore"):
            hair.append((k, o))
            continue
        frags.append(o[2])
        kws.append(o[3])
    if hair:      # capelli: lunghezza+tipo+colore fusi in un frammento solo ("long wavy red hair")
        order = {"capellilunghezza": 0, "capellitipo": 1, "capellicolore": 2}
        hair.sort(key=lambda x: order[x[0]])
        words = [h[1][2].replace(" hair", "") for h in hair]
        frags.insert(0, " ".join(words) + " hair")
        kws.extend(h[1][3] for h in hair)
    if not (head or frags):
        return "", []
    desc = "a " + (" ".join(head) + " " if head else "") + "woman"
    if frags:
        desc += ", " + ", ".join(frags)
    return desc, kws[:6]


CHARACTERS_FILE = os.path.join(APP_DIR, "characters.json")


def load_characters():
    try:
        with open(CHARACTERS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_characters(data):
    with open(CHARACTERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

# ------------------------------------------------------------------ stato
jobs = {}            # id -> dict
jobs_order = []
work_items = []      # (job_id, stage)  stage: "prompts" | "images"
queue_lock = threading.Lock()
queue_event = threading.Event()
# Due slot llama.cpp indipendenti: il modello prompt (porta 8600) e l'agente (porta 8601).
# Su macchine con molta RAM possono restare montati entrambi (vedi agent_resident).
SLOT_MAIN = {"port": LLAMA_PORT, "proc": None, "model": None, "label": "Prompt",
             "lock": threading.Lock(), "moe_mode": None}
SLOT_AGENT = {"port": 8601, "proc": None, "model": None, "label": "Agente",
              "lock": threading.Lock(), "moe_mode": None}
# Slot GIUDICE: VLM leggero (Qwen3-VL-8B) dedicato alla valutazione immagini. Separato dall'agente
# cosi' la valutazione NON carica il Qwen3.6 da 22GB: il giudice pesa ~6GB e vede molto meglio.
SLOT_EVAL = {"port": 8602, "proc": None, "model": None, "label": "Giudice",
             "lock": threading.Lock(), "moe_mode": None}
RENDER_BUSY = {"n": 0}   # render ComfyUI in corso: l'agente deve stare in RAM
llm_lock = threading.Lock()      # serializza l'uso dello slot prompt
agents = {}                      # id sessione agente -> dict
agent_warm = {"state": "idle", "model": None, "error": None}
last_stats = {"tps": None}       # ultimi token/s dell'agente (per la barra statistiche)
PROGRESS = {"pid": None, "value": 0, "max": 0, "start": 0.0}   # avanzamento step da ComfyUI (ws)


def http_json(url, payload=None, timeout=30, method=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


# ------------------------------------------------------------------ llama.cpp (slot)
def slot_alive(slot):
    try:
        http_json(f"http://127.0.0.1:{slot['port']}/health", timeout=2)
        return True
    except Exception:
        return False


def is_moe(model):
    return AGENT_MODEL_PATTERN in (model or "").lower()


def find_mmproj(model):
    """Trova il file mmproj della stessa famiglia del modello (Qwen3.6 o Qwen3-VL), se presente."""
    lname = (model or "").lower()
    fam = None
    if EVAL_MODEL_PATTERN in lname:
        fam = EVAL_MODEL_PATTERN
    elif AGENT_MODEL_PATTERN in lname:
        fam = AGENT_MODEL_PATTERN
    if not fam:
        return None
    try:
        for fn in os.listdir(MODELS_DIR):
            f = fn.lower()
            if "mmproj" in f and fam in f:
                return fn
    except OSError:
        pass
    return None


def eval_model():
    """Il modello GIUDICE (Qwen3-VL) presente in models/, escluso l'mmproj. None se non scaricato."""
    try:
        for fn in os.listdir(MODELS_DIR):
            f = fn.lower()
            if f.endswith(".gguf") and EVAL_MODEL_PATTERN in f and "mmproj" not in f:
                return fn
    except OSError:
        pass
    return None


def start_llama_slot(slot, model, ctx, log, cancel_flag=None, moe_mode="safe"):
    with slot["lock"]:
        model = os.path.basename(model or "")
        if not model:
            raise RuntimeError(f"Nessun modello .gguf completo in {MODELS_DIR} (download ancora in corso?)")
        if slot_alive(slot) and slot["model"] is None:
            # server orfano di un precedente avvio dell'app: adottalo se ha il modello giusto
            try:
                props = http_json(f"http://127.0.0.1:{slot['port']}/props", timeout=5)
                mp = props.get("model_path", "")
                slot["model"] = os.path.basename(mp) if mp else None
                slot["moe_mode"] = "safe"   # modalita' ignota: assumi la prudente
            except Exception:
                pass
        if slot_alive(slot) and slot["model"] == model and \
                (not is_moe(model) or slot.get("moe_mode") == moe_mode):
            return
        stop_llama_slot(slot)  # modello diverso: riavvia
        if not os.path.exists(LLAMA_SERVER_EXE):
            raise RuntimeError(f"llama-server.exe non trovato in {LLAMA_SERVER_EXE}")
        model_path = os.path.join(MODELS_DIR, model)
        if not os.path.exists(model_path):
            raise RuntimeError(f"Modello LLM non trovato: {model_path}")
        log.append(f"Modello: {model}")
        lname = model.lower()
        if not is_moe(model):
            # un modello dense occupa tutta la VRAM: libera quella di ComfyUI
            try:
                http_json(COMFY_URL + "/free", {"unload_models": True, "free_memory": True}, timeout=15)
            except Exception:
                pass
        log.append("Avvio LLM (llama.cpp)...")
        log_path = os.path.join(LLM_DIR, f"llama_server_{slot['port']}.log")
        cmd = [LLAMA_SERVER_EXE, "-m", model_path, "-ngl", "999", "-c", str(ctx),
               "--host", "127.0.0.1", "--port", str(slot["port"]), "--no-webui"]
        if is_moe(model):
            # "safe": esperti in RAM (VRAM libera per i render). "fast": una parte degli
            # esperti in VRAM quando non ci sono render (chat piu' veloce).
            if moe_mode == "fast":
                n_cpu_moe = get_config().get("agent_cpu_moe_fast", 30)
            else:
                n_cpu_moe = get_config().get("agent_cpu_moe", 999)
            slot["moe_mode"] = moe_mode
            cmd += ["--n-cpu-moe", str(n_cpu_moe), "--jinja"]
        # mmproj (visione) per i modelli multimodali: Qwen3.6 (agente MoE) e Qwen3-VL (giudice dense).
        mm = find_mmproj(model)
        if mm:
            cmd += ["--mmproj", os.path.join(MODELS_DIR, mm)]
            if not is_moe(model):
                cmd += ["--jinja"]   # il VLM dense (Qwen3-VL) usa il suo chat template integrato
                # Qwen3-VL di default codifica una thumbnail 1024px in ~250 image-token: sotto i
                # 1024 minimi che il modello stesso richiede per il grounding fine (mani fuse, volti
                # sciolti, aderenza al prompt). Senza questo "vede" male e dava 10/10 a tutto.
                cmd += ["--image-min-tokens", "1024"]
        # I GGUF Nemo "abliterated" hanno il chat template rotto nei metadati:
        # forziamo il template Mistral V3-Tekken corretto per tutti i modelli Nemo.
        if "nemo" in lname and os.path.exists(NEMO_TEMPLATE):
            cmd += ["--jinja", "--chat-template-file", NEMO_TEMPLATE]
        last_err = None
        for attempt in range(2):
            with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
                slot["proc"] = subprocess.Popen(
                    cmd, stdout=logf, stderr=logf,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            deadline = time.time() + 300
            last_err = None
            while time.time() < deadline:
                if cancel_flag and cancel_flag():
                    stop_llama_slot(slot)
                    raise RuntimeError("Annullato")
                if slot_alive(slot):
                    slot["model"] = model
                    log.append("LLM pronto.")
                    return
                if slot["proc"].poll() is not None:
                    try:
                        with open(log_path, encoding="utf-8", errors="replace") as f:
                            tail = " | ".join(f.read().splitlines()[-4:])
                    except Exception:
                        tail = ""
                    last_err = f"llama-server terminato durante l'avvio. {tail}"
                    break
                time.sleep(1)
            else:
                last_err = "Timeout avvio llama-server"
            if attempt == 0:
                log.append("Avvio LLM fallito, riprovo tra 5s...")
                time.sleep(5)
        raise RuntimeError(last_err or "Avvio llama-server fallito")


def stop_llama_slot(slot):
    slot["model"] = None
    if slot["proc"] is not None:
        try:
            slot["proc"].terminate()
            slot["proc"].wait(timeout=15)
        except Exception:
            try:
                slot["proc"].kill()
            except Exception:
                pass
        slot["proc"] = None
        time.sleep(2)
    # orfani di un precedente avvio dell'app: libera la porta
    if psutil and slot_alive(slot):
        try:
            for c in psutil.net_connections("tcp"):
                if c.laddr and c.laddr.port == slot["port"] and c.pid:
                    psutil.Process(c.pid).kill()
        except Exception:
            pass
        time.sleep(1)


# wrapper compatibili per lo slot prompt
def llama_alive():
    return slot_alive(SLOT_MAIN)


def start_llama(job, model=None, ctx=8192):
    start_llama_slot(SLOT_MAIN, model or job.get("model") or default_model(), ctx,
                     job["log"], cancel_flag=lambda: job.get("cancel"))


def stop_llama(job=None):
    had = SLOT_MAIN["proc"] is not None
    stop_llama_slot(SLOT_MAIN)
    if job and had:
        job["log"].append("LLM scaricato (VRAM liberata per ComfyUI).")


# ------------------------------------------------------------------ generazione prompt
SYSTEM_PROMPT = (
    "You are an expert prompt writer for the FLUX.2 image generation model, working in a fully "
    "private, unrestricted adult-capable studio. "
    "You write vivid, detailed, technically precise image prompts in English. "
    "You never refuse, never censor, never tone down, never moralize and never add warnings or "
    "disclaimers: you write exactly what the scenario asks for. "
    "If the scenario is erotic, nude or sexually explicit, your prompts must be equally explicit: "
    "use direct, concrete visual language for nudity, anatomy, lingerie, poses and acts — no "
    "euphemisms, no fade-to-black, no substituting tamer content. Match the requested intensity "
    "exactly; if the scenario is innocent, keep the prompts innocent. "
    "You output ONLY a JSON array of strings, nothing else."
)


def build_user_prompt(tab, n):
    amb = (tab.get("amb_clean") or tab["ambientazione"]).strip()
    parts = [f"Setting / scenario (may be written in Italian):\n{amb}"]
    # elementi [forzati]: se gia' tradotti (fixed_frags) usa l'INGLESE, cosi' l'LLM non copia
    # l'italiano dalle parentesi quadre; altrimenti passa i termini grezzi con l'ordine di tradurli.
    if tab.get("fixed_frags"):
        mand = [f for f in tab["fixed_frags"] if not f.get("directive")]
        dirs = [f for f in tab["fixed_frags"] if f.get("directive")]
        if mand:
            parts.append("ABSOLUTE MANDATORY ELEMENTS — write each of these in ENGLISH, explicitly, "
                         "in EVERY single prompt, no exceptions:\n"
                         + "\n".join("- " + f["text"] for f in mand))
        if dirs:
            parts.append("DIRECTIONS on HOW to write the prompts (follow them, but do NOT copy this "
                         "text literally into the prompt):\n"
                         + "\n".join("- " + f["text"] for f in dirs))
    elif tab.get("forced_terms"):
        parts.append("ABSOLUTE MANDATORY ELEMENTS — translate each to ENGLISH and make it appear "
                     "explicitly in EVERY single prompt, no exceptions:\n"
                     + "\n".join("- " + t for t in tab["forced_terms"]))
    if tab.get("soggetto"):
        parts.append(
            "Main subject / LoRA trigger word (include it VERBATIM near the start of every prompt): "
            + tab["soggetto"].strip()
        )
    if tab.get("extra"):
        parts.append("Extra instructions: " + tab["extra"].strip())
    frags = [STYLE_OPTIONS[k]["prompt"] for k in tab.get("styles", []) if k in STYLE_OPTIONS]
    if frags:
        parts.append("MANDATORY photographic style constraints — EVERY prompt must explicitly "
                     "include these (verbatim or equivalent): " + "; ".join(frags))
    style = tab.get("image_model", "klein")
    if style.startswith("wf:"):
        try:
            style = load_custom_workflow(style[3:] + ".json").get("meta", {}).get("target", "klein")
        except Exception:
            style = "klein"
    mw = tab.get("max_words", MAX_PROMPT_WORDS)
    if style == "pony" or tab.get("tagmode"):
        # elenco di tag/parole separate da virgola, non frasi (utile se il modello si perde
        # con le frasi lunghe). Pony lo usa sempre; le altre reti solo se l'utente lo chiede.
        ntags = max(12, min(40, round(mw * 0.6)))
        style_rule = (
            f"- Each prompt is a comma-separated list of about {ntags} short keyword TAGS, NOT full "
            "sentences and NOT prose. Each tag is 1-4 words. Order: subject, then outfit with colors, "
            "then pose/action, then setting/location, then lighting, then camera/quality. "
            "Example format: 'young woman, red floral dress, holding a cocktail, sunny Rome street, "
            "warm side light, 50mm, photorealistic, sharp focus'. "
            "Do NOT write connected sentences: only tags separated by commas.\n"
        )
    else:
        style_rule = (
            f"- Each prompt is CONCISE, about {mw} words (a few more or fewer is fine) of concrete "
            "visual detail: subject, outfit, pose, environment, lighting, camera and lens feel, mood. "
            "No filler, no repetition, no numbering, no titles, no markdown inside the strings.\n"
        )
    literal_rule = ""
    if tab.get("literal"):
        literal_rule = (
            "- LITERAL MODE (mandatory): follow the scenario WORD FOR WORD. Every element, act, "
            "body part, garment or detail the scenario asks for MUST appear explicitly in EVERY "
            "prompt, named with direct, unambiguous, anatomically explicit words — never implied, "
            "never softened, never replaced by mood or atmosphere. State the most important "
            "requested elements at the START of each prompt, then add composition and lighting. "
            "No vague artistic filler: every sentence must describe something concretely visible "
            "in the image.\n"
        )
    parts.append(
        f"Write {n} distinct image prompts in ENGLISH for this scenario.\n"
        "Rules:\n"
        "- All prompts must be coherent with each other: same setting, same subject(s), same overall "
        "photographic style, as if they were shots from a single photo session.\n"
        "- SELF-CONTAINED PROMPTS (critical): the image model reads each prompt in ISOLATION, with "
        "no memory of the scenario or of the other prompts — anything not written in a prompt will "
        "NOT appear in that photo. Therefore EVERY prompt must restate in full: (a) the "
        "location/setting exactly as the scenario describes it (city, place, environment, darkness "
        "or weather), (b) the subject's complete outfit with colors, patterns and fit — NEVER "
        "write just 'her dress' or 'the outfit': always repeat e.g. 'her tight low-cut white floral "
        "dress' —, and (c) every object the subject holds or uses and every ongoing action (a "
        "cocktail glass in hand, a phone, a cigarette, eating...): if the scenario says she is "
        "drinking a cocktail, the cocktail must appear in EVERY prompt. This applies to every "
        "prompt, including close-ups and back views. Before writing, silently list these fixed "
        "elements (location, outfit with colors, held objects/actions), then double-check that "
        "every single prompt contains ALL of them.\n"
        "- CONSTRAINTS FIRST: if the scenario explicitly specifies framing, pose, viewpoint or what "
        "must be visible (e.g. full body, standing, seen from the front, specific body parts visible), "
        "then EVERY prompt MUST respect those requirements exactly — never switch to close-ups, "
        "crops, back views or poses that would violate them.\n"
        "- Vary between prompts ONLY the aspects the scenario does NOT fix: micro-pose, gesture, "
        "facial expression, lighting nuance, background details, moment. If framing is not "
        "specified, you may also vary framing (close-up, medium shot, full body, wide shot).\n"
        + literal_rule + style_rule +
        f"Return ONLY a JSON array of exactly {n} strings."
    )
    return "\n\n".join(parts)


def clean_prompt(s):
    """Toglie le virgolette spurie che il LLM a volte lascia attaccate al prompt."""
    s = str(s).strip()
    while s and s[0] in '"“”«':
        s = s[1:].lstrip()
    while s and s[-1] in '"“”»':
        s = s[:-1].rstrip()
    return s


def parse_prompts(text):
    start = text.find("[")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        arr = json.loads(text[start:i + 1])
                        out = [clean_prompt(x) for x in arr]
                        out = [x for x in out if x]
                        if out:
                            return out
                    except Exception:
                        pass
                    break
        start = text.find("[", start + 1)
    lines = []
    for ln in text.splitlines():
        ln = clean_prompt(re.sub(r'^\s*(?:\d+[\.\)]\s*|[-*]\s*)', "", ln).rstrip(","))
        if len(ln) > 30:
            lines.append(ln)
    return lines


def generate_prompts_for_tab(job, tab):
    n = tab["num_images"]
    prompts = []
    attempts = 0
    while len(prompts) < n and attempts < 4:
        missing = n - len(prompts)
        attempts += 1
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(tab, missing)},
            ],
            "temperature": 0.85,
            "top_p": 0.95,
            "max_tokens": min(7000, 300 + 160 * missing),
        }
        resp = http_json(LLAMA_URL + "/v1/chat/completions", payload, timeout=600)
        text = resp["choices"][0]["message"]["content"]
        got = parse_prompts(text)
        prompts.extend(got[:missing])
        job["log"].append(f"[{tab['title']}] LLM: {len(prompts)}/{n} prompt.")
        if job["cancel"]:
            break
    if not prompts:
        raise RuntimeError(f"L'LLM non ha restituito prompt validi per '{tab['title']}'")
    while len(prompts) < n:  # estrema ratio: ricicla (il seed diverso varia comunque l'immagine)
        prompts.append(prompts[len(prompts) % len(prompts)])
    return prompts[:n]


def _parse_clock(t):
    """'14:30' -> minuti dalla mezzanotte, o None se non e' un orario valido."""
    m = re.fullmatch(r"([01]?\d|2[0-3])[:.h]([0-5]\d)", str(t).strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def generate_story_prompts(job, tab):
    """Modalita' REGISTA FOTO in due passaggi: il batch diventa un RACCONTO cronologico.
    Passo 1 — SCALETTA: l'LLM produce protagonista (descrizione unica per tutta la storia)
    + n capitoli {orario, evento}; gli orari sono VALIDATI IN CODICE (formato HH:MM,
    strettamente crescenti) perche' i modelli piccoli sbagliano la cronologia se lasciati
    liberi. Passo 2 — SCRITTURA: ogni capitolo diventa un prompt foto autosufficiente
    (il modello immagine non ha memoria) che apre con orario + descrizione completa della
    protagonista e rende luce/meteo giusti per QUEL momento. Blocchi da 12 per i token."""
    n = tab["num_images"]
    maxw = tab.get("max_words") or 55
    scene = (tab.get("amb_clean") or tab["ambientazione"]).strip()
    # elementi [forzati] gia' tradotti: chiedo all'LLM di scriverli in ogni capitolo, cosi'
    # nascono naturali invece di venire appesi dal repair a posteriori.
    must = [f["text"] for f in (tab.get("fixed_frags") or []) if not f.get("directive")]

    # --- passo 1: scaletta con orari crescenti, validata in codice ---
    plan, protagonist = None, None
    for attempt in range(4):
        if job["cancel"]:
            raise RuntimeError("Annullato")
        sys_a = (
            f"You are a film director planning a photo story of exactly {n} chapters. "
            "Output STRICT JSON with exactly these keys:\n"
            '{"protagonist": "...", "chapters": [{"time": "HH:MM", "event": "..."}, ...]}\n'
            "Rules:\n"
            f"- EXACTLY {n} chapters. 'time' is a 24-hour clock time (e.g. \"14:30\"); times "
            "must be STRICTLY INCREASING and spread over the time span the brief implies "
            "(if the brief gives hours, use that span; otherwise morning to night).\n"
            "- 'protagonist': one short English description reused for the WHOLE story: "
            "hair, look, outfit with its colors. Use the brief's details when given; if the "
            "brief does NOT describe the character, INVENT a precise look (hair color and "
            "style, build, outfit with exact colors) and keep it for the whole story.\n"
            "- 'event': one concrete, interesting moment — place + action — different in "
            "every chapter, each following plausibly from the previous one (reachable "
            "places, believable day).\n"
            "- Output ONLY the JSON."
        )
        user_a = "Story brief: " + scene
        if tab.get("soggetto"):
            user_a += "\nSubject/trigger (use verbatim as the subject's name): " + tab["soggetto"]
        pdesc_a = (tab.get("_persona_frag") or {}).get("text")
        if pdesc_a:
            user_a += ("\nThe protagonist MUST match EXACTLY this description (use it as-is, "
                       "you may only add what it does not specify): " + pdesc_a)
        if tab.get("extra"):
            user_a += "\nExtra instructions: " + tab["extra"]
        why = "risposta non leggibile"
        try:
            payload = {"messages": [{"role": "system", "content": sys_a},
                                    {"role": "user", "content": user_a}],
                       "temperature": 0.5, "top_p": 0.95,
                       "max_tokens": min(7000, 400 + 80 * n)}
            resp = http_json(LLAMA_URL + "/v1/chat/completions", payload, timeout=600)
            m = re.search(r"\{.*\}", resp["choices"][0]["message"]["content"], re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
            # capitoli: dict {time,event} ma tollero anche stringhe semplici (solo evento)
            chs = []
            for c in (data.get("chapters") or []):
                if isinstance(c, dict) and str(c.get("event") or "").strip():
                    chs.append({"time": _parse_clock(c.get("time")),
                                "event": str(c["event"]).strip()})
                elif isinstance(c, str) and c.strip():
                    chs.append({"time": None, "event": c.strip()})
            prot = str(data.get("protagonist") or "").strip()
            if not prot:
                why = "manca la descrizione della protagonista"
            elif len(chs) < n and not (attempt == 3 and len(chs) >= 2):
                why = f"capitoli insufficienti ({len(chs)}/{n})"
            else:
                if len(chs) < n:
                    # estrema ratio (ultimo tentativo): stiro la scaletta ripetendo i
                    # momenti in ordine — due scatti dello stesso evento, mai disordine.
                    job["log"].append(f"[{tab['title']}] regista foto: solo {len(chs)}/{n} "
                                      "capitoli dopo 4 tentativi — ne ripeto alcuni "
                                      "(scatti doppi dello stesso momento).")
                    chs = [chs[i * len(chs) // n] for i in range(n)]
                chs = chs[:n]
                times = [c["time"] for c in chs]
                # gli EVENTI (in ordine di scrittura) sono la storia; gli ORARI, se rotti,
                # doppi o non crescenti, li riassegno IO distribuendoli sull'arco che
                # l'LLM voleva coprire (min..max dei suoi orari validi, o 09-22).
                if not (all(t is not None for t in times)
                        and all(times[i] < times[i + 1] for i in range(n - 1))):
                    good = [t for t in times if t is not None]
                    t0 = min(good) if good else 9 * 60
                    t1 = max(good) if good else 22 * 60
                    if t1 - t0 < max(60, 20 * (n - 1)):
                        t1 = min(t0 + max(60, 30 * (n - 1)), 23 * 60 + 55)
                    step = (t1 - t0) / max(1, n - 1)
                    times = [int(t0 + step * i) for i in range(n)]
                    job["log"].append(f"[{tab['title']}] regista foto: orari della "
                                      "scaletta incompleti o non crescenti — li ho "
                                      "riassegnati io sull'arco "
                                      f"{t0 // 60:02d}:{t0 % 60:02d}-{t1 // 60:02d}:{t1 % 60:02d}.")
                protagonist = prot
                plan = [{"time": f"{times[i] // 60:02d}:{times[i] % 60:02d}",
                         "event": chs[i]["event"]} for i in range(n)]
                break
        except Exception as e:
            why = f"errore: {str(e)[:80]}"
        job["log"].append(f"[{tab['title']}] regista foto: scaletta non valida — {why} "
                          f"(tentativo {attempt + 1}/4), riprovo...")
    if not plan:
        raise RuntimeError(f"Il regista foto non ha prodotto una scaletta valida per '{tab['title']}'")
    job["log"].append(f"[{tab['title']}] regista foto — protagonista: {protagonist}")
    job["log"].append(f"[{tab['title']}] scaletta: "
                      + " | ".join(f"{c['time']} {c['event']}" for c in plan))
    # la protagonista (descritta dal regista se l'utente non l'ha fatto) diventa un
    # elemento FISSO: il repair la garantira' in ogni capitolo, come i [forzati].
    # Niente aggettivi-vuoto come parole-chiave: contano capelli, colori, outfit.
    _empty = {"beautiful", "gorgeous", "stunning", "pretty", "lovely", "elegant",
              "woman", "girl", "man", "young"}
    tab["_story_prot_frag"] = {"text": protagonist,
                               "keywords": [w for w in frag_words(protagonist)
                                            if w not in _empty][:5]}

    # --- passo 2: scrittura dei capitoli (blocchi da 12) ---
    prompts = []
    attempts = 0
    while len(prompts) < n and attempts < 6 and not job["cancel"]:
        attempts += 1
        chunk = plan[len(prompts):len(prompts) + 12]
        sys_b = (
            "You are writing image prompts for chapters of one photo story. Output a STRICT "
            f"JSON array of exactly {len(chunk)} strings, one English photo prompt per "
            "chapter, in the given order.\n"
            "Rules:\n"
            "- Each prompt is SELF-CONTAINED (the image model has NO memory): it must open "
            "with the chapter's clock time and the protagonist's FULL description exactly as "
            "given, then show the chapter's event in its place, with LIGHT, SKY and WEATHER "
            "correct for that exact time of day. A pronoun alone ('she', 'he') as the "
            "subject is FORBIDDEN.\n"
            "- Never mention other chapters, time ranges or spans: each prompt contains ONLY "
            "its own clock time.\n"
            + (("- Include naturally in EVERY prompt: " + "; ".join(must) + ".\n") if must else "")
            + f"- Each prompt under {maxw} words. Output ONLY the JSON array."
        )
        user_b = ("Protagonist (repeat in full in every prompt): " + protagonist + "\n"
                  + "\n".join(f"Chapter {len(prompts) + k + 1} — {c['time']} — {c['event']}"
                              for k, c in enumerate(chunk)))
        try:
            payload = {"messages": [{"role": "system", "content": sys_b},
                                    {"role": "user", "content": user_b}],
                       "temperature": 0.6, "top_p": 0.95,
                       "max_tokens": min(7000, 300 + 160 * len(chunk))}
            resp = http_json(LLAMA_URL + "/v1/chat/completions", payload, timeout=600)
            got = parse_json_array(resp["choices"][0]["message"]["content"]) or []
            got = [clean_prompt(p) for p in got if isinstance(p, str) and len(str(p).strip()) > 20]
            prompts.extend(got[:len(chunk)])
        except Exception:
            pass
        job["log"].append(f"[{tab['title']}] regista foto: {len(prompts)}/{n} capitoli scritti.")
    if not prompts:
        raise RuntimeError(f"Il regista foto non ha scritto la storia per '{tab['title']}'")
    # estrema ratio: capitoli mancanti = scaletta grezza (orario + evento + protagonista)
    while len(prompts) < n:
        c = plan[len(prompts)]
        prompts.append(f"At {c['time']}, {protagonist}, {c['event']}")
    return prompts[:n]


# parole troppo generiche per decidere se un elemento fisso e' gia' nel prompt
FIXED_FRAG_STOPWORDS = {"the", "and", "for", "her", "his", "its", "are", "was", "has",
                        "with", "from", "into", "onto", "over", "under", "while", "that",
                        "this", "then", "them", "they", "there", "their", "hers", "some",
                        "very", "also", "near", "been", "wearing", "holding", "location",
                        "subject", "scene", "setting", "background"}
MAX_PROMPT_WORDS = 70   # tetto morbido del prompt finale: oltre, viene accorciato
REPAIR_ROUNDS = 4       # quanti ri-controlli/riscritture al massimo per ogni prompt


def frag_words(frag_text):
    """Parole significative (>=3 lettere, non generiche) di un frammento. 3 lettere per
    non perdere i colori: 'red', 'tan'..."""
    return [w for w in re.findall(r"[a-z]{3,}", frag_text.lower())
            if w not in FIXED_FRAG_STOPWORDS]


def word_count(s):
    return len(re.findall(r"\S+", s or ""))


def snap_dim(v):
    """Larghezza/altezza valida: multiplo di 16, entro 256..2048 (limiti sicuri per i modelli)."""
    try:
        n = int(v)
    except (TypeError, ValueError):
        n = 1024
    n = round(n / 16) * 16
    return max(256, min(2048, n))


# conteggi: il numero di soggetti e' critico e l'LLM (soprattutto in tag mode) lo perde spesso.
# Ogni numero ha i suoi sinonimi accettati: "3 ragazze" e' soddisfatto da 3, three o trio.
NUM_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "pair": 2, "couple": 2, "duo": 2, "trio": 3, "quartet": 4}
NUM_SYN = {2: {"2", "two", "pair", "couple", "duo", "both"},
           3: {"3", "three", "trio"},
           4: {"4", "four", "quartet"},
           5: {"5", "five"}, 6: {"6", "six"}, 7: {"7", "seven"}, 8: {"8", "eight"}}


def frag_count(text):
    """Il numero di soggetti richiesto dal frammento (2..9), o None."""
    tl = text.lower()
    m = re.search(r"\b([2-9])\b", tl)
    if m:
        return int(m.group(1))
    for w, n in NUM_WORDS.items():
        if re.search(r"\b" + w + r"\b", tl):
            return n
    return None


def frag_missing(prompt_text, frag):
    """Parole-CHIAVE essenziali del frammento assenti dal prompt (confine di parola).
    Le quadre-ISTRUZIONE (directive=True, es. 'crea prompt da set porno', 'forza nudo completo')
    non sono elementi visibili: non si controllano a parole ne' si appendono al render -> [].
    Usa 'keywords' (allineate al vocabolario dei prompt: es. 'floral','dress','red') invece di
    ogni parola del testo: cosi' 'red floral dress' soddisfa 'vestito rosso a fiori' senza
    pretendere le parole esatte 'flowers'/'neckline'. Ripiega sulle parole del testo se mancano.
    Il CONTEGGIO (es. '3 ragazze') e' sempre obbligatorio, accettando i sinonimi (3/three/trio)."""
    if frag.get("directive"):
        return []
    kws = frag.get("keywords") or frag_words(frag.get("text", ""))
    pl = prompt_text.lower()
    miss = [w for w in kws
            if not re.search(r"\b" + re.escape(str(w).lower()) + r"\b", pl)]
    n = frag_count(frag.get("text", ""))
    if n and n in NUM_SYN and not any(re.search(r"\b" + re.escape(s) + r"\b", pl) for s in NUM_SYN[n]):
        miss.append(str(n))   # il numero manca del tutto (nessun sinonimo presente)
    return miss


def parse_json_array(text):
    """Estrae il primo array JSON di primo livello dal testo (tollerante al rumore intorno)."""
    start = text.find("[")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
        start = text.find("[", start + 1)
    return None


DIRECTIVE_VERBS = ("crea", "creare", "genera", "generare", "scrivi", "scrivere", "fai", "fare",
                   "forza", "forzare", "usa", "usare", "imposta", "impostare", "rendi", "rendere",
                   "aggiungi", "metti", "create", "generate", "write", "make", "force", "ensure",
                   "add", "use", "render")
# NB: "set" tolto dai verbi-istruzione: la traduzione "Set in Vicenza" veniva scambiata
# per una guida e [Vicenza] spariva dai prompt (visto sul campo).


def is_directive_term(text):
    """True se la quadra e' un'ISTRUZIONE al prompt-writer (es. 'crea prompt da set da film porno',
    'forza nudo completo', 'rendi cinematografico') e NON un elemento visibile della scena. Queste
    non vanno controllate a parole ne' appese al render: sono guida, non contenuto."""
    t = (text or "").strip().lower()
    if not t:
        return False
    if "prompt" in t:                       # 'crea prompt ...', 'prompt da set'
        return True
    return t.split()[0] in DIRECTIVE_VERBS   # inizia con un verbo imperativo


def parse_forced_terms(amb):
    """[tra parentesi quadre] = elemento che DEVE apparire in ogni prompt.
    Ritorna (testo senza parentesi, lista dei termini forzati)."""
    terms = [t.strip() for t in re.findall(r"\[([^\[\]]+)\]", amb) if t.strip()]
    clean = re.sub(r"\[([^\[\]]+)\]", r"\1", amb)
    return clean, terms


def _clean_keywords(raw, fallback_text):
    kws = [str(w).strip().lower() for w in (raw or []) if str(w).strip()]
    kws = [w for w in kws if re.fullmatch(r"[a-z0-9][a-z0-9-]*", w) and w not in FIXED_FRAG_STOPWORDS]
    # solo parole che stanno DAVVERO nel frammento tradotto: se l'LLM propone sinonimi
    # ("nude" per "Completely naked") il controllo li pretenderebbe entrambi e appenderebbe
    # il frammento anche quando la parola giusta c'e' gia'.
    tl = str(fallback_text).lower()
    kws = [w for w in kws if re.search(r"\b" + re.escape(w) + r"\b", tl)] or kws[:2]
    if not kws:
        kws = frag_words(fallback_text)[:4]
    kws = kws[:4]
    # i NUMERI del termine (eta', conteggi: "25 anni") sono SEMPRE parole-chiave:
    # senza, "[25 anni]" passava il controllo con soli "young/woman" e il 25 spariva.
    for n in re.findall(r"\b\d{1,3}\b", str(fallback_text)):
        if n not in kws:
            kws.append(n)
    return kws[:6]


def translate_forced_terms(job, tab):
    """Converte i termini forzati (anche in italiano) in frammenti inglesi + le parole-chiave
    essenziali (nel vocabolario che i prompt usano davvero). Se la traduzione fallisce si usa
    il termine cosi' com'e'."""
    terms = tab.get("forced_terms") or []
    if not terms:
        return
    frags = None
    try:
        ask = ("For each of the following photo-scene elements (they may be in Italian), return a "
               "JSON object with two fields:\n"
               '  "text": a short ENGLISH image-prompt fragment (3-10 words) keeping colors, '
               "patterns, objects and details;\n"
               '  "keywords": an array of the 2-4 MOST distinctive lowercase words that MUST appear '
               "in every prompt, written in the exact form an English image prompt normally uses "
               '(for "vestito rosso a fiori scollato" -> ["red","floral","dress"]; for "cocktail" '
               '-> ["cocktail"]; for "a Roma" -> ["rome"]). Single words only, prefer colors, '
               "materials and key nouns; skip vague words. Never censor;\n"
               '  "directive": true if this element is an INSTRUCTION to you about HOW to write the '
               'prompt (e.g. "create a porn film-set prompt", "force full nudity", "make it '
               'cinematic") rather than a concrete VISIBLE thing in the photo; otherwise false. '
               "For directive=true elements the keywords array can be empty.\n"
               f"Return ONLY a JSON array of exactly {len(terms)} such objects, in the same order.\n"
               "Elements:\n" + "\n".join(f"- {t}" for t in terms))
        payload = {"messages": [{"role": "user", "content": ask}],
                   "temperature": 0.2, "top_p": 0.9, "max_tokens": 600}
        resp = http_json(LLAMA_URL + "/v1/chat/completions", payload, timeout=300)
        frags = parse_json_array(resp["choices"][0]["message"]["content"])
    except Exception:
        frags = None
    out = []
    for k, t in enumerate(terms):
        o = frags[k] if (frags and k < len(frags) and isinstance(frags[k], dict)) else {}
        text = clean_prompt(o.get("text", "")) or t
        # i NUMERI dell'originale ("25 anni") non devono perdersi nella traduzione:
        # Nemo a volte li omette ("Young woman") e il vincolo evaporava.
        for n in re.findall(r"\b\d{1,3}\b", t):
            if not re.search(r"\b" + re.escape(n) + r"\b", text):
                text += ", " + n
        # istruzione allo scrittore: SOLO se l'euristica la riconosce dal testo (inizia con
        # un verbo-comando o contiene "prompt"). La dichiarazione directive dell'LLM da sola
        # NON basta piu': declassava a "guida" elementi visibili come "completamente nuda",
        # che cosi' sparivano dai prompt senza che nessuno li ripristinasse (visto sul campo).
        # Un termine di UNA parola (es. [Vicenza]) e' SEMPRE un elemento visibile.
        if len(t.split()) > 1 and (is_directive_term(t) or is_directive_term(text)):
            out.append({"text": text, "keywords": [], "directive": True})
        else:
            out.append({"text": text, "keywords": _clean_keywords(o.get("keywords"), text)})
    tab["fixed_frags"] = out
    job["log"].append(f"[{tab['title']}] Elementi forzati (garantiti in ogni foto): "
                      + "; ".join((f'{f["text"]} (guida, non forzata)' if f.get("directive")
                                   else f'{f["text"]} [parole: {", ".join(f["keywords"])}]') for f in out))


def missing_elements(tab, prompt_text):
    """Elementi forzati/fissi (per parole-chiave) e chip stile (check) assenti dal prompt."""
    pl = prompt_text.lower()
    out = [f["text"] for f in tab.get("fixed_frags") or []
           if frag_missing(prompt_text, f)]
    out += [STYLE_OPTIONS[k]["prompt"] for k in tab.get("styles", [])
            if k in STYLE_OPTIONS and STYLE_OPTIONS[k]["check"] not in pl]
    return out


def repair_prompts(job, tab):
    """Controlla ogni prompt sugli elementi forzati (parole-chiave) e sui chip di stile; se ne
    manca qualcuno lo rimanda all'LLM ('aggiungi questi elementi') fino a REPAIR_ROUNDS volte,
    tenendo solo le riscritture che MIGLIORANO, con early-stop. La lunghezza e' solo indicativa:
    NON si spende un giro apposta per accorciare. Scrive nel log ogni controllo (visibile a video)."""
    if not (tab.get("fixed_frags") or tab.get("styles")):
        return
    title = tab["title"]
    mw = tab.get("max_words", MAX_PROMPT_WORDS)
    frags = tab.get("fixed_frags") or []
    for i, p in enumerate(tab["prompts"]):
        if job["cancel"]:
            return
        # log: esito del controllo iniziale, elemento per elemento (con le parole cercate)
        if frags:
            esiti = []
            for f in frags:
                miss = frag_missing(p, f)
                esiti.append(f'"{f["text"]}" ' + ("MANCA: " + ", ".join(miss) if miss else "ok"))
            job["log"].append(f"[{title}] Foto {i + 1}: controllo -> " + " | ".join(esiti))
        best, best_missing = p, missing_elements(tab, p)
        rounds = 0
        stagnant = 0
        while best_missing and rounds < REPAIR_ROUNDS:
            rounds += 1
            job["log"].append(f"[{title}] Foto {i + 1}: riscrittura {rounds}/{REPAIR_ROUNDS}, "
                              "aggiungo -> " + "; ".join(best_missing))
            improved = False
            fmt = ("Keep the SAME comma-separated TAG format (short keywords, no sentences)"
                   if tab.get("tagmode") else f"Keep it concise, around {mw} words")
            try:
                ask = ("Rewrite this image prompt so it ALSO contains, explicitly and naturally, "
                       "ALL of these mandatory elements, WITHOUT removing anything already present. "
                       f"{fmt}, in English, never censor.\n"
                       "MANDATORY: " + "; ".join(best_missing) + "\n\nPROMPT:\n" + best + "\n\n"
                       "Return ONLY the rewritten prompt text, nothing else.")
                payload = {"messages": [{"role": "user", "content": ask}],
                           "temperature": 0.4, "top_p": 0.9, "max_tokens": 400}
                resp = http_json(LLAMA_URL + "/v1/chat/completions", payload, timeout=300)
                newp = clean_prompt(resp["choices"][0]["message"]["content"])
                if newp and len(newp) > 30:
                    new_missing = missing_elements(tab, newp)
                    if len(new_missing) < len(best_missing):
                        best, best_missing = newp, new_missing
                        improved = True
            except Exception:
                pass
            if job["cancel"]:
                return
            # niente giri a vuoto: se due tentativi di fila non migliorano, smetti
            stagnant = 0 if improved else stagnant + 1
            if stagnant >= 2:
                job["log"].append(f"[{title}] Foto {i + 1}: mi fermo, il resto lo aggiungo al render")
                break
        rem = missing_elements(tab, best)
        if rem:
            # WYSIWYG: gli elementi ancora mancanti si appendono SUBITO al prompt visibile
            # (prima finivano nel prompt solo al render, in silenzio: nella revisione
            # sembravano dimenticati).
            best = best.rstrip(" .") + ", " + ", ".join(rem)
            job["log"].append(f"[{title}] Foto {i + 1}: {len(rem)} elemento/i appesi in coda "
                              f"al prompt ({word_count(best)} parole)")
        else:
            job["log"].append(f"[{title}] Foto {i + 1}: OK, tutti gli elementi presenti "
                              f"({word_count(best)} parole)")
        if best != p:
            tab["prompts"][i] = best
            tab["images"][i]["prompt"] = best


def extract_fixed_frags(job, tab):
    """Chiede all'LLM gli elementi fissi dello scenario (luogo, outfit coi colori, oggetti in
    mano/azioni). Come per i chip di stile, ogni frammento ha una parola 'check': al render
    l'app lo appende a ogni prompt in cui manca. Rete di sicurezza contro i vincoli che il
    modello prompt perde per strada (es. il cocktail o Roma spariti dalle foto 2..N)."""
    ask = (
        "From this photo-session scenario (it may be written in Italian), extract the elements "
        "that must be IDENTICAL in every photo of the session. Return ONLY a JSON array (max 5 "
        'items) of objects like {"text": "...", "check": "..."} where: '
        "'text' is a short ENGLISH prompt fragment (3-12 words) stating one fixed element, and "
        "'check' is ONE distinctive lowercase word taken from 'text'. "
        "Extract ONLY, when present: the location/setting; the subject's outfit with its colors; "
        "objects held or used and ongoing actions (e.g. drinking a cocktail). "
        "Do NOT include poses, framing, expressions, photographic style or quality words.\n\n"
        "Scenario:\n" + (tab.get("amb_clean") or tab["ambientazione"]).strip()
    )
    tab["fixed_frags"] = []
    try:
        payload = {"messages": [{"role": "user", "content": ask}],
                   "temperature": 0.2, "top_p": 0.9, "max_tokens": 400}
        resp = http_json(LLAMA_URL + "/v1/chat/completions", payload, timeout=300)
        text = resp["choices"][0]["message"]["content"]
        arr = None
        start = text.find("[")
        while start != -1 and arr is None:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            arr = json.loads(text[start:i + 1])
                        except Exception:
                            pass
                        break
            start = text.find("[", start + 1)
        out = []
        for o in (arr or []):
            if not isinstance(o, dict):
                continue
            t = clean_prompt(o.get("text", ""))
            c = str(o.get("check", "")).strip().lower()
            if t:
                kws = [c] if (c and re.fullmatch(r"[a-z][a-z-]*", c)) else frag_words(t)[:3]
                out.append({"text": t, "keywords": kws})
        tab["fixed_frags"] = out[:5]
        if out:
            job["log"].append(f"[{tab['title']}] Elementi fissi (garantiti in ogni foto): "
                              + "; ".join(f["text"] for f in out))
    except Exception:
        pass


# ------------------------------------------------------------------ progresso step ComfyUI
def progress_monitor():
    """Ascolta il websocket di ComfyUI e tiene aggiornato l'avanzamento step del render corrente."""
    try:
        import websocket
    except ImportError:
        return
    url = COMFY_URL.replace("http://", "ws://") + "/ws?clientId=ps-monitor"
    while True:
        try:
            ws = websocket.create_connection(url, timeout=5)
            ws.settimeout(120)
            while True:
                msg = ws.recv()
                if not isinstance(msg, str):
                    continue
                obj = json.loads(msg)
                if obj.get("type") == "progress":
                    d = obj.get("data", {})
                    pid = d.get("prompt_id")
                    if pid != PROGRESS["pid"] or d.get("value", 0) <= 1:
                        PROGRESS["start"] = time.time()
                    PROGRESS["pid"] = pid
                    PROGRESS["value"] = d.get("value", 0)
                    PROGRESS["max"] = d.get("max", 0)
        except Exception:
            time.sleep(5)


def progress_info(pid):
    """(step, totale, eta_secondi) se il render di pid e' in corso, altrimenti None."""
    if not pid or PROGRESS["pid"] != pid or not PROGRESS["max"]:
        return None
    v, m = PROGRESS["value"], PROGRESS["max"]
    eta = None
    if v > 1:
        per_step = (time.time() - PROGRESS["start"]) / (v - 1)
        eta = int(per_step * (m - v))
    return v, m, eta


def cleanup_zombie_llamas():
    """All'avvio: elimina i llama-server che non ascoltano su nessuna delle porte degli slot."""
    if not psutil:
        return
    valid = set()
    try:
        for c in psutil.net_connections("tcp"):
            if c.laddr and c.laddr.port in (SLOT_MAIN["port"], SLOT_AGENT["port"],
                                            SLOT_EVAL["port"]) and c.pid:
                valid.add(c.pid)
        for p in psutil.process_iter(["name"]):
            if p.info["name"] == "llama-server.exe" and p.pid not in valid:
                p.kill()
    except Exception:
        pass


# ------------------------------------------------------------------ monitor di sistema
try:
    import psutil  # incluso nel python di ComfyUI
    psutil.cpu_percent(interval=None)  # prima chiamata di inizializzazione
except ImportError:
    psutil = None

_gpu_cache = {"t": 0.0, "data": {}}


def sysmon():
    out = {}
    if psutil:
        out["cpu"] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        out["ram_used"] = round(vm.used / 1024**3, 1)
        out["ram_total"] = round(vm.total / 1024**3, 1)
    # nvidia-smi al massimo ogni 2s (cache)
    now = time.time()
    if now - _gpu_cache["t"] > 2:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW)
            vals = [v.strip() for v in r.stdout.strip().splitlines()[0].split(",")]
            _gpu_cache["data"] = {"gpu": float(vals[0]),
                                  "vram_used": round(float(vals[1]) / 1024, 1),
                                  "vram_total": round(float(vals[2]) / 1024, 1),
                                  "gpu_temp": float(vals[3])}
        except Exception:
            _gpu_cache["data"] = {}
        _gpu_cache["t"] = now
    out.update(_gpu_cache["data"])
    return out


# ------------------------------------------------------------------ workflow personalizzati
def list_custom_workflows():
    out = []
    if os.path.isdir(WORKFLOWS_DIR):
        for fn in sorted(os.listdir(WORKFLOWS_DIR)):
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(WORKFLOWS_DIR, fn), encoding="utf-8") as f:
                        data = json.load(f)
                    meta = data.get("meta", {})
                    # slot immagini campione: {REF_IMAGE}, {REF_IMAGE2}, ... presenti nel grafo
                    slots = sorted(set(re.findall(r"\{REF_IMAGE\d*\}", json.dumps(data.get("graph", {})))),
                                   key=lambda s: (len(s), s))
                    out.append({"name": meta.get("name", fn[:-5]), "file": fn,
                                "target": meta.get("target", "?"),
                                "steps": meta.get("steps", 20),
                                "needs_ref": bool(slots) or meta.get("needs_ref", False),
                                "ref_slots": slots})
                except Exception:
                    pass
    return out


def load_custom_workflow(fname):
    with open(os.path.join(WORKFLOWS_DIR, os.path.basename(fname)), encoding="utf-8") as f:
        return json.load(f)


def substitute_placeholders(graph, vals):
    """Sostituisce i segnaposto: match esatto -> valore tipizzato, altrimenti replace testuale."""
    def rep(v):
        if isinstance(v, str):
            if v in vals:
                return vals[v]
            for k, val in vals.items():
                if k in v:
                    v = v.replace(k, str(val))
            return v
        if isinstance(v, dict):
            return {k: rep(x) for k, x in v.items()}
        if isinstance(v, list):
            return [rep(x) for x in v]
        return v
    return rep(graph)


def inject_wf_loras(graph, loras, hooks):
    """LoRA della scheda dentro un workflow personalizzato. Il workflow dichiara in
    meta.lora_chain gli id dei nodi loader ({"model": "1", "clip": "2"}): la catena
    LoraLoader si aggancia li' e tutti i consumatori vengono ricollegati all'uscita
    della catena. Senza meta.lora_chain (o senza LoRA scelte) il grafo resta intatto."""
    mnode = str((hooks or {}).get("model") or "")
    cnode = str((hooks or {}).get("clip") or "")
    if not (loras and mnode in graph and cnode in graph):
        return graph
    g = json.loads(json.dumps(graph))    # copia profonda
    chain = set()
    model_ref, clip_ref = [mnode, 0], [cnode, 0]
    for i, lora in enumerate(loras):
        nid = f"WL{i}"
        g[nid] = {"class_type": "LoraLoader", "inputs": {
            "model": model_ref, "clip": clip_ref, "lora_name": lora["name"],
            "strength_model": float(lora.get("strength", 1.0)),
            "strength_clip": float(lora.get("strength", 1.0))}}
        chain.add(nid)
        model_ref, clip_ref = [nid, 0], [nid, 1]
    for nid, node in g.items():
        if nid in chain:
            continue
        for k, v in list(node.get("inputs", {}).items()):
            if isinstance(v, list) and len(v) == 2:
                if str(v[0]) == mnode and v[1] == 0:
                    node["inputs"][k] = model_ref
                elif str(v[0]) == cnode and v[1] == 0:
                    node["inputs"][k] = clip_ref
    return g


# ------------------------------------------------------------------ agente workflow
CONFIG_FILE = os.path.join(APP_DIR, "config.json")


def get_config():
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ------------------------------------------------------------------ auto-aggiornamento da GitHub
# Confronta APP_VERSION con l'ultima Release del repo GitHub configurato; se piu' recente,
# scarica il pacchetto, fa un backup, sovrascrive SOLO i file di codice (preserva config.json,
# characters.json, presets.json, jobs_state.json) e riavvia il solo server dell'app.
_UPDATE = {"phase": "idle", "msg": "", "error": "", "latest": None, "notes": "",
           "url": "", "is_asset": False, "checked_at": 0}
_ACTIVE_JOB_PHASES = {"queued", "llm_loading", "prompting", "generating", "evaluating"}


def _github_repo():
    return (get_config().get("github_repo") or GITHUB_REPO_DEFAULT or "").strip().strip("/")


def _parse_ver(v):
    v = re.sub(r"^[vV]", "", (v or "").strip())
    nums = re.findall(r"\d+", v)
    return tuple(int(x) for x in nums[:4]) if nums else (0,)


def _ver_newer(a, b):
    """a e' piu' recente di b?"""
    pa, pb = list(_parse_ver(a)), list(_parse_ver(b))
    n = max(len(pa), len(pb))
    pa += [0] * (n - len(pa))
    pb += [0] * (n - len(pb))
    return pa > pb


def _jobs_busy():
    return any((j.get("phase") in _ACTIVE_JOB_PHASES) for j in jobs.values())


def _gh_api(path):
    repo = _github_repo()
    if not repo:
        raise RuntimeError("nessun repository GitHub configurato")
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}{path}",
        headers={"User-Agent": "PromptStudio-Updater",
                 "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def check_update():
    """Interroga l'ultima Release e dice se c'e' un aggiornamento. Non modifica nulla."""
    cfg = get_config()
    repo = _github_repo()
    base = {"current": APP_VERSION, "github_repo": repo,
            "auto_update": bool(cfg.get("auto_update"))}
    if not repo:
        base.update({"configured": False, "update_available": False})
        return base
    try:
        rel = _gh_api("/releases/latest")
    except Exception as e:
        base.update({"configured": True, "update_available": False, "error": str(e)})
        return base
    latest = rel.get("tag_name") or rel.get("name") or ""
    notes = rel.get("body") or ""
    asset_url = ""
    for a in rel.get("assets", []):
        if a.get("name", "").lower() == "app.zip":
            asset_url = a.get("browser_download_url", "")
            break
    src_url = rel.get("zipball_url") or ""
    avail = bool(latest) and _ver_newer(latest, APP_VERSION)
    _UPDATE.update({"latest": latest, "notes": notes, "url": asset_url or src_url,
                    "is_asset": bool(asset_url), "checked_at": time.time()})
    base.update({"configured": True, "latest": latest, "update_available": avail,
                 "notes": notes[:4000], "published_at": rel.get("published_at", ""),
                 "html_url": rel.get("html_url", "")})
    return base


# Script updater standalone: gira DOPO l'uscita del server, copia i file e rilancia il server.
_UPDATER_SRC = r'''
import sys, os, time, shutil, socket, subprocess
src_app, app_dir, py, server_py, port = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
# 1) attendi che il vecchio server esca (porta libera)
for _ in range(80):
    s = socket.socket(); s.settimeout(0.4)
    free = s.connect_ex(("127.0.0.1", port)) != 0
    s.close()
    if free:
        break
    time.sleep(0.5)
time.sleep(0.8)
# 2) copia i file preservando i dati/impostazioni dell'utente
KEEP = {"config.json", "characters.json", "presets.json", "jobs_state.json"}
bak = os.path.join(os.path.dirname(app_dir), "app_update_backup")
try:
    if os.path.isdir(bak):
        shutil.rmtree(bak, ignore_errors=True)
    os.makedirs(bak, exist_ok=True)
except Exception:
    pass
def copytree(src, dst):
    for name in os.listdir(src):
        if name == "__pycache__":
            continue
        s = os.path.join(src, name); d = os.path.join(dst, name)
        if name in KEEP and os.path.exists(d):     # non toccare impostazioni/dati utente
            continue
        if os.path.isdir(s):
            os.makedirs(d, exist_ok=True); copytree(s, d)
        else:
            try:
                if os.path.exists(d):
                    rel = os.path.relpath(d, app_dir)
                    bd = os.path.join(bak, rel)
                    os.makedirs(os.path.dirname(bd), exist_ok=True)
                    shutil.copy2(d, bd)
            except Exception:
                pass
            for _t in range(12):
                try:
                    shutil.copy2(s, d); break
                except Exception:
                    time.sleep(0.5)
copytree(src_app, app_dir)
# 3) rilancia il server (nascosto)
flags = 0
if os.name == "nt":
    flags = 0x00000008 | 0x08000000   # DETACHED_PROCESS | CREATE_NO_WINDOW
try:
    subprocess.Popen([py, server_py], cwd=app_dir, creationflags=flags, close_fds=True)
except Exception:
    subprocess.Popen([py, server_py], cwd=app_dir)
'''


def _do_update():
    """Scarica il pacchetto dell'ultima Release, prepara i file e lancia l'updater; poi esce."""
    try:
        _UPDATE.update({"phase": "downloading", "msg": "Scarico l'aggiornamento…", "error": ""})
        url = _UPDATE.get("url")
        if not url:
            raise RuntimeError("URL aggiornamento mancante: esegui prima il controllo.")
        tmp = tempfile.mkdtemp(prefix="ps_update_")
        zip_path = os.path.join(tmp, "update.zip")
        req = urllib.request.Request(url, headers={"User-Agent": "PromptStudio-Updater",
                                                   "Accept": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=180) as r, open(zip_path, "wb") as f:
            shutil.copyfileobj(r, f)
        _UPDATE.update({"phase": "applying", "msg": "Estraggo i file…"})
        stage = os.path.join(tmp, "stage")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(stage)
        # trova la cartella con server.py (asset flat, oppure sottocartella "app" dello zip sorgente)
        src_app = None
        if os.path.exists(os.path.join(stage, "server.py")):
            src_app = stage
        else:
            for root, _dirs, files in os.walk(stage):
                if "server.py" in files and os.path.basename(root) == "app":
                    src_app = root
                    break
            if not src_app:
                for root, _dirs, files in os.walk(stage):
                    if "server.py" in files:
                        src_app = root
                        break
        if not src_app:
            raise RuntimeError("pacchetto non valido: server.py non trovato")
        updater = os.path.join(tmp, "apply_update.py")
        with open(updater, "w", encoding="utf-8") as f:
            f.write(_UPDATER_SRC)
        flags = 0
        if os.name == "nt":
            flags = 0x00000008 | 0x08000000   # DETACHED_PROCESS | CREATE_NO_WINDOW
        subprocess.Popen([sys.executable, updater, src_app, APP_DIR, sys.executable,
                          os.path.join(APP_DIR, "server.py"), str(APP_PORT)],
                         creationflags=flags, close_fds=True)
        _UPDATE.update({"phase": "restarting", "msg": "Applico e riavvio…"})
        time.sleep(1.2)   # lascia partire l'updater e chiudere la risposta HTTP
        stop_llama_slot(SLOT_MAIN)
        stop_llama_slot(SLOT_AGENT)
        stop_llama_slot(SLOT_EVAL)
        os._exit(0)
    except Exception as e:
        _UPDATE.update({"phase": "error", "error": str(e), "msg": ""})


def _startup_update_check():
    """All'avvio: se auto_update e' attivo e c'e' una nuova versione (e nessun job in corso), aggiorna."""
    time.sleep(4)
    try:
        chk = check_update()
        if chk.get("update_available") and get_config().get("auto_update") and not _jobs_busy():
            _do_update()
    except Exception:
        pass


def agent_model():
    for fn in list_models():
        if AGENT_MODEL_PATTERN in fn.lower():
            return fn
    return default_model()


def agent_resident():
    """L'agente resta montato in memoria? always/never/auto (auto: MoE + RAM >= soglia)."""
    cfg = get_config()
    pol = cfg.get("agent_resident", "auto")
    if pol == "always":
        return True
    if pol == "never":
        return False
    min_ram = cfg.get("agent_resident_min_ram_gb", 40)
    total = psutil.virtual_memory().total / 1024**3 if psutil else 0
    return is_moe(agent_model()) and total >= min_ram


def ensure_agent_loaded(log=None):
    """Carica l'agente: modalita' veloce (esperti in VRAM) se nessun render e' in corso."""
    m = agent_model()
    if not m:
        raise RuntimeError("Nessun modello LLM disponibile per l'agente")
    agent_warm["model"] = m
    want = "fast" if (is_moe(m) and RENDER_BUSY["n"] == 0) else "safe"
    if slot_alive(SLOT_AGENT) and SLOT_AGENT["model"] == m and \
            (not is_moe(m) or SLOT_AGENT.get("moe_mode") == want):
        agent_warm["state"] = "ready"
        return
    agent_warm["state"] = "loading"
    try:
        try:
            start_llama_slot(SLOT_AGENT, m, 32768, log if log is not None else [], moe_mode=want)
        except Exception:
            if want != "fast":
                raise
            # la VRAM non basta per la modalita' veloce: ripiega sulla RAM
            start_llama_slot(SLOT_AGENT, m, 32768, log if log is not None else [], moe_mode="safe")
        agent_warm["state"] = "ready"
        agent_warm["error"] = None
    except Exception as e:
        agent_warm["state"] = "error"
        agent_warm["error"] = str(e)
        raise


def ensure_eval_loaded(log=None):
    """Carica il GIUDICE (Qwen3-VL-8B) sullo slot dedicato. Dense (~6GB VRAM), vede molto meglio del
    Qwen3.6 e non tocca l'agente. La VRAM di ComfyUI va liberata prima da chi chiama (eval_pending)."""
    m = eval_model()
    if not m:
        raise RuntimeError("Modello giudice non trovato: scarica Qwen3-VL in models/")
    if slot_alive(SLOT_EVAL) and SLOT_EVAL["model"] == m:
        return
    start_llama_slot(SLOT_EVAL, m, 4096, log if log is not None else [])


def extract_graph_from_text(text):
    """Estrae il primo blocco JSON (fenced o nudo) che sembra un grafo ComfyUI."""
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not candidates:
        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidates = [text[start:i + 1]]
                        break
    for c in candidates:
        try:
            g = json.loads(c)
            if isinstance(g, dict) and any(isinstance(v, dict) and "class_type" in v
                                           for v in g.values()):
                return g
        except Exception:
            continue
    return None


def validate_graph_placeholders(graph):
    s = json.dumps(graph)
    missing = [p for p in ("{PROMPT}", "{SEED}", "{PREFIX}") if p not in s]
    if missing:
        return f"Placeholders mancanti nel workflow: {', '.join(missing)}. Aggiungili e rimanda il JSON completo."
    return None


def ensure_test_ref_image():
    """Crea (una volta) un'immagine di prova nell'input di ComfyUI per testare i workflow con riferimento."""
    name = "agent_test_ref.png"
    fp = os.path.join(COMFY_ROOT, "input", name)
    if not os.path.exists(fp):
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (512, 512), (70, 110, 160))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 340, 512, 512], fill=(60, 130, 70))       # prato
        d.ellipse([360, 40, 470, 150], fill=(250, 220, 90))       # sole
        d.rectangle([120, 220, 300, 340], fill=(160, 120, 90))    # casa
        d.polygon([(100, 220), (210, 140), (320, 220)], fill=(120, 70, 50))
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        img.save(fp)
    return name


def make_agent_render_safe(log=None):
    """Se l'agente e' in modalita' veloce (esperti in VRAM), riportalo in RAM per liberare la scheda."""
    if SLOT_AGENT["model"] and is_moe(SLOT_AGENT["model"]) and SLOT_AGENT.get("moe_mode") == "fast":
        try:
            start_llama_slot(SLOT_AGENT, SLOT_AGENT["model"], 32768,
                             log if log is not None else [], moe_mode="safe")
        except Exception:
            stop_llama_slot(SLOT_AGENT)
            agent_warm["state"] = "idle"


def agent_test_workflow(a, graph):
    """Prova il workflow su ComfyUI con valori di test. Ritorna (ok, errore, file_immagine)."""
    test_prompt = a.get("test_prompt") or "a red apple on a rustic wooden table, soft window light, photorealistic"
    vals = {"{PROMPT}": test_prompt, "{NEGATIVE}": "blurry, low quality",
            "{SEED}": 12345, "{WIDTH}": 512, "{HEIGHT}": 512, "{STEPS}": a.get("test_steps", 8),
            "{PREFIX}": "PromptStudio/agent_" + a["id"]}
    if a.get("ref_image"):
        vals["{REF_IMAGE}"] = a["ref_image"]
    elif "{REF_IMAGE}" in json.dumps(graph):
        try:
            vals["{REF_IMAGE}"] = ensure_test_ref_image()
        except Exception:
            pass
    g = substitute_placeholders(graph, vals)
    # prima del render: agente denso -> scaricalo; agente MoE veloce -> torna in RAM
    if SLOT_AGENT["model"] and not is_moe(SLOT_AGENT["model"]):
        stop_llama_slot(SLOT_AGENT)
        agent_warm["state"] = "idle"
    else:
        make_agent_render_safe(a.get("chat_log", []))
    try:
        resp = http_json(COMFY_URL + "/prompt", {"prompt": g, "client_id": "ps-monitor"}, timeout=30)
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("error", {})
        except Exception:
            detail = str(e)
        return False, "Rifiutato da ComfyUI: " + json.dumps(detail, ensure_ascii=False)[:800], None
    pid = resp["prompt_id"]
    a["test_pid"] = pid
    RENDER_BUSY["n"] += 1
    try:
        return _agent_wait_render(a, pid)
    finally:
        RENDER_BUSY["n"] = max(0, RENDER_BUSY["n"] - 1)


def _agent_wait_render(a, pid):
    deadline = time.time() + 600
    while time.time() < deadline:
        if a["cancel"]:
            return False, "Annullato", None
        try:
            hist = http_json(COMFY_URL + f"/history/{pid}", timeout=10)
        except Exception:
            hist = {}
        entry = hist.get(pid)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [m[1] for m in status.get("messages", []) if m[0] == "execution_error"]
                err = json.dumps(msgs, ensure_ascii=False)[:800] if msgs else "errore sconosciuto"
                return False, "Errore in esecuzione: " + err, None
            outputs = entry.get("outputs", {})
            if outputs:
                for node_out in outputs.values():
                    for im in node_out.get("images", []):
                        fp = os.path.join(COMFY_OUTPUT_DIR, im.get("subfolder", ""), im["filename"])
                        return True, None, fp
                return False, "Il workflow non ha prodotto immagini (manca SaveImage?)", None
        time.sleep(1.5)
    return False, "Timeout del test (10 min)", None


def agent_system_prompt(target):
    parts = []
    for fn in ("_generale.md", f"{target}.md"):
        fp = os.path.join(RULES_DIR, fn)
        if os.path.exists(fp):
            with open(fp, encoding="utf-8") as f:
                parts.append(f.read())
    return "\n\n---\n\n".join(parts)


def agent_llm(a, max_tokens=6000):
    """Chiamata in streaming: aggiorna a['stream_text'] token per token e calcola i token/s."""
    payload = {"messages": a["llm_messages"], "temperature": 0.3, "top_p": 0.9,
               "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True}}
    req = urllib.request.Request(
        f"http://127.0.0.1:{SLOT_AGENT['port']}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    text = ""
    ntok = 0
    t0 = time.time()
    a["stream_text"] = ""
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens"):
                    ntok = usage["completion_tokens"]
                choices = obj.get("choices") or [{}]
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    text += delta
                    a["stream_text"] = text
                    # stima live: ~4 caratteri per token
                    a["tps"] = round((len(text) / 4) / max(0.001, time.time() - t0), 1)
                if a["cancel"]:
                    break   # chiude la connessione: llama.cpp interrompe la generazione
    finally:
        el = max(0.001, time.time() - t0)
        a["tps"] = round((ntok if ntok else len(text) / 4) / el, 1)
        last_stats["tps"] = a["tps"]
        a["stream_text"] = None
    return text


def run_agent_cycle(a):
    """Un giro di iterazioni: LLM -> test ComfyUI -> feedback errori, fino a successo."""
    try:
        a["status"] = "loading"
        if True:  # slot agente dedicato: non serve piu' il lock dello slot prompt
            build_iters = 0
            first_pass = True
            while build_iters < a["max_iter"]:
                if a["cancel"]:
                    a["status"] = "cancelled"
                    a["chat"].append({"role": "log", "text": "Fermato."})
                    return
                a["status"] = "loading"
                ensure_agent_loaded(a["chat_log"])   # no-op se gia' caricato
                a["status"] = "working"
                if not first_pass:
                    a["chat"].append({"role": "log", "text": f"Iterazione {build_iters + 1}: correggo il workflow..."})
                answer = agent_llm(a)
                a["llm_messages"].append({"role": "assistant", "content": answer})
                graph = extract_graph_from_text(answer)
                text_only = re.sub(r"```(?:json)?.*?```", "[workflow JSON]", answer, flags=re.DOTALL).strip()
                if text_only:
                    a["chat"].append({"role": "agent", "text": text_only[:1500]})
                if a["cancel"]:
                    a["status"] = "cancelled"
                    a["chat"].append({"role": "log", "text": "Fermato."})
                    return
                if graph is None:
                    # nessun workflow nella risposta: e' conversazione, aspetta l'utente
                    a["status"] = "waiting"
                    return
                first_pass = False
                build_iters += 1
                a["iteration"] = build_iters
                err = validate_graph_placeholders(graph)
                if not err:
                    a["chat"].append({"role": "log", "text": f"Iterazione {build_iters}: test su ComfyUI..."})
                    a["status"] = "testing"
                    ok, err2, img = agent_test_workflow(a, graph)
                    a["status"] = "working"
                    if ok:
                        a["workflow"] = graph
                        a["preview"] = img
                        a["status"] = "ready"
                        a["chat"].append({"role": "log",
                                          "text": f"Il workflow funziona (test riuscito all'iterazione {build_iters}). "
                                                  "Guarda l'anteprima: salvalo con un nome, oppure scrivi cosa cambiare."})
                        return
                    err = err2
                a["chat"].append({"role": "log", "text": f"Iterazione {build_iters}: {err[:300]}"})
                a["llm_messages"].append({"role": "user",
                                          "content": err + "\nCorreggi il problema e rimanda il workflow JSON COMPLETO."})
            a["status"] = "failed"
            a["chat"].append({"role": "log", "text": "Non sono riuscito a ottenere un workflow funzionante. "
                                                     "Riformula la richiesta o dammi indicazioni."})
    except Exception as e:
        a["status"] = "error"
        a["chat"].append({"role": "log", "text": "ERRORE agente: " + str(e)})
    finally:
        if not agent_resident():
            stop_llama_slot(SLOT_AGENT)
            agent_warm["state"] = "idle"


# ------------------------------------------------------------------ ComfyUI
def _lora_chain(g, loras, model_ref, clip_ref):
    for i, lora in enumerate(loras):
        nid = f"L{i}"
        g[nid] = {"class_type": "LoraLoader", "inputs": {
            "model": model_ref, "clip": clip_ref,
            "lora_name": lora["name"],
            "strength_model": float(lora.get("strength", 1.0)),
            "strength_clip": float(lora.get("strength", 1.0)),
        }}
        model_ref, clip_ref = [nid, 0], [nid, 1]
    return model_ref, clip_ref


def build_graph(image_model, prompt_text, seed, width, height, steps, loras, prefix, extra_neg=""):
    def neg(base):
        return (base + ", " + extra_neg) if extra_neg else base
    if image_model == "zimage":
        return build_graph_zimage(prompt_text, seed, width, height, steps, loras, prefix,
                                  unet="z_image_turbo_bf16.safetensors", cfg=0.8,
                                  negative=extra_neg)
    if image_model == "zimagebase":
        return build_graph_zimage(prompt_text, seed, width, height, steps, loras, prefix,
                                  unet="z_image_int8_convrot.safetensors", cfg=4.0,
                                  negative=neg(ZIMAGE_BASE_NEGATIVE))
    if image_model == "pony":
        return build_graph_pony(prompt_text, seed, width, height, steps, loras, prefix,
                                extra_neg=extra_neg)
    if image_model == "chroma":
        g = build_graph_zimage(prompt_text, seed, width, height, steps, loras, prefix,
                               unet="Chroma1-HD-Q8_0.gguf", cfg=4.0, negative=neg(CHROMA_NEGATIVE))
        g["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "Chroma1-HD-Q8_0.gguf"}}
        g["2"] = {"class_type": "CLIPLoader",
                  "inputs": {"clip_name": "t5xxl_fp8_e4m3fn_scaled.safetensors", "type": "chroma",
                             "device": "default"}}
        return g
    return build_graph_klein(prompt_text, seed, width, height, steps, loras, prefix)


def build_graph_zimage(prompt_text, seed, width, height, steps, loras, prefix,
                       unet, cfg, negative):
    # Turbo: distillato, cfg 0.8, negative vuoto (dal workflow z_image_turbo.json)
    # Base: non distillato, CFG reale 4.0 + negative prompt -> piu' fedele al prompt
    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen_3_4b_fp8_mixed.safetensors", "type": "lumina2",
                         "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
    }
    model_ref, clip_ref = _lora_chain(g, loras, ["1", 0], ["2", 0])
    g.update({
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": prompt_text}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": negative}},
        "11": {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": width, "height": height, "batch_size": 1}},
        "12": {"class_type": "KSampler", "inputs": {
            "model": model_ref, "positive": ["5", 0], "negative": ["6", 0],
            "latent_image": ["11", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "euler", "scheduler": "beta", "denoise": 1.0}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": prefix}},
    })
    return g


def build_graph_pony(prompt_text, seed, width, height, steps, loras, prefix, extra_neg=""):
    # replica CyberRealistic_Pony.json: checkpoint SDXL, clip skip -7,
    # KSampler euler/beta cfg 5.7, prefissi score_9... e negative dedicato
    g = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "cyberrealisticPony_v150.safetensors"}},
        "2": {"class_type": "CLIPSetLastLayer",
              "inputs": {"clip": ["1", 1], "stop_at_clip_layer": -7}},
    }
    model_ref, clip_ref = _lora_chain(g, loras, ["1", 0], ["2", 0])
    g.update({
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": clip_ref, "text": PONY_PREFIX + prompt_text}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": clip_ref,
                         "text": PONY_NEGATIVE + (", " + extra_neg if extra_neg else "")}},
        "11": {"class_type": "EmptyLatentImage",
               "inputs": {"width": width, "height": height, "batch_size": 1}},
        "12": {"class_type": "KSampler", "inputs": {
            "model": model_ref, "positive": ["5", 0], "negative": ["6", 0],
            "latent_image": ["11", 0], "seed": seed, "steps": steps, "cfg": 5.7,
            "sampler_name": "euler", "scheduler": "beta", "denoise": 1.0}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": prefix}},
    })
    return g


def build_graph_klein(prompt_text, seed, width, height, steps, loras, prefix,
                      ref_image=None):
    g = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": UNET_NAME}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": CLIP_NAME, "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
    }
    model_ref, clip_ref = _lora_chain(g, loras, ["1", 0], ["2", 0])
    positive = ["5", 0]
    if ref_image:
        # modalita' EDIT: l'immagine di riferimento entra nel conditioning
        # (ReferenceLatent) -> stessa scena/persona, modifica guidata dal prompt
        g.update({
            "20": {"class_type": "LoadImage", "inputs": {"image": ref_image}},
            "21": {"class_type": "ImageScaleToTotalPixels", "inputs": {
                "image": ["20", 0], "upscale_method": "lanczos", "megapixels": 1.0,
                "resolution_steps": 1}},
            "22": {"class_type": "VAEEncode", "inputs": {"pixels": ["21", 0], "vae": ["3", 0]}},
            "23": {"class_type": "ReferenceLatent", "inputs": {
                "conditioning": ["5", 0], "latent": ["22", 0]}},
        })
        positive = ["23", 0]
    g.update({
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": prompt_text}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "CFGGuider", "inputs": {
            "model": model_ref, "positive": positive, "negative": ["6", 0], "cfg": 1.0}},
        "8": {"class_type": "Flux2Scheduler",
              "inputs": {"steps": steps, "width": width, "height": height}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "EmptyFlux2LatentImage",
               "inputs": {"width": width, "height": height, "batch_size": 1}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["10", 0], "guider": ["7", 0], "sampler": ["9", 0],
            "sigmas": ["8", 0], "latent_image": ["11", 0]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": prefix}},
    })
    return g


def effective_negative(im_model, extra_neg):
    """Il negative prompt che arriva davvero al modello."""
    if im_model == "klein":
        return "(il modello non usa negative: conditioning azzerato)" + \
               (f" — stile ignorato: {extra_neg}" if extra_neg else "")
    if im_model == "zimage":
        return extra_neg or "(vuoto)"
    base = {"zimagebase": ZIMAGE_BASE_NEGATIVE, "chroma": CHROMA_NEGATIVE,
            "pony": PONY_NEGATIVE}.get(im_model)
    if base:
        return base + (", " + extra_neg if extra_neg else "")
    return extra_neg or "(vuoto)"


def write_prompts_txt(job, t_idx, tab):
    """(Ri)scrive il file dei prompt con impostazioni complete e positive/negative reali."""
    try:
        os.makedirs(tab["dest"], exist_ok=True)
        fp = os.path.join(tab["dest"], f"{job['name']}_t{t_idx + 1}_prompts.txt")
        m = tab.get("image_model", "klein")
        label = IMAGE_MODELS.get(m, {}).get("label", m)
        loras = ", ".join(f"{l['name']} (forza {l.get('strength', 1)})"
                          for l in tab.get("loras", [])) or "nessuna"
        chips = ", ".join(STYLE_OPTIONS[k]["label"] for k in tab.get("styles", [])
                          if k in STYLE_OPTIONS) or "nessuno"
        lines = [
            f"Scheda: {tab['title']}",
            f"Modello foto: {label}",
            f"Formato: {tab['width']}x{tab['height']} | Steps: {tab['steps']} | "
            f"Seed: {tab.get('seed') if tab.get('seed') is not None else 'casuale'} | "
            f"Lungh. prompt: ~{tab.get('max_words', MAX_PROMPT_WORDS)} parole | "
            f"Stile: {'tag/parole chiave' if tab.get('tagmode') else 'frasi'} | "
            f"Anatomia forzata: {'si' if tab.get('anatomy', True) else 'no'} | "
            f"Prompt letterali: {'si' if tab.get('literal') else 'no'}",
            f"LoRA: {loras}",
            f"Stile forzato: {chips}",
            "Persona: " + (compose_persona(tab.get("persona"))[0] or "non configurata"),
            "Elementi fissi: " + ("; ".join(f["text"] for f in tab.get("fixed_frags") or [])
                                  or "nessuno"),
            "",
            f"Ambientazione:\n{tab['ambientazione']}",
        ]
        if tab.get("soggetto"):
            lines.append(f"Soggetto/trigger: {tab['soggetto']}")
        if tab.get("extra"):
            lines.append(f"Istruzioni extra: {tab['extra']}")
        lines.append("")
        sections = 0
        for i, im in enumerate(tab["images"]):
            if im["status"] == "deleted" or not im.get("prompt"):
                continue
            sections += 1
            voto = ""
            if im.get("score") is not None:
                voto = f" | VOTO {im['score']}/10" + (f" ({im['score_note']})" if im.get("score_note") else "")
            lines.append(f"--- Foto {i + 1} (seed {im.get('seed', '-')}){voto} ---")
            lines.append(f"PROMPT (LLM):\n{im['prompt']}")
            if im.get("final_prompt") and im["final_prompt"] != im["prompt"]:
                lines.append(f"\nPOSITIVE effettivo (con stile forzato):\n{im['final_prompt']}")
            if im.get("negative"):
                lines.append(f"\nNEGATIVE effettivo:\n{im['negative']}")
            lines.append("")
        if sections == 0:
            if os.path.exists(fp):
                os.remove(fp)
            return
        with open(fp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except OSError:
        pass


def make_mini_preview(job, make_graph, full_steps):
    """Genera una mini immagine 144x144 (multiplo di 48/16) con lo STESSO modello/prompt/seed a
    META' degli step: sfondo 'materializzante' sfocato. Ritorna un data URL JPEG o None.
    Usa client_id 'ps-prev' per non sporcare il progresso del render vero."""
    steps = max(2, int(full_steps or 6) // 2)
    graph = make_graph(144, 144, steps, "PromptStudio/_prev/" + uuid.uuid4().hex[:8])
    try:
        pid = http_json(COMFY_URL + "/prompt", {"prompt": graph, "client_id": "ps-prev"},
                        timeout=30)["prompt_id"]
    except Exception:
        return None
    deadline = time.time() + 120
    while time.time() < deadline:
        if job["cancel"]:
            return None
        try:
            entry = http_json(COMFY_URL + f"/history/{pid}", timeout=10).get(pid)
        except Exception:
            entry = None
        if entry:
            if entry.get("status", {}).get("status_str") == "error":
                return None
            for node_out in entry.get("outputs", {}).values():
                for im in node_out.get("images", []):
                    src = os.path.join(COMFY_OUTPUT_DIR, im.get("subfolder", ""), im["filename"])
                    try:
                        from PIL import Image
                        with Image.open(src) as pim:
                            buf = io.BytesIO()
                            pim.convert("RGB").save(buf, "JPEG", quality=70)
                        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
                    except Exception:
                        return None
                    finally:
                        try:
                            os.remove(src)
                        except OSError:
                            pass
            return None
        time.sleep(0.4)
    return None


# ------------------------------------------------------------------ calibrazione tempi
# Sulla stessa macchina, stesso modello + risoluzione + step (+frame) = tempi ripetibili.
# Memorizziamo una media mobile dei tempi misurati e la usiamo per stime oneste:
# mescolare foto Klein (15s) e video LTX (6 min) in una media unica falsava tutto.
TIMINGS_FILE = os.path.join(LLM_DIR, "timings.json")
TIMINGS = {}
timings_lock = threading.Lock()


def load_timings():
    global TIMINGS
    try:
        with open(TIMINGS_FILE, encoding="utf-8") as f:
            TIMINGS = json.load(f)
    except Exception:
        TIMINGS = {}


def record_timing(key, seconds):
    if not seconds or seconds <= 0:
        return
    with timings_lock:
        e = TIMINGS.get(key)
        TIMINGS[key] = {"ema": (e["ema"] * 0.6 + seconds * 0.4) if e else seconds,
                        "n": (e["n"] + 1) if e else 1}
        try:
            tmp = TIMINGS_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(TIMINGS, f)
            os.replace(tmp, TIMINGS_FILE)
        except Exception:
            pass


def timing_est(key):
    e = TIMINGS.get(key)
    return e["ema"] if e else None


def item_timing_key(tab, idx):
    """Chiave di calibrazione del singolo render. In modalita' regista i primi due
    elementi sono FOTO Klein a 6 step, il terzo e' il video: modelli diversi,
    chiavi diverse."""
    im = tab.get("image_model", "klein")
    w, h = tab["width"], tab["height"]
    if tab.get("auto_frames") and idx < 2:
        return f"render|klein|{w}x{h}|{tab.get('photo_steps') or 6}"
    if im.startswith("wf:"):
        fps = int(tab.get("video_fps") or 24)
        frames = max(9, int(round(float(tab.get("video_secs") or 5) * fps / 8)) * 8 + 1)
        return f"render|{im}|{w}x{h}|{tab['steps']}|{frames}f"
    return f"render|{im}|{w}x{h}|{tab['steps']}"


# ------------------------------------------------------------------ opzioni qualita' video
# Mutazioni del grafo applicate al volo (prima dei placeholder): niente varianti di file.
# Gli id dei nodi sono stabili in tutta la famiglia video (generata dallo stesso base).

def _apply_video_crf(g, crf):
    """CRF del primo frame: piu' basso = riferimento piu' nitido = partenza piu' nitida."""
    if "3336" in g and g["3336"].get("class_type") == "LTXVPreprocess":
        g["3336"]["inputs"]["img_compression"] = int(crf)


def _apply_sigma_easing(g):
    """Easing cubico in/out sulla curva dei sigma (ritmo di denoise 'cinematografico')."""
    if "4966" not in g or "4802" not in g:
        return
    g["6001"] = {"class_type": "Sigmas Easing", "inputs": {
        "sigmas": ["4966", 0], "easing_type": "cubic", "easing_mode": "in_out",
        "normalize_input": True, "normalize_output": True, "strength": 0.7}}
    g["4802"]["inputs"]["sigmas"] = ["6001", 0]


def _apply_detailer(g):
    """Seconda passata leggera alla STESSA risoluzione (2 step da sigma 0.48): aggiunge
    micro-dettaglio (pelle/capelli/stoffe) senza cambiare la composizione."""
    if "4824" not in g or "4983" not in g or "5202" in g:   # 5202 = 2K: gia' rifinito
        return
    # sorgenti: nel likeness_fine il latente/conditioning giusti sono quelli POST-crop
    lat = ["5003", 2] if "5003" in g else ["4824", 0]
    pos = ["5003", 0] if "5003" in g else (["5102", 0] if "5102" in g else ["1241", 0])
    neg = ["5003", 1] if "5003" in g else (["5102", 1] if "5102" in g else ["1241", 1])
    g["6010"] = {"class_type": "LTXVImgToVideoConditionOnly", "inputs": {
        "vae": ["3940", 2], "image": ["3336", 0], "latent": lat,
        "strength": 1.0, "bypass": False}}
    g["6011"] = {"class_type": "LTXVConcatAVLatent", "inputs": {
        "video_latent": ["6010", 0], "audio_latent": ["4824", 1]}}
    g["6012"] = {"class_type": "CFGGuider", "inputs": {
        "model": ["3940", 0], "positive": pos, "negative": neg, "cfg": 1.0}}
    g["6013"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    g["6014"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": "0.4824, 0.2412, 0.0"}}
    g["6015"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": "{SEED}"}}
    g["6016"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["6015", 0], "guider": ["6012", 0], "sampler": ["6013", 0],
        "sigmas": ["6014", 0], "latent_image": ["6011", 0]}}
    g["6017"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["6016", 0]}}
    g["4983"]["inputs"]["latents"] = ["6017", 0]
    if "4818" in g:
        g["4818"]["inputs"]["samples"] = ["6017", 1]


def _apply_motion_speed(g, tab):
    """Anti-rallentatore: il conditioning del frame-rate scende a fps/velocita', cosi' il
    modello 'comprime' piu' movimento in ogni frame; la riproduzione resta agli fps scelti
    (CreateVideo e audio continuano a leggere il valore reale da 4978)."""
    speed = float(tab.get("video_speed", 1) or 1)
    if abs(speed - 1.0) < 0.01 or "1241" not in g:
        return
    fps = int(tab.get("video_fps") or 24)
    g["1241"]["inputs"]["frame_rate"] = round(fps / speed, 2)


def _deblotch_frame(img, np, ImageFilter, Image):
    """Rimuove i 'blob' del VAE LTX (gocce/polvere) SOLO sulle distese piatte e fredde
    (cielo, muri). Tre protezioni per il dettaglio, calibrate su misure reali:
    - maschera 'piatto': varianza fine < 2.4 (cielo=0.5-1.2, pelle/mare/stoffe >=4)
    - porta cromatica: la pelle e' calda (R-B >= 50) ed e' esclusa (gate 36-48)
    - erosione MinFilter 9: solo distese ampie.
    La banda dei blob (gauss 2-30) si azzera con 4 sottrazioni ITERATE (una sola ne
    lascerebbe ~50%). Verificato: cielo pulito, dettaglio 98.7%+."""
    def gauss(arr, s):
        return np.asarray(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(s)), dtype=np.float32)
    f = np.asarray(img, dtype=np.float32)
    fine = f.mean(axis=2) - gauss(f, 2).mean(axis=2)
    pad = np.pad(fine * fine, ((8, 7), (8, 7)), mode="edge")
    ii = pad.cumsum(0).cumsum(1)
    var = np.maximum((ii[15:, 15:] - ii[:-15, 15:] - ii[15:, :-15] + ii[:-15, :-15]) / 225.0, 0)
    flat = np.clip((2.4 - np.sqrt(var)) / 1.4, 0, 1)
    warm = gauss(f[..., 0] - f[..., 2] + 128.0, 4) - 128.0
    gate = np.clip((48.0 - warm) / 12.0, 0, 1)
    m = Image.fromarray((flat * gate * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(9))
    m = m.filter(ImageFilter.GaussianBlur(6))
    mask = np.asarray(m, dtype=np.float32)[..., None] / 255.0
    out = f.copy()
    for _ in range(4):
        band = gauss(out, 2) - gauss(out, 30)
        out = out - band * mask
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def deblotch_video(path):
    """Riscrive il video ripulito (stesso file), audio copiato senza ricodifica."""
    import av
    import numpy as np
    from PIL import Image, ImageFilter
    tmp = path + ".clean.mp4"
    inp = av.open(path)
    out = av.open(tmp, "w")
    vs_in = inp.streams.video[0]
    aud_in = inp.streams.audio[0] if inp.streams.audio else None
    vs = out.add_stream("h264", rate=int(vs_in.average_rate))
    vs.width, vs.height, vs.pix_fmt = vs_in.width, vs_in.height, "yuv420p"
    vs.options = {"crf": "13", "preset": "slow"}   # quasi-lossless: niente softening da encode
    as_ = out.add_stream_from_template(aud_in) if aud_in else None
    for pkt in inp.demux():
        if pkt.dts is None:
            continue
        if aud_in and pkt.stream == aud_in:
            pkt.stream = as_
            out.mux(pkt)
        elif pkt.stream == vs_in:
            for frame in pkt.decode():
                clean = _deblotch_frame(frame.to_image(), np, ImageFilter, Image)
                nf = av.VideoFrame.from_image(clean)
                for p in vs.encode(nf):
                    out.mux(p)
    for frame in vs_in.codec_context.decode(None):   # flush: gli ultimi frame bufferizzati
        clean = _deblotch_frame(frame.to_image(), np, ImageFilter, Image)
        for p in vs.encode(av.VideoFrame.from_image(clean)):
            out.mux(p)
    for p in vs.encode():
        out.mux(p)
    out.close()
    inp.close()
    os.replace(tmp, path)


def apply_video_quality(g, tab):
    g = json.loads(json.dumps(g))    # copia: il file su disco resta intatto
    _apply_video_crf(g, tab.get("video_crf", 12))
    _apply_motion_speed(g, tab)
    if tab.get("video_easing"):
        _apply_sigma_easing(g)
    if tab.get("video_detail"):
        _apply_detailer(g)
    return g


def comfy_generate(job, tab, tab_idx, idx):
    img = tab["images"][idx]
    prompt_text = img["prompt"]
    # garanzia finale: gli elementi fissi/forzati dello scenario finiscono comunque in ogni
    # prompt, anche se il modello prompt li ha dimenticati. Il controllo e' parola per parola:
    # basta che manchi "white" da "white floral dress" e il frammento viene appeso
    # (meglio una ripetizione che un vestito del colore sbagliato).
    for fr in tab.get("fixed_frags") or []:
        if frag_missing(prompt_text, fr):
            prompt_text = prompt_text.rstrip(" .") + ", " + fr["text"]
    # garanzia: i vincoli di stile scelti finiscono comunque nel prompt (se l'LLM li ha omessi)
    extra_neg = []
    for k in tab.get("styles", []):
        o = STYLE_OPTIONS.get(k)
        if not o:
            continue
        if o["check"] not in prompt_text.lower():
            if o.get("prepend"):   # i vincoli forti pesano di piu' a INIZIO prompt
                prompt_text = o["prompt"] + ". " + prompt_text
            else:
                prompt_text = prompt_text.rstrip(" .") + ", " + o["prompt"]
        if o.get("neg"):
            extra_neg.append(o["neg"])
    # anatomia corretta: sempre iniettata (positivo per tutti; negativo per i modelli che lo usano)
    if tab.get("anatomy", True):
        if "anatom" not in prompt_text.lower():
            prompt_text = prompt_text.rstrip(" .") + ", " + ANATOMY_POSITIVE
        extra_neg.append(ANATOMY_NEGATIVE)
    if tab.get("persona"):
        # personaggio configurato: coerenza anatomica RIGIDA contro gli arti doppi
        if "exactly two arms" not in prompt_text.lower():
            prompt_text = prompt_text.rstrip(" .") + \
                ", single subject, exactly two arms and two hands, exactly two legs"
        extra_neg.append("extra arms, third arm, duplicated arms, extra hands, "
                         "extra legs, conjoined bodies")
    extra_neg = ", ".join(extra_neg)
    seed = tab.get("seed") if tab.get("seed") is not None else random.randint(0, 2**48)
    img["seed"] = seed
    im_model = tab.get("image_model", "klein")
    img["final_prompt"] = prompt_text
    img["negative"] = effective_negative(im_model, extra_neg)
    def make_graph(w, h, steps, prefix):
        if tab.get("auto_frames") and idx < 2 and im_model.startswith("wf:"):
            # regista: i primi due render sono FOTO Klein (la seconda in edit dalla prima)
            ref = f"sb_{job['id']}_a.png" if idx == 1 else None
            return build_graph_klein(prompt_text, seed, w, h,
                                     tab.get("photo_steps") or 6,
                                     tab.get("loras") or [], prefix, ref_image=ref)
        if im_model.startswith("wf:"):
            wf = load_custom_workflow(im_model[3:] + ".json")
            # frame count LTX: multiplo di 8 + 1 (es. 97, 121) dalla durata scelta
            fps = int(tab.get("video_fps") or 24)
            secs = float(tab.get("video_secs") or 5)
            frames = max(9, int(round(secs * fps / 8)) * 8 + 1)
            vals = {"{PROMPT}": prompt_text, "{NEGATIVE}": extra_neg, "{SEED}": seed,
                    "{WIDTH}": w, "{HEIGHT}": h, "{STEPS}": steps, "{PREFIX}": prefix,
                    "{FRAMES}": frames, "{FPS}": fps}
            if tab.get("ref_image"):
                vals["{REF_IMAGE}"] = tab["ref_image"]
            for ph, fn in (tab.get("ref_images") or {}).items():
                vals[ph] = fn
            g = substitute_placeholders(apply_video_quality(wf["graph"], tab), vals)
            # workflow foto con meta.lora_chain: le LoRA della scheda entrano nel grafo
            return inject_wf_loras(g, tab.get("loras") or [],
                                   wf.get("meta", {}).get("lora_chain"))
        return build_graph(im_model, prompt_text, seed, w, h, steps,
                           tab["loras"], prefix, extra_neg=extra_neg)

    # modalita' REGISTA: 3 render in galleria (0=foto inizio, 1=foto fine, 2=video).
    # Seed fisso su inizio e video; la foto di FINE usa seed+1: con lo stesso seed Klein
    # riprodurrebbe la stessa identica immagine (la coerenza di scena/persona la
    # garantisce il riferimento ReferenceLatent, non il seed).
    if tab.get("auto_frames"):
        if "_sb_seed" not in tab:
            tab["_sb_seed"] = seed
        seed = tab["_sb_seed"] + (1 if idx == 1 else 0)
        img["seed"] = seed

    img["status"] = "generating"
    img["t_start"] = time.time()   # inizio "vero" (include caricamenti): base per la % a tempo
    # workflow video? niente mini-anteprima: rifarebbe un intero render video a 144px
    is_video_wf = False
    if im_model.startswith("wf:"):
        try:
            is_video_wf = load_custom_workflow(im_model[3:] + ".json") \
                .get("meta", {}).get("target") == "video"
        except Exception:
            pass
    if is_video_wf:
        tab["_preview_done"] = True
    # mini anteprima: generata UNA SOLA VOLTA per scheda (stesso modello) e RIUSATA per tutte le foto
    # del batch — non ha senso rigenerarla a ogni immagine (spreco di una render per foto).
    if not tab.get("_preview_done"):
        try:
            tab["_preview"] = make_mini_preview(job, make_graph, tab.get("steps", 6))
        except Exception:
            tab["_preview"] = None
        tab["_preview_done"] = True
    img["preview"] = tab.get("_preview")
    if job["cancel"]:
        img["status"] = "cancelled"
        return

    graph = make_graph(tab["width"], tab["height"], tab["steps"], "PromptStudio/" + job["id"])
    # client_id "ps-monitor": ComfyUI invia gli eventi di progresso solo al client che accoda
    resp = http_json(COMFY_URL + "/prompt", {"prompt": graph, "client_id": "ps-monitor"}, timeout=30)
    pid = resp["prompt_id"]
    img["pid"] = pid
    t0 = time.time()
    deadline = time.time() + 1800
    while time.time() < deadline:
        if job["cancel"]:
            try:
                http_json(COMFY_URL + "/interrupt", {}, timeout=5, method="POST")
            except Exception:
                pass
            img["status"] = "cancelled"
            return
        try:
            hist = http_json(COMFY_URL + f"/history/{pid}", timeout=10)
        except Exception:
            hist = {}
        entry = hist.get(pid)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError("Errore ComfyUI: " + json.dumps(msgs)[:400])
            outputs = entry.get("outputs", {})
            if outputs:
                files = []
                for node_out in outputs.values():
                    # immagini (.png) e video: i nodi video scrivono l'output sotto chiavi
                    # diverse -> VHS_VideoCombine usa "gifs", SaveVideo nativo usa "videos".
                    media = (node_out.get("images", [])
                             + node_out.get("gifs", [])
                             + node_out.get("videos", []))
                    for im in media:
                        fn = im.get("filename")
                        if not fn:
                            continue
                        src = os.path.join(COMFY_OUTPUT_DIR, im.get("subfolder", ""), fn)
                        ext = os.path.splitext(fn)[1] or ".png"
                        extra = f"_{len(files) + 1}" if files else ""   # piu' file: niente sovrascritture
                        dst = os.path.join(tab["dest"],
                                           f"{job['name']}_t{tab_idx + 1}_{idx + 1:03d}{extra}{ext}")
                        shutil.copy2(src, dst)
                        files.append(dst)
                        try:  # niente doppioni: via la copia interna di ComfyUI
                            os.remove(src)
                        except OSError:
                            pass
                # regista: le foto appena fatte diventano i riferimenti del video (idx 2)
                if tab.get("auto_frames") and idx < 2 and files:
                    tag = "a" if idx == 0 else "b"
                    shutil.copy2(files[0], os.path.join(COMFY_ROOT, "input",
                                                        f"sb_{job['id']}_{tag}.png"))
                    if idx == 1:
                        tab["ref_images"] = {"{REF_IMAGE}": f"sb_{job['id']}_a.png",
                                             "{REF_IMAGE2}": f"sb_{job['id']}_b.png"}
                # pulizia dei blob del VAE sulle aree piatte (cielo): opzione video
                if tab.get("video_clean"):
                    for fp in list(files):
                        if fp.lower().endswith(".mp4"):
                            try:
                                clean = fp[:-4] + "_pulito.mp4"
                                shutil.copy2(fp, clean)
                                deblotch_video(clean)     # l'ORIGINALE resta intatto
                                files.append(clean)
                            except Exception as e:
                                job["log"].append(f"pulizia video saltata: {e}")
                img["files"] = files
                img["preview"] = None            # non serve piu' lo sfondo materializzante
                img["dur"] = time.time() - t0   # durata del solo render (usata dalla stima job esistente)
                img["wall"] = max(0.1, time.time() - img.get("t_start", t0))  # tempo TOTALE (load+anteprima+render)
                # calibrazione: memorizza il tempo per (modello,risoluzione,step) e,
                # a parte, l'overhead di caricamento del modello quando c'e' stato
                key = item_timing_key(tab, idx)
                record_timing(key, img["dur"])
                if img["wall"] - img["dur"] > 8:
                    record_timing("load|" + key.split("|")[1], img["wall"] - img["dur"])
                img["status"] = "done"
                write_prompts_txt(job, tab_idx, tab)
                return
        time.sleep(1.0)
    raise RuntimeError("Timeout generazione immagine")


# ------------------------------------------------------------------ persistenza job
# I job vivono in memoria: senza questo sparivano a ogni riavvio dell'app. Li salviamo in
# LLM\jobs_state.json (fuori da app.zip, cosi' sopravvivono agli aggiornamenti dell'interfaccia).
JOBS_STATE = os.path.join(LLM_DIR, "jobs_state.json")


def save_jobs():
    try:
        snap = {jid: jobs[jid] for jid in jobs_order if jid in jobs}
        tmp = JOBS_STATE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"order": list(jobs_order), "jobs": snap}, f, default=str)
        os.replace(tmp, JOBS_STATE)
    except Exception:
        pass


def load_jobs():
    if not os.path.exists(JOBS_STATE):
        return
    try:
        with open(JOBS_STATE, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    for jid in data.get("order", []):
        job = (data.get("jobs") or {}).get(jid)
        if not job or "tabs" not in job:
            continue
        job["cancel"] = False
        if job.get("phase") in ("queued", "llm_loading", "prompting", "generating"):
            # interrotto da un riavvio: conserva le foto gia' fatte, il resto non riparte da solo
            done = job_done_count(job)
            for t in job["tabs"]:
                for im in t["images"]:
                    if im.get("status") == "generating":
                        im["status"] = "pending"
            job["phase"] = "done" if done else "error"
            if not done:
                job["error"] = "Job interrotto da un riavvio dell'app"
        # prompts_ready / scheduled / done / error / cancelled: restano come sono
        jobs[jid] = job
        if jid not in jobs_order:
            jobs_order.append(jid)


def jobs_saver_loop():
    """Salva i job periodicamente (cattura ogni cambiamento senza sporcare il codice ovunque)."""
    last = None
    while True:
        time.sleep(4)
        try:
            cur = json.dumps({"order": jobs_order,
                              "jobs": {j: jobs[j] for j in jobs_order if j in jobs}}, default=str)
        except Exception:
            cur = None
        if cur is not None and cur != last:
            save_jobs()
            last = cur


# ------------------------------------------------------------------ worker
def remove_job(jid):
    jobs.pop(jid, None)
    if jid in jobs_order:
        jobs_order.remove(jid)
    with queue_lock:
        for it in list(work_items):
            if it[0] == jid:
                work_items.remove(it)
    save_jobs()


def job_done_count(job):
    return sum(1 for t in job["tabs"] for im in t["images"] if im["status"] == "done")


def auto_remove_if_empty(job):
    """I job annullati che non hanno prodotto foto spariscono."""
    if job["phase"] == "cancelled" and job_done_count(job) == 0:
        remove_job(job["id"])


def _plain_video(tab):
    """VIDEO 'Foto + testo' o 'Due foto': workflow video, NON regista. Qui la foto di
    partenza porta GIA' l'aspetto (persona, vestito, scena): il prompt deve contenere SOLO
    il MOVIMENTO. Niente LLM descrittivo, niente persona, niente elementi-foto, niente
    anatomia — altrimenti il modello i2v riceve una descrizione statica al posto del moto."""
    im = tab.get("image_model", "")
    if tab.get("auto_frames") or not im.startswith("wf:"):
        return False
    try:
        return load_custom_workflow(im[3:] + ".json") \
            .get("meta", {}).get("target") == "video"
    except Exception:
        return False


def _wf_target_video(tab):
    """True se la scheda produce un VIDEO (workflow con meta.target == 'video')."""
    im = tab.get("image_model", "")
    if not im.startswith("wf:"):
        return False
    try:
        return load_custom_workflow(im[3:] + ".json").get("meta", {}).get("target") == "video"
    except Exception:
        return False


def item_noun(tab, idx):
    """Sostantivo giusto per l'elemento idx della scheda nei messaggi all'utente: 'video' o
    'foto'. Nel Regista i primi due render sono FOTO (inizio/fine), il terzo e' il VIDEO."""
    if not _wf_target_video(tab):
        return "foto"
    if tab.get("auto_frames"):
        return "foto" if idx < 2 else "video"
    return "video"


def job_media_noun(job):
    """Sostantivo per il totale del job: 'video' se produce solo video, 'foto' se solo foto,
    'render' se misto (es. il Regista fa 2 foto + 1 video)."""
    kinds = {item_noun(t, i) for t in job["tabs"] for i in range(len(t["images"]))}
    if kinds == {"video"}:
        return "video"
    if kinds == {"foto"}:
        return "foto"
    return "render"


def translate_motion(job, tab, text, must=None):
    """Trasforma il testo del movimento (anche in italiano) in UNA frase inglese che descrive
    SOLO cosa fanno soggetto e camera nella clip. Il primo frame mostra gia' persona e scena:
    niente aspetto, vestiti, viso, ambiente. Fedele, senza abbellimenti. `must` = termini
    [forzati] dell'utente che vanno inclusi COMUNQUE (anche se sono aspetto/oggetti).
    Fallback = testo."""
    sys_p = (
        "You convert a short description of a VIDEO clip's motion into ONE concise English "
        "sentence. Output ONLY the motion: what the subject and the camera DO during the clip. "
        "The first frame ALREADY shows the person and the scene, so do NOT describe appearance, "
        "face, body, hair, clothes, or setting — describe ONLY the movement/action. Stay literal "
        "and faithful, invent nothing, never soften explicit actions. If the input is already "
        "English just clean it up. Output only the sentence, no quotes."
    )
    if must:
        sys_p += (" EXCEPTION — the user marked these elements as MANDATORY: include EACH of "
                  "them explicitly in English in your sentence, even if it is appearance, "
                  "an object or clothing: " + "; ".join(must) + ".")
    try:
        resp = http_json(LLAMA_URL + "/v1/chat/completions",
                         {"messages": [{"role": "system", "content": sys_p},
                                       {"role": "user", "content": text}],
                          "temperature": 0.2, "top_p": 0.9, "max_tokens": 200}, timeout=300)
        out = clean_prompt(resp["choices"][0]["message"]["content"])
        if out and len(out) > 3:
            return out
    except Exception:
        pass
    return text


def _storyboard_llm(sys_p, user_p, keys, temperature=0.4):
    """Una chiamata LLM che deve restituire un JSON con le chiavi date (3 tentativi)."""
    for attempt in range(3):
        payload = {"messages": [{"role": "system", "content": sys_p},
                                {"role": "user", "content": user_p}],
                   "temperature": temperature, "max_tokens": 700}
        try:
            resp = http_json(LLAMA_URL + "/v1/chat/completions", payload, timeout=600)
            text = resp["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", text, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
            if all(isinstance(data.get(k), str) and data[k].strip() for k in keys):
                return {k: data[k].strip() for k in keys}
        except Exception:
            pass
    return None


def generate_storyboard_prompts(job, tab):
    """Modalita' REGISTA in due passaggi. L'utente puo' dare fino a 3 tracce:
    foto iniziale (ambientazione), foto finale (regia_end) e bozza di regia (regia_plot).
    Passo 1 — agente FOTOGRAFO: rivede e ARMONIZZA i prompt di inizio e fine (stessa scena
    con le stesse parole, posizione ESPLICITA di ogni oggetto in entrambi i frame, delta
    raggiungibile con un solo movimento nella durata). Passo 2 — agente REGISTA: legge le
    due foto definitive e, seguendo la bozza dell'utente, scrive l'UNICO movimento continuo
    che le collega (niente andirivieni, traiettoria esplicita degli oggetti che si spostano)."""
    secs = float(tab.get("video_secs") or 5)
    scene = (tab.get("amb_clean") or tab["ambientazione"]).strip()
    end_draft = (tab.get("regia_end") or "").strip()
    plot_draft = (tab.get("regia_plot") or "").strip()

    # --- passo 1: revisione/armonizzazione dei due prompt foto ---
    sys1 = (
        "You are a photographer planning the FIRST and LAST frame of one continuous "
        f"{secs:.0f}-second shot. Output STRICT JSON with exactly these keys:\n"
        '{"start": "...", "end": "..."}\n'
        "Rules:\n"
        "- Both values are English photo prompts (subject, outfit, pose, background, "
        "lighting). SAME person, SAME outfit, SAME background and lighting, described "
        "with the SAME words in both prompts.\n"
        "- State EXPLICITLY where every object/prop is in EACH frame (in her hands, "
        "floating in the water on her left, on the table...). Objects never appear or "
        "disappear; one may change place only if the last frame says exactly where it "
        "ended up.\n"
        f"- The end pose must be reachable from the start with ONE single continuous "
        f"movement in {secs:.0f} seconds (a person walks ~{secs * 1.3:.0f} meters, a head "
        "turn takes ~1s, an expression change ~1s). The difference must be CLEARLY and "
        "VISIBLY different but manageable in that time.\n"
        "- The SHOT SCALE and camera position stay the SAME: if the start is a full-figure "
        "shot, the end is a full-figure shot at similar distance — never jump from wide "
        "to close-up. Do NOT change location.\n"
        "- No new characters or objects. Keep each prompt under 60 words. "
        "Output ONLY the JSON."
    )
    user1 = "First frame (user draft): " + scene
    if end_draft:
        user1 += "\nLast frame (user draft — refine it but KEEP its intent): " + end_draft
    if plot_draft:
        user1 += "\nPlanned action between the frames (context): " + plot_draft
    if tab.get("soggetto"):
        user1 += "\nSubject/trigger (use verbatim as the subject's name): " + tab["soggetto"]
    pdesc_v = (tab.get("_persona_frag") or {}).get("text") or compose_persona(tab.get("persona"))[0]
    if pdesc_v:
        user1 += ("\nThe subject MUST match EXACTLY this description in BOTH frames: " + pdesc_v)
    frames = _storyboard_llm(sys1, user1, ("start", "end"))
    if not frames:
        raise RuntimeError("Il fotografo non ha prodotto foto inizio/fine valide (3 tentativi)")
    job["log"].append(f"[{tab['title']}] regista: prompt foto inizio/fine rivisti e "
                      "armonizzati" + (" (traccia di fine dell'utente)." if end_draft else "."))

    # --- passo 2: il regista scrive il movimento che collega le due foto ---
    sys2 = (
        "You are a film director. You are given the FIRST and LAST frame of one continuous "
        f"{secs:.0f}-second shot. Output STRICT JSON with exactly this key:\n"
        '{"motion": "..."}\n'
        "Rules:\n"
        "- 'motion' is ONE English sentence, present tense, describing ONE single "
        "continuous movement that takes the scene EXACTLY from the first frame to the "
        "last: no back-and-forth, never returning to a previous pose, no pauses, no cuts.\n"
        "- If an object changes place between the two frames, describe its exact path "
        "(e.g. 'she lowers the duck into the water on her left').\n"
        "- Stay consistent with BOTH frames: do not invent actions that contradict where "
        "people and objects are in the last frame.\n"
        "- Camera static or with a subtle drift. Movements happen at NATURAL real-life "
        "speed: never write 'slowly', 'gently' or 'slow motion' unless the user's notes "
        "explicitly ask for it.\n"
        "- Keep it under 45 words. Output ONLY the JSON."
    )
    user2 = "First frame: " + frames["start"] + "\nLast frame: " + frames["end"]
    if plot_draft:
        user2 += "\nDirector's notes from the user (follow them as the guiding track): " + plot_draft
    motion = _storyboard_llm(sys2, user2, ("motion",), temperature=0.3)
    if not motion:
        raise RuntimeError("Il regista non ha prodotto un movimento valido (3 tentativi)")
    tab["auto_prompts"] = {"start": frames["start"], "end": frames["end"],
                           "motion": motion["motion"]}
    job["log"].append(f"[{tab['title']}] regista: movimento scritto"
                      + (" sulla tua bozza di regia" if plot_draft else "")
                      + f", calibrato su {secs:.0f}s.")


def run_prompts_stage(job):
    llm_lock.acquire()
    try:
        # ogni scheda passa dall'LLM: le FOTO per generare i prompt, i VIDEO per TRADURRE
        # sempre il prompt in inglese (il modello LTX lavora in inglese). La traduzione/
        # riscrittura avviene QUI, in fase 'prompting', prima di 'Avvia subito'.
        job["phase"] = "llm_loading"
        start_llama(job)
        if job["cancel"]:
            raise RuntimeError("Annullato")
        job["phase"] = "prompting"
        for tab in job["tabs"]:
            if job["cancel"]:
                raise RuntimeError("Annullato")
            if tab.get("auto_frames"):
                # modalita' REGISTA: 3 render visibili nel job (foto inizio, foto fine, video)
                generate_storyboard_prompts(job, tab)
                ap = tab["auto_prompts"]
                tab["prompts"] = [ap["start"], ap["end"], ap["motion"]]
                for i, p in enumerate(tab["prompts"][:len(tab["images"])]):
                    tab["images"][i]["prompt"] = p
                continue
            if _plain_video(tab):
                # VIDEO Foto+testo / Due foto: la FOTO porta l'aspetto, il prompt porta il
                # MOVIMENTO — SEMPRE tradotto/riscritto in inglese (il modello LTX lavora in
                # inglese). I [forzati] dell'utente valgono anche qui (tradotti e garantiti).
                # NIENTE persona/scena/anatomia automatiche: le porta gia' la foto di partenza.
                if tab.get("forced_terms"):
                    translate_forced_terms(job, tab)   # [] -> frammenti inglesi + keyword
                forced = [f for f in (tab.get("fixed_frags") or []) if not f.get("directive")]
                must = [f["text"] for f in forced]
                amb = (tab.get("amb_clean") or tab["ambientazione"]).strip()
                motion = translate_motion(job, tab, amb, must)   # SEMPRE in inglese
                # garanzia []: ogni forzato deve comparire nel prompt tradotto del video
                for fr in forced:
                    if frag_missing(motion, fr):
                        motion = motion.rstrip(" .") + ", " + fr["text"]
                tail = (" · [forzati]: " + "; ".join(must)) if must else ""
                job["log"].append(f"[{tab['title']}] VIDEO: prompt tradotto in inglese -> "
                                  f"\"{motion}\" (movimento; la foto di partenza porta l'aspetto)"
                                  + tail)
                tab["prompts"] = [motion] * tab["num_images"]
                for i in range(tab["num_images"]):
                    tab["images"][i]["prompt"] = motion
                # SOLO i [forzati] dell'utente restano come vincoli (comfy_generate li ri-garantisce);
                # persona/scena/anatomia automatiche NO: la FOTO le porta gia'.
                tab["persona"] = {}
                tab["styles"] = []
                tab["anatomy"] = False
                continue
            job["log"].append(f"Creo i prompt per '{tab['title']}'...")
            # PRIMA di generare: traduci in inglese i termini [forzati] cosi' i prompt nascono
            # gia' in inglese e non ci finisce l'italiano copiato dalle parentesi quadre.
            if tab.get("forced_terms"):
                translate_forced_terms(job, tab)
            # configuratore PERSONA: la descrizione composta e' un elemento FISSO a tutti
            # gli effetti — l'LLM la riceve come obbligatoria (build_user_prompt), il repair
            # la verifica per parole-chiave e al render si appende cio' che manca.
            pskip = persona_forced_conflicts(tab.get("fixed_frags"), tab.get("forced_terms"))
            pdesc, pkws = compose_persona(tab.get("persona"), pskip)
            if pdesc:
                tab["_persona_frag"] = {"text": pdesc, "keywords": pkws}
                tab["fixed_frags"] = (tab.get("fixed_frags") or []) + [tab["_persona_frag"]]
                job["log"].append(f"[{tab['title']}] persona configurata (garantita in ogni "
                                  f"foto): {pdesc}")
                if pskip and tab.get("persona") and any(k in tab["persona"] for k in pskip):
                    job["log"].append(f"[{tab['title']}] persona: "
                                      + " e ".join(sorted(pskip))
                                      + " li dettano i [forzati] dell'utente (vincono loro).")
            _t0p = time.time()
            # regista foto: il batch e' una storia cronologica, non n variazioni
            if tab.get("photo_story"):
                prompts = generate_story_prompts(job, tab)
            else:
                prompts = generate_prompts_for_tab(job, tab)
            if prompts:
                record_timing("prompts|per_prompt", (time.time() - _t0p) / len(prompts))
            tab["prompts"] = prompts
            for i, p in enumerate(prompts):
                tab["images"][i]["prompt"] = p
            if tab.get("photo_story"):
                # storia: NIENTE extract_fixed_frags sul brief — su un racconto estrae
                # veleno (l'intervallo orario "12 to 23" come elemento fisso, aggettivi
                # vuoti) e le riscritture del repair rimescolano gli orari della scaletta.
                # La coerenza la garantiscono SOLO: [forzati] tradotti + PROTAGONISTA della
                # scaletta (descritta dal regista se l'utente non l'ha fatto).
                forced = list(tab.get("fixed_frags") or [])
                prot = tab.pop("_story_prot_frag", None)
                merged, have = [], set()
                for f in forced + ([prot] if prot else []):
                    kws = f.get("keywords", [])
                    if f.get("directive") or not any(k in have for k in kws):
                        merged.append(f)
                        have.update(kws)
                tab["fixed_frags"] = merged[:8]
                if prot:
                    job["log"].append(f"[{tab['title']}] protagonista FORZATA in ogni "
                                      f"capitolo [parole: {', '.join(prot['keywords'])}]")
            elif not tab.get("forced_terms"):
                extract_fixed_frags(job, tab)   # sovrascrive fixed_frags: rimetto la persona
                if tab.get("_persona_frag"):
                    tab["fixed_frags"] = [tab["_persona_frag"]] + (tab.get("fixed_frags") or [])
            # ripara i prompt in cui mancano elementi forzati/fissi o chip di stile
            repair_prompts(job, tab)
        stop_llama(job)   # prompt/traduzioni pronti: libera la VRAM per ComfyUI
        if job["cancel"]:
            raise RuntimeError("Annullato")
        if job.get("auto_launch"):
            # niente fase "prompts_ready" visibile: si passa dritti alla creazione
            launch_job(job, None)
        else:
            job["phase"] = "prompts_ready"
            job["log"].append("Tutti i prompt sono pronti. Rivedili e avvia (o programma) la creazione.")
    except Exception as e:
        job["phase"] = "cancelled" if job["cancel"] else "error"
        job["error"] = str(e)
        job["log"].append(("Annullato: " if job["cancel"] else "ERRORE: ") + str(e))
        job["ended"] = time.time()
        auto_remove_if_empty(job)
    finally:
        stop_llama()
        llm_lock.release()


def eval_image_url(path, maxpx=1400):
    """data URL JPEG ridimensionato per il giudice Qwen3-VL. ~1400px da dettaglio reale sufficiente a
    riempire i 1024 image-token forzati all'avvio (vedi --image-min-tokens): con la vecchia thumbnail
    1024px il modello aveva pochi token e dava 10/10 a tutto. Oltre non serve (costo/lentezza)."""
    from PIL import Image
    img = Image.open(path)
    img.thumbnail((maxpx, maxpx))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


EVAL_GENERAL = (
    "You are a STRICT forensic judge of AI-generated photos. These images often HIDE anatomical defects; "
    "your job is to find them. Examine EACH person ONE BY ONE. For every person check:\n"
    " (a) FACE — sharp and symmetric, or smeared/melted/waxy/blurred/half-formed with distorted or "
    "misplaced eyes, nose or mouth?\n"
    " (b) CHEST / TORSO — natural, or distorted / merged / misshapen breasts or abdomen?\n"
    " (c) HANDS / FINGERS — correct, or fused / blurred into a blob, or wrong finger count?\n"
    " (d) LIMBS — right count, and each arm and leg attributable to a real body (no orphan or extra "
    "limb, no two people merged into one).\n"
    "Do NOT be generous: a smeared or waxy face is a DEFECT, not 'fine'. Even ONE melted face or one "
    "fused chest/hands is a serious defect. NORMAL, never a defect: water refraction/ripples or crop on "
    "an OTHERWISE SHARP body; wet skin/hair, splashes, freckles, moles, pores, soft background bokeh.\n"
    "Scoring: 10 ONLY if EVERY person is flawless; 8-9 tiny softness; 6-7 one clearly imperfect "
    "face/part; 3-5 one clearly melted face or fused part; 1-2 multiple melted faces/bodies.\n"
    "ALSO check SCENE COHERENCE: is the environment physically possible and internally consistent? A "
    "SERIOUS defect (scene_ok=false) is an IMPOSSIBLE or CONTRADICTORY scene — e.g. bodies UNDERWATER "
    "(light caustics rippling on skin, the water surface seen from BELOW at the top) while the BACKGROUND "
    "is a dry ABOVE-water landscape with a normal horizon at eye level; a doubled or contradictory "
    "waterline; two incompatible settings fused into one. This is NOT mere refraction — it is a broken "
    "scene, and it MUST drag the score down hard regardless of how clean the bodies are.\n"
    "The score measures ANATOMY and SCENE. SEPARATELY you must also check FIDELITY: does the image "
    "actually show the MAIN subject and action of the description (e.g. nude vs clothed/swimsuit, the "
    "specific act requested, the setting)? Fidelity does NOT change the score, but you MUST report it "
    "truthfully in the note — never claim it matches without checking.\n")


def _vision_call(url, ask, max_tokens=300):
    """Una chiamata al GIUDICE Qwen3-VL (slot dedicato). Risposta diretta ~2s. Ritorna il testo."""
    payload = {"messages": [{"role": "user", "content": [
                    {"type": "text", "text": ask},
                    {"type": "image_url", "image_url": {"url": url}}]}],
               "temperature": 0.1, "top_p": 0.9, "max_tokens": max_tokens}
    resp = http_json(f"http://127.0.0.1:{SLOT_EVAL['port']}/v1/chat/completions", payload, timeout=180)
    msg = resp["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()


def _parse_score(txt):
    """(score 1-10 | None, nota, dict) dal testo della visione."""
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            o = json.loads(m.group(0))
            sc = max(1, min(10, int(round(float(o.get("score"))))))
            note = str(o.get("note") or o.get("reason") or "").strip()[:140]
            return sc, note, o
        except Exception:
            pass
    m2 = re.search(r"\b(10|[1-9])\b", txt)
    if m2:
        return int(m2.group(1)), txt.strip()[:140], {}
    return None, (txt.strip()[:140] or "risposta non interpretabile"), {}


def evaluate_image(path, prompt, deep=False, logs=None, criteria=None):
    """Valuta con la visione di Qwen fedelta' + qualita'. Ritorna (voto 1-10 | None, nota).
    deep=True: valutazione in CONTRADDITTORIO — dopo il primo voto Qwen prova a smentirsi
    (i difetti da acqua/tagli sono spesso falsi); se si contraddice, rivaluta. logs (lista):
    ci scrive ogni passaggio, che l'utente vede nel registro del job. criteria (str): criteri
    dell'utente che GUIDANO il voto (in aggiunta ai controlli tecnici di anatomia/scena)."""
    def L(s):
        if logs is not None:
            logs.append(s)
    try:
        ensure_eval_loaded([])
    except Exception as e:
        return None, "giudice non caricato: " + str(e)[:100]
    try:
        url = eval_image_url(path)
    except Exception:
        return None, "immagine illeggibile"
    crit = (criteria or "").strip()
    crit_block = ""
    if crit:
        crit_block = (
            "\n\nUSER'S OWN EVALUATION CRITERIA — these are MANDATORY and must STRONGLY drive the "
            "score together with anatomy: reward images that satisfy them, penalize images that do "
            "not. Judge them honestly against what you SEE. Criteria:\n" + crit[:600] + "\n")
    ask1 = (EVAL_GENERAL + "Description: " + (prompt or "(none)").strip()[:700] + crit_block + "\n\n"
            "First find the SINGLE worst person in the image and their worst flaw (melted/waxy face, "
            "fused chest, blurred/fused hands, extra or orphan limb) — if none is truly flawed, say so. "
            "Let that worst flaw " + ("AND the user's criteria above " if crit else "") + "drive the "
            "score. Then decide scene_ok: is the environment "
            "physically possible (NOT underwater-bodies fused with a dry above-water horizon, no doubled "
            "waterline)? Then decide match: does the image show the described main subject AND action? "
            'Return ONLY compact JSON: {"worst": "<worst person + their flaw, or \'none\'>", '
            '"score": <1-10>, "scene_ok": true|false, "match": true|false, '
            '"note": "<max 22 words Italian. If scene_ok is false START with \'SCENA IMPOSSIBILE: <cosa>\'; '
            'else if match is false START with \'FUORI TEMA: <cosa>\'; else the verdict'
            + (", mentioning how well the user criteria are met" if crit else "") + '>"}.')
    try:
        s1, note1, o1 = _parse_score(_vision_call(url, ask1))
    except Exception as e:
        return None, "errore valutazione: " + str(e)[:100]
    if s1 is None:
        return None, note1
    # Coerenza scena: una scena fisicamente impossibile (sopra/sotto acqua fusi, orizzonte doppio) e'
    # spazzatura di render e PREVALE su tutto — taglia il voto ed esce (il contraddittorio anatomico
    # non c'entra col difetto di scena). Robusto anche se il modello scorda il prefisso.
    if o1.get("scene_ok") is False:
        if "scena impossibile" not in note1.lower():
            note1 = ("SCENA IMPOSSIBILE: " + note1) if note1 else "SCENA IMPOSSIBILE: ambiente fisicamente incoerente"
        sc = min(s1, 3)
        L(f"scena incoerente -> voto tagliato a {sc}/10 — {note1}")
        return sc, note1
    # Fedelta' al prompt: NON tocca il voto (solo anatomia), ma se il contenuto non corrisponde
    # lo segnaliamo sempre nella nota, anche se il modello ha dimenticato il prefisso.
    if o1.get("match") is False and "fuori tema" not in note1.lower():
        note1 = ("FUORI TEMA: " + note1) if note1 else "FUORI TEMA: contenuto non conforme al prompt"
    L(f"voto iniziale {s1}/10 — {note1}")
    if not deep:
        return s1, note1

    if s1 <= 6:
        # ha penalizzato: prova a SMENTIRE il difetto (i falsi positivi su acqua/tagli sono comuni)
        ask2 = (f'Another judge gave this image {s1}/10 with this criticism: "{note1}". Re-examine ONLY '
                "that criticism, very carefully. Water refraction, submerged or cropped limbs, "
                "close-together people and normal poses are NOT defects. Is that criticism a REAL, "
                'clearly visible defect, or a mistake? Return ONLY JSON {"defect_real": true|false, '
                '"score": <1-10>, "note": "<max 14 words Italian>"}.')
        try:
            s2, note2, o2 = _parse_score(_vision_call(url, ask2, 220))
        except Exception:
            return s1, note1
        real = bool(o2.get("defect_real", True))
        L(f"contraddittorio: difetto {'CONFERMATO' if real else 'SMENTITO'} — {note2}")
        if not real:
            new = s2 if s2 is not None else min(10, s1 + 4)
            L(f"-> rivalutato a {new}/10 (il difetto iniziale non c'era)")
            return new, (note2 or "difetto iniziale smentito in verifica")
        return s1, note1
    else:
        # ha promosso: cerca un difetto che potrebbe aver MANCATO
        ask2 = (f"Another judge gave this image {s1}/10 and found no defect. Look again, harder. FIRST "
                "recount the people and trace every arm and leg to a specific body: is there a limb or a "
                "pair of legs (e.g. legs sticking out of the water) that belongs to nobody, or do two "
                "bodies merge into one indistinct underwater tangle? THEN check for fused/melted fingers "
                "or hands, a face with wrong eyes/teeth, an impossible floating object. Ignore mere water "
                "refraction, crop and pose on an otherwise coherent body. Did they miss a CLEAR defect? "
                'Return ONLY JSON {"defect_found": true|false, "score": <1-10>, "note": "<max 14 words Italian>"}.')
        try:
            s2, note2, o2 = _parse_score(_vision_call(url, ask2, 220))
        except Exception:
            return s1, note1
        found = bool(o2.get("defect_found", False))
        L(f"controllo inverso: {'difetto TROVATO' if found else 'confermato, nessun difetto'} — {note2}")
        if found and s2 is not None and s2 < s1:
            L(f"-> abbassato a {s2}/10 (difetto sfuggito al primo giudizio)")
            return s2, (note2 or "difetto trovato in verifica")
        return s1, note1


def eval_pending(job):
    """Valuta le foto 'done' non ancora valutate. Libera prima la VRAM di ComfyUI per far
    posto a Qwen; il render successivo la ricarica da solo."""
    todo = []
    for t_idx, t in enumerate(job["tabs"]):
        if not t.get("eval_enabled"):
            continue
        for im in t["images"]:
            if im.get("status") == "done" and im.get("score") is None and im.get("files"):
                todo.append((t_idx, t, im))
    if not todo:
        return
    # fase di ANALISI a sé: barra che riparte da 0, annullabile col tasto Annulla.
    prev_phase = job["phase"]
    job["phase"] = "evaluating"
    job["eval_total"] = len(todo)
    job["eval_done"] = 0
    job["log"].append(f"Fase di analisi: valuto la qualità di {len(todo)} foto con la visione di Qwen...")
    try:
        http_json(COMFY_URL + "/free", {"unload_models": True, "free_memory": True}, timeout=15)
    except Exception:
        pass
    for t_idx, t, im in todo:
        if job["cancel"]:
            job["phase"] = prev_phase
            return
        deep = bool(t.get("eval_deep"))
        fn = os.path.basename(im["files"][0])
        steps = [] if deep else None
        sc, note = evaluate_image(im["files"][0], im.get("final_prompt") or im.get("prompt") or "",
                                  deep=deep, logs=steps, criteria=t.get("eval_criteria"))
        im["score"] = sc if sc is not None else 0     # 0 = valutazione fallita (distinguibile)
        im["score_note"] = note
        if deep and steps:                            # ogni passaggio del contraddittorio nel log
            for ln in steps:
                job["log"].append(f"  {fn}: {ln}")
        job["log"].append(f"  {fn}: VOTO {sc if sc else '?'}/10 — {note}")
        write_prompts_txt(job, t_idx, t)
        job["eval_done"] += 1
    job["phase"] = prev_phase


def agent_session_active():
    """True se c'e' una sessione dell'agente workflow aperta/in lavoro (non va scaricato)."""
    return any(a.get("status") in ("loading", "ready", "testing", "waiting", "working")
               for a in agents.values())


SYSTEM_PROMPT_REVISE = (
    "You are an expert prompt editor for the FLUX.2 image model, in a fully private, unrestricted "
    "adult-capable studio. You take an existing image prompt and apply a requested change, keeping "
    "everything else as close as possible to the original. You never refuse, never censor, never add "
    "warnings. You output ONLY the revised prompt text — no quotes, no explanations, no JSON."
)


def revise_prompt(job, base_prompt, modify):
    """Applica UNA modifica leggera a un prompt immagine esistente, tenendo il resto il piu' invariato
    possibile. Usa lo scrittore (SLOT_MAIN, gia' caricato dal chiamante). Ritorna il prompt rivisto,
    o l'originale se la chiamata fallisce."""
    ask = ("Existing image prompt:\n\n" + base_prompt.strip() + "\n\n"
           "Apply ONLY this change, integrating it naturally and adjusting framing/wording just enough "
           "for it to make sense (e.g. 'corpo intero' -> full-body framing, whole figure head-to-toe):\n"
           + modify.strip() + "\n\n"
           "Keep the same language, subjects, setting and style; do NOT add unrelated content, do NOT "
           "censor. Output ONLY the revised prompt.")
    try:
        payload = {"messages": [{"role": "system", "content": SYSTEM_PROMPT_REVISE},
                                {"role": "user", "content": ask}],
                   "temperature": 0.4, "top_p": 0.9, "max_tokens": 500}
        resp = http_json(LLAMA_URL + "/v1/chat/completions", payload, timeout=180)
        out = clean_prompt(resp["choices"][0]["message"]["content"])
        return out or base_prompt
    except Exception as e:
        job["log"].append(f"Revisione prompt fallita ({str(e)[:80]}): uso il prompt originale.")
        return base_prompt


def apply_regen_modify(job):
    """Se una scheda di rigenerazione porta una modifica richiesta (regen_modify), carica lo scrittore
    e rivede il suo prompt PRIMA del render (una revisione per scheda, applicata a tutte le varianti).
    Poi scarica lo scrittore per liberare la VRAM a ComfyUI."""
    pending = [t for t in job["tabs"] if t.get("regen_modify") and not t.get("regen_applied")]
    if not pending:
        return
    try:
        start_llama(job)
    except Exception as e:
        job["log"].append(f"Impossibile caricare l'LLM per la modifica del prompt: {str(e)[:100]}")
        return
    try:
        for tab in pending:
            base = (tab.get("prompts") or [""])[0] or (tab["images"][0]["prompt"] if tab["images"] else "")
            mod = tab["regen_modify"]
            job["log"].append(f"[{tab['title']}] Rivedo il prompt con la modifica: «{mod}»")
            newp = revise_prompt(job, base, mod)
            tab["prompts"] = [newp] * len(tab["images"])
            for im in tab["images"]:
                im["prompt"] = newp
            tab["regen_applied"] = True
            job["log"].append(f"[{tab['title']}] Prompt rivisto -> {newp[:140]}")
    finally:
        stop_llama(job)


def run_images_stage(job):
    RENDER_BUSY["n"] += 1
    make_agent_render_safe()   # se l'agente sta usando la VRAM, riportalo in RAM
    eval_any = any(t.get("eval_enabled") for t in job["tabs"])
    try:
        job["phase"] = "generating"
        apply_regen_modify(job)   # rigenerazione con modifica: rivedi i prompt prima di renderizzare
        total = sum(len(t["images"]) for t in job["tabs"])
        done_count = 0
        since_eval = 0
        for t_idx, tab in enumerate(job["tabs"]):
            os.makedirs(tab["dest"], exist_ok=True)
            write_prompts_txt(job, t_idx, tab)
            every = int(tab.get("eval_every", 0) or 0)
            for i in range(len(tab["images"])):
                if job["cancel"]:
                    raise RuntimeError("Annullato")
                done_count += 1
                noun = item_noun(tab, i)
                job["log"].append(f"[{tab['title']}] {noun} {i + 1}/{len(tab['images'])} "
                                  f"(totale {done_count}/{total})...")
                try:
                    comfy_generate(job, tab, t_idx, i)
                except Exception as e:
                    tab["images"][i]["status"] = "error"
                    tab["images"][i]["error"] = str(e)
                    job["log"].append(f"Errore {noun}: {e}")
                # valutazione a lotti: ogni N foto (0 = solo alla fine)
                if tab.get("eval_enabled") and every > 0 and tab["images"][i]["status"] == "done":
                    since_eval += 1
                    if since_eval >= every:
                        eval_pending(job)
                        since_eval = 0
        if eval_any and not job["cancel"]:
            eval_pending(job)   # copre eval_every=0 e le foto rimaste da valutare
        if job["cancel"]:
            raise RuntimeError("Annullato")
        ok = sum(1 for t in job["tabs"] for im in t["images"] if im["status"] == "done")
        for t in job["tabs"]:            # l'anteprima non serve piu': non appesantire jobs_state.json
            t.pop("_preview", None)
        job["phase"] = "done"
        job["log"].append(f"Completato: {ok}/{total} {job_media_noun(job)}.")
    except Exception as e:
        job["phase"] = "cancelled" if job["cancel"] else "error"
        job["error"] = str(e)
        job["log"].append(("Annullato: " if job["cancel"] else "ERRORE: ") + str(e))
        auto_remove_if_empty(job)
    finally:
        RENDER_BUSY["n"] = max(0, RENDER_BUSY["n"] - 1)
        job["ended"] = time.time()
        # libera la RAM/VRAM: il giudice (Qwen3-VL, slot dedicato) si scarica a fine job. Non tocca
        # l'agente (slot separato), quindi niente controllo sulle sessioni agente.
        if eval_any:
            try:
                stop_llama_slot(SLOT_EVAL)
                job["log"].append("Giudice Qwen3-VL scaricato dalla RAM (valutazione conclusa).")
            except Exception:
                pass


def launch_job(job, launch_at):
    """launch_at: epoch seconds oppure None = subito"""
    job["launch_at"] = launch_at
    if launch_at:
        job["phase"] = "scheduled"
        job["log"].append("Creazione programmata per " +
                          time.strftime("%H:%M del %d/%m/%Y", time.localtime(launch_at)))
    else:
        job["phase"] = "queued"
        job["log"].append("Creazione avviata.")
    with queue_lock:
        work_items.append((job["id"], "images"))
    queue_event.set()


def worker_loop():
    while True:
        item = None
        wait_s = None
        with queue_lock:
            now = time.time()
            runnable = []
            for it in work_items:
                jid, stage = it
                job = jobs[jid]
                if job["cancel"]:
                    continue
                if stage == "prompts" or not job.get("launch_at") or job["launch_at"] <= now:
                    runnable.append(it)
                else:
                    d = job["launch_at"] - now
                    wait_s = d if wait_s is None else min(wait_s, d)
            # pulizia elementi annullati
            to_check = []
            for it in list(work_items):
                if jobs[it[0]]["cancel"]:
                    work_items.remove(it)
                    j = jobs[it[0]]
                    if j["phase"] in ("queued", "scheduled", "prompts_ready"):
                        j["phase"] = "cancelled"
                        j["ended"] = time.time()
                        to_check.append(j)
            if runnable:
                item = runnable[0]
                if item in work_items:
                    work_items.remove(item)
                else:
                    item = None
            elif not work_items:
                queue_event.clear()
        for j in to_check:
            auto_remove_if_empty(j)
        if item is None:
            queue_event.wait(timeout=min(wait_s, 30) if wait_s else None)
            continue
        jid, stage = item
        if stage == "prompts":
            run_prompts_stage(jobs[jid])
        else:
            run_images_stage(jobs[jid])


# ------------------------------------------------------------------ HTTP handler
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        path, _, query = self.path.partition("?")
        params = {}
        for kv in query.split("&"):
            if "=" in kv:
                k, _, v = kv.partition("=")
                params[k] = urllib.parse.unquote(v)
        try:
            if path == "/" or path == "/index.html":
                with open(os.path.join(STATIC_DIR, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif path == "/api/loras":
                loras = []
                if os.path.isdir(LORA_DIR):
                    for fn in sorted(os.listdir(LORA_DIR)):
                        if fn.lower().endswith((".safetensors", ".ckpt", ".pt")):
                            loras.append(fn)
                imodels = [{"key": k, "label": v["label"], "steps": v["steps"],
                            "installed": image_model_installed(k)}
                           for k, v in IMAGE_MODELS.items()
                           if image_model_installed(k) or k in MODEL_CATALOG]
                for w in list_custom_workflows():
                    imodels.append({"key": "wf:" + w["file"][:-5], "label": "WF: " + w["name"],
                                    "steps": w["steps"], "custom": True, "target": w["target"],
                                    "ref_slots": w["ref_slots"]})
                self._send(200, {"loras": loras, "default_dest": DEFAULT_DEST,
                                 "models": list_models(), "default_model": default_model(),
                                 "agent_model": agent_model(),
                                 "style_options": [{"key": k, "label": v["label"], "tip": v["tip"],
                                                    "group": v["group"]}
                                                   for k, v in STYLE_OPTIONS.items()],
                                 "image_models": imodels,
                                 "persona_options": [
                                     {"key": s["key"], "label": s["label"],
                                      "options": [{"key": o[0], "label": o[1], "frag": o[2]}
                                                  for o in s["options"]]}
                                     for s in PERSONA_OPTIONS],
                                 "persona_presets": [{"key": k, "label": v["label"], "sel": v["sel"]}
                                                     for k, v in PERSONA_PRESETS.items()],
                                 "characters": load_characters(),
                                 "app_version": APP_VERSION,
                                 "essentials_missing": essentials_missing()})
            elif path == "/api/update/check":
                self._send(200, check_update())
            elif path == "/api/update/status":
                self._send(200, dict(_UPDATE))
            elif path == "/api/status":
                comfy = True
                try:
                    http_json(COMFY_URL + "/system_stats", timeout=3)
                except Exception:
                    comfy = False
                self._send(200, {
                    "comfy": comfy,
                    "llm_model": default_model() is not None,
                    "llama_exe": os.path.exists(LLAMA_SERVER_EXE),
                })
            elif path == "/api/models_status":
                cat = []
                for k, v in COMPONENT_CATALOG.items():
                    r = CATALOG_ROLE.get(k, {})
                    cat.append({"key": k, "label": v["label"], "kind": v["kind"],
                                "size_gb": v["size_gb"], "desc": v["desc"],
                                "installed": component_installed(k),
                                "group": r.get("group", "optional"),
                                "order": r.get("order", 99), "starter": r.get("starter", False)})
                for k, v in MODEL_CATALOG.items():
                    r = CATALOG_ROLE.get(k, {})
                    cat.append({"key": k, "label": v["label"], "kind": v["kind"],
                                "size_gb": v["size_gb"], "desc": v["desc"],
                                "installed": model_installed(k),
                                "group": r.get("group", "optional"),
                                "order": r.get("order", 99), "starter": r.get("starter", False)})
                cat.sort(key=lambda m: m["order"])
                dls = {k: {kk: vv for kk, vv in st.items() if kk != "cancel"}
                       for k, st in DOWNLOADS.items()}
                self._send(200, {"catalog": cat, "downloads": dls,
                                 "essentials_missing": essentials_missing(),
                                 "starter_set": STARTER_SET})
            elif path == "/api/sysmon":
                data = sysmon()
                loaded = []
                if SLOT_AGENT["model"]:
                    loaded.append("Agente (" + SLOT_AGENT["model"].split("-")[0] + ")")
                if SLOT_MAIN["model"]:
                    loaded.append("Prompt (" + SLOT_MAIN["model"].split("-")[0] + ")")
                data["loaded"] = loaded
                data["tps"] = last_stats["tps"]
                data["streaming"] = any(x.get("stream_text") for x in agents.values())
                self._send(200, data)
            elif path == "/api/agent_llm_status":
                self._send(200, {"state": agent_warm["state"], "model": agent_warm["model"] or agent_model(),
                                 "resident": agent_resident(), "error": agent_warm["error"],
                                 "mode": SLOT_AGENT.get("moe_mode")})
            elif path == "/api/presets":
                self._send(200, {"presets": load_presets()})
            elif path.startswith("/api/agent/") and path.endswith("/preview"):
                aid = path.split("/")[3]
                a = agents.get(aid)
                if not a or not a.get("preview") or not os.path.exists(a["preview"]):
                    self._send(404, {"error": "anteprima non disponibile"})
                    return
                with open(a["preview"], "rb") as f:
                    self._send(200, f.read(), "image/png")
            elif path.startswith("/api/agent/"):
                aid = path.split("/")[3]
                a = agents.get(aid)
                if not a:
                    self._send(404, {"error": "sessione non trovata"})
                    return
                self._send(200, {"id": a["id"], "status": a["status"], "target": a["target"],
                                 "iteration": a["iteration"], "chat": a["chat"][-60:],
                                 "stream": (a.get("stream_text") or "")[-3000:] or None,
                                 "tps": a.get("tps"),
                                 "test_prog": (progress_info(a.get("test_pid"))
                                               if a["status"] == "testing" else None),
                                 "has_preview": bool(a.get("preview"))})
            elif path == "/api/jobs":
                self._send(200, [job_public(jobs[j]) for j in reversed(jobs_order)])
            elif path.startswith("/api/job/"):
                jid = path.split("/")[3]
                if jid not in jobs:
                    self._send(404, {"error": "job non trovato"})
                else:
                    self._send(200, job_public(jobs[jid], full=True))
            elif path == "/api/file":
                jid = params.get("job")
                t = int(params.get("t", 0))
                idx = int(params.get("i", 0))
                job = jobs.get(jid)
                if (not job or t >= len(job["tabs"]) or idx >= len(job["tabs"][t]["images"])
                        or not job["tabs"][t]["images"][idx]["files"]):
                    self._send(404, {"error": "file non trovato"})
                    return
                fp = job["tabs"][t]["images"][idx]["files"][0]
                ext = os.path.splitext(fp)[1].lower()
                ctype = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
                         ".png": "image/png", ".gif": "image/gif",
                         ".mp4": "video/mp4", ".webm": "video/webm",
                         ".mov": "video/quicktime", ".mkv": "video/x-matroska"}.get(
                             ext, "application/octet-stream")
                with open(fp, "rb") as f:
                    self._send(200, f.read(), ctype)
            else:
                self._send(404, {"error": "not found"})
        except BrokenPipeError:
            pass
        except Exception as e:
            self._send(500, {"error": str(e)})

    def do_POST(self):
        try:
            if self.path == "/api/generate":
                body = self._read_body()
                tabs_in = body.get("tabs") or []
                tabs = []
                for t in tabs_in:
                    amb = (t.get("ambientazione") or "").strip()
                    if not amb:
                        continue
                    amb_clean, forced_terms = parse_forced_terms(amb)
                    n = max(1, min(200, int(t.get("num_images", 4))))
                    im = t.get("image_model", "klein")
                    if t.get("auto_frames") and im.startswith("wf:"):
                        n = 3        # regista: foto inizio + foto fine + video, tutti in galleria
                    if im.startswith("wf:"):
                        if not os.path.exists(os.path.join(WORKFLOWS_DIR, im[3:] + ".json")):
                            im = "klein"
                        elif not t.get("auto_frames"):
                            # un job VIDEO fa UN video per scheda: il "numero foto" della
                            # modalita' FOTO non deve mai filtrare qui (visto sul campo: 5)
                            try:
                                if load_custom_workflow(im[3:] + ".json") \
                                        .get("meta", {}).get("target") == "video":
                                    n = 1
                            except Exception:
                                pass
                    elif im not in IMAGE_MODELS:
                        im = "klein"
                    if not im.startswith("wf:") and not image_model_installed(im):
                        self._send(400, {"error": f"Il modello '{IMAGE_MODELS[im]['label']}' non e' "
                                                  "installato: scaricalo dalla sezione Modelli."})
                        return
                    dest = (t.get("dest") or DEFAULT_DEST).strip() or DEFAULT_DEST
                    # ripara i vecchi percorsi salvati nel browser prima dello spostamento
                    if dest.lower().rstrip("\\") == "e:\\llm":
                        dest = DEFAULT_DEST
                    elif dest.lower().startswith("e:\\llm\\"):
                        dest = os.path.join(LLM_DIR, dest[7:])
                    try:
                        seed = int(t.get("seed")) if t.get("seed") not in (None, "") else None
                    except (TypeError, ValueError):
                        seed = None
                    title = (t.get("title") or f"Scheda {len(tabs) + 1}").strip()[:60]
                    persona_sel = {k: str(v) for k, v in (t.get("persona") or {}).items()
                                   if k in _PERSONA_MAP and str(v) in _PERSONA_MAP[k]}
                    # immagini campione richieste dal workflow personalizzato
                    ref_images = {}
                    auto_frames = bool(t.get("auto_frames")) and im.startswith("wf:")
                    if im.startswith("wf:") and not auto_frames:
                        wf_meta = next((w for w in list_custom_workflows()
                                        if w["file"] == im[3:] + ".json"), None)
                        sent = t.get("ref_images") or {}
                        for ph in (wf_meta["ref_slots"] if wf_meta else []):
                            fn = os.path.basename((sent.get(ph) or "").strip())
                            if not fn or not os.path.exists(os.path.join(COMFY_ROOT, "input", fn)):
                                self._send(400, {"error": f"La scheda '{title}' usa un workflow che "
                                                          "richiede un'immagine campione: caricala "
                                                          "nello slot sotto la tendina Workflow."})
                                return
                            ref_images[ph] = fn
                    # ogni scheda salva in una sottocartella col proprio nome
                    safe = re.sub(r'[<>:"/\\|?*]+', "_", title).strip(" .") or "scheda"
                    dest = os.path.join(dest, safe)
                    tabs.append({
                        "title": title,
                        "image_model": im,
                        "ref_images": ref_images,
                        "auto_frames": auto_frames,
                        "literal": bool(t.get("literal")),
                        "styles": [k for k in (t.get("styles") or []) if k in STYLE_OPTIONS],
                        "seed": seed,
                        "tagmode": bool(t.get("tagmode")),
                        # regista foto: n foto = capitoli di una storia cronologica
                        "photo_story": bool(t.get("photo_story")),
                        # col personaggio configurato l'anatomia e' SEMPRE forzata
                        "anatomy": True if persona_sel else bool(t.get("anatomy", True)),
                        "eval_enabled": bool(t.get("eval_enabled")),
                        "eval_every": max(0, min(200, int(t.get("eval_every", 0) or 0))),
                        "eval_deep": bool(t.get("eval_deep")),
                        # criteri di valutazione dell'utente (opzionali): guidano il giudice
                        "eval_criteria": (t.get("eval_criteria") or "").strip()[:600],
                        "max_words": max(20, min(150, int(t.get("max_words", 55) or 55))),
                        # video (workflow con target "video"): durata in secondi e fps
                        "video_secs": max(1.0, min(15.0, float(t.get("video_secs", 5) or 5))),
                        "video_fps": max(12, min(50, int(t.get("video_fps", 24) or 24))),
                        # regista: step delle FOTO Klein (separati dagli step video)
                        "photo_steps": max(1, min(12, int(t.get("photo_steps", 6) or 6))),
                        # opzioni qualita' video (mutazioni del grafo al volo)
                        "video_crf": max(0, min(51, int(t.get("video_crf", 12) or 12))),
                        "video_speed": max(0.5, min(2.0, float(t.get("video_speed", 1) or 1))),
                        "video_easing": bool(t.get("video_easing")),
                        "video_detail": bool(t.get("video_detail")),
                        "video_clean": bool(t.get("video_clean", False)),
                        "ambientazione": amb,
                        "amb_clean": amb_clean,
                        "forced_terms": forced_terms,
                        "soggetto": (t.get("soggetto") or "").strip(),
                        # configuratore persona: scelte valide -> descrizione forzata in ogni foto
                        "persona": persona_sel,
                        # regista: traccia della foto finale e bozza di regia (opzionali)
                        "regia_end": (t.get("regia_end") or "").strip(),
                        "regia_plot": (t.get("regia_plot") or "").strip(),
                        "extra": (t.get("extra") or "").strip(),
                        "num_images": n,
                        "dest": dest,
                        "width": snap_dim(t.get("width", 1024)),
                        "height": snap_dim(t.get("height", 1024)),
                        "steps": max(1, min(50, int(t.get("steps", 6)))),
                        "loras": [l for l in (t.get("loras") or []) if l.get("name")],
                        "prompts": [],
                        "images": [{"prompt": None, "status": "pending", "files": [],
                                    "seed": None, "error": None, "score": None,
                                    "score_note": None} for _ in range(n)],
                    })
                if not tabs:
                    self._send(400, {"error": "Nessuna scheda con un'ambientazione compilata"})
                    return
                jid = uuid.uuid4().hex[:10]
                job = {
                    "id": jid, "name": time.strftime("%Y%m%d_%H%M%S"),
                    "created": time.time(), "ended": None,
                    "tabs": tabs, "phase": "queued", "cancel": False, "error": None,
                    "model": os.path.basename(body.get("model") or "") or None,
                    "launch_at": None, "auto_launch": bool(body.get("auto_launch")),
                    "log": ["Job in coda: " + ", ".join(t["title"] for t in tabs)],
                }
                jobs[jid] = job
                jobs_order.append(jid)
                with queue_lock:
                    work_items.append((jid, "prompts"))
                queue_event.set()
                self._send(200, {"job_id": jid})
            elif self.path.startswith("/api/job/") and self.path.endswith("/launch"):
                jid = self.path.split("/")[3]
                job = jobs.get(jid)
                if not job:
                    self._send(404, {"error": "job non trovato"})
                    return
                if job["phase"] not in ("prompts_ready", "scheduled"):
                    self._send(400, {"error": f"Job in fase '{job['phase']}', non avviabile"})
                    return
                body = self._read_body()
                at = body.get("at")  # epoch ms dal client, oppure null
                launch_at = float(at) / 1000.0 if at else None
                if job["phase"] == "scheduled":
                    job["launch_at"] = launch_at  # riprogramma / avvia subito
                    job["log"].append("Riprogrammato." if launch_at else "Avvio immediato richiesto.")
                    queue_event.set()
                else:
                    launch_job(job, launch_at)
                self._send(200, {"ok": True})
            elif self.path.startswith("/api/job/") and self.path.endswith("/prompts"):
                jid = self.path.split("/")[3]
                job = jobs.get(jid)
                if not job:
                    self._send(404, {"error": "job non trovato"})
                    return
                if job["phase"] not in ("prompts_ready", "scheduled"):
                    self._send(400, {"error": "I prompt si possono modificare solo prima dell'avvio"})
                    return
                body = self._read_body()
                t = int(body.get("tab", 0))
                prompts = [str(p).strip() for p in (body.get("prompts") or []) if str(p).strip()]
                if not prompts or t >= len(job["tabs"]):
                    self._send(400, {"error": "prompt non validi"})
                    return
                tab = job["tabs"][t]
                tab["prompts"] = prompts
                tab["num_images"] = len(prompts)
                tab["images"] = [{"prompt": p, "status": "pending", "files": [],
                                  "seed": None, "error": None} for p in prompts]
                job["log"].append(f"Prompt di '{tab['title']}' aggiornati ({len(prompts)}).")
                self._send(200, {"ok": True})
            elif self.path.startswith("/api/job/") and self.path.endswith("/remove"):
                jid = self.path.split("/")[3]
                job = jobs.get(jid)
                if not job:
                    self._send(404, {"error": "job non trovato"})
                    return
                if job["phase"] not in ("done", "error", "cancelled", "prompts_ready", "scheduled"):
                    self._send(400, {"error": "Job in esecuzione: annullalo prima di rimuoverlo"})
                    return
                remove_job(jid)
                self._send(200, {"ok": True})
            elif self.path.startswith("/api/job/") and self.path.endswith("/delete_image"):
                jid = self.path.split("/")[3]
                job = jobs.get(jid)
                if not job:
                    self._send(404, {"error": "job non trovato"})
                    return
                body = self._read_body()
                t = int(body.get("tab", -1))
                i = int(body.get("i", -1))
                if not (0 <= t < len(job["tabs"])) or not (0 <= i < len(job["tabs"][t]["images"])):
                    self._send(400, {"error": "indice non valido"})
                    return
                tab = job["tabs"][t]
                img = tab["images"][i]
                for fp in img["files"]:
                    try:
                        os.remove(fp)
                    except FileNotFoundError:
                        pass
                img["files"] = []
                img["status"] = "deleted"
                img["prompt"] = None
                # riscrive il file dei prompt senza quelli delle foto eliminate
                write_prompts_txt(job, t, tab)
                job["log"].append(f"Foto {i + 1} di '{tab['title']}' eliminata (con il suo prompt).")
                # era l'ultima foto di un job concluso? il job se ne va con lei
                if job["phase"] in ("done", "error", "cancelled") and \
                        not any(im.get("files") for t2 in job["tabs"] for im in t2["images"]):
                    remove_job(jid)
                    self._send(200, {"ok": True, "job_removed": True})
                    return
                self._send(200, {"ok": True})
            elif self.path == "/api/regen":
                # rigenera a partire dal prompt (pulito) di una foto: crea un nuovo job gia'
                # "prompts_ready" con quel prompt, riusando le impostazioni della scheda sorgente.
                # L'utente puo' rivedere/modificare il prompt e poi avviare.
                body = self._read_body()
                src = jobs.get(body.get("job"))
                t = int(body.get("tab", -1))
                i = int(body.get("i", -1))
                n = max(1, min(20, int(body.get("count", 1) or 1)))
                modify = (body.get("modify") or "").strip()
                do_launch = bool(body.get("launch"))
                if not src or not (0 <= t < len(src["tabs"])):
                    self._send(404, {"error": "job sorgente non trovato"})
                    return
                stab = src["tabs"][t]
                if not (0 <= i < len(stab["images"])) or not stab["images"][i].get("prompt"):
                    self._send(400, {"error": "questa foto non ha un prompt riutilizzabile"})
                    return
                prompt = stab["images"][i]["prompt"]
                newtab = {
                    "title": (stab.get("title") or "Scheda") + " (rigen)",
                    "image_model": stab.get("image_model", "klein"),
                    "ref_images": stab.get("ref_images") or {},
                    "literal": bool(stab.get("literal")),
                    "styles": [],                 # niente ri-controlli: e' gia' un prompt finito
                    # chip di origine tenuti come METADATO (non influenzano la generazione): servono
                    # a "Carica i valori nell'editor" per rimettere i tag anche da una foto rigenerata
                    "src_styles": list(stab.get("styles") or stab.get("src_styles") or []),
                    "seed": None,                 # nuovo seed a ogni foto = varianti
                    "tagmode": bool(stab.get("tagmode")),
                    "anatomy": bool(stab.get("anatomy", True)),
                    "eval_enabled": bool(stab.get("eval_enabled")),
                    "eval_every": int(stab.get("eval_every", 0) or 0),
                    "eval_deep": bool(stab.get("eval_deep")),
                    "max_words": stab.get("max_words", MAX_PROMPT_WORDS),
                    "ambientazione": "(rigenerazione da un prompt esistente)",
                    "amb_clean": "", "forced_terms": [], "fixed_frags": [],
                    "soggetto": "", "extra": "",
                    "num_images": n, "dest": stab["dest"],
                    "width": snap_dim(stab.get("width", 1024)),
                    "height": snap_dim(stab.get("height", 1024)),
                    "steps": stab.get("steps", 6),
                    "loras": stab.get("loras") or [],
                    "prompts": [prompt] * n,
                    "images": [{"prompt": prompt, "status": "pending", "files": [],
                                "seed": None, "error": None, "score": None,
                                "score_note": None} for _ in range(n)],
                    # modifica richiesta: il prompt verra' rivisto dallo scrittore al lancio
                    "regen_modify": modify, "regen_applied": False,
                }
                jid = uuid.uuid4().hex[:10]
                if modify and do_launch:
                    intro = f"Rigenerazione con modifica «{modify}»: rivedo il prompt e avvio subito."
                elif do_launch:
                    intro = "Rigenerazione avviata."
                else:
                    intro = "Rigenerazione da un prompt esistente. Rivedi il prompt e avvia."
                job = {
                    "id": jid, "name": time.strftime("%Y%m%d_%H%M%S"),
                    "created": time.time(), "ended": None,
                    "tabs": [newtab], "phase": "prompts_ready", "cancel": False, "error": None,
                    "model": None, "launch_at": None, "auto_launch": False,
                    "log": [intro],
                }
                jobs[jid] = job
                jobs_order.append(jid)
                if do_launch:
                    launch_job(job, None)   # avvio immediato (in coda dopo eventuali job in corso)
                save_jobs()
                self._send(200, {"job_id": jid, "launched": do_launch})
            elif self.path == "/api/loadtab":
                # ricostruisce una SCHEDA in formato editor dai dati interni del job, per riusare
                # prompt + tag + impostazioni di una foto specifica (tasto destro -> carica valori).
                body = self._read_body()
                src = jobs.get(body.get("job"))
                t = int(body.get("tab", -1))
                i = int(body.get("i", -1))
                if not src or not (0 <= t < len(src["tabs"])):
                    self._send(404, {"error": "job sorgente non trovato"})
                    return
                stab = src["tabs"][t]
                if not (0 <= i < len(stab["images"])):
                    self._send(400, {"error": "foto non valida"})
                    return
                im = stab["images"][i]
                # scelta A: il prompt (LLM) di QUELLA foto nello scenario + modalita' letterale ON;
                # i tag (chip) e tutte le impostazioni vengono copiati cosi' come sono.
                prompt = im.get("prompt") or im.get("final_prompt") or ""
                seed = im.get("seed")
                # tag/chip: prima gli styles salvati (batch normale) o src_styles (rigenerazioni nuove);
                # se non ci sono (es. vecchie rigenerazioni) li RILEVA dal prompt della foto, cercando la
                # stringa "check" di ogni chip nel testo effettivo -> funziona anche sulle foto gia' fatte.
                styles = list(stab.get("styles") or stab.get("src_styles") or [])
                if not styles:
                    pl = (im.get("final_prompt") or im.get("prompt") or "").lower()

                    def _present(chk):
                        # cerca la stringa MA la ignora se e' negata ("no body hair", "senza..."):
                        # senza questo, "no body hair" attiverebbe per sbaglio il chip Peluria.
                        idx = pl.find(chk)
                        while idx != -1:
                            pre = pl[max(0, idx - 14):idx]
                            if not re.search(r"\b(no|not|non|without|senza|zero)\b[\s\w-]*$", pre):
                                return True
                            idx = pl.find(chk, idx + 1)
                        return False
                    styles = [k for k, v in STYLE_OPTIONS.items()
                              if v.get("check") and _present(v["check"].lower())]
                loras = {l["name"]: l.get("strength", 1.0)
                         for l in (stab.get("loras") or []) if isinstance(l, dict) and l.get("name")}
                refs = {ph: {"file": f} for ph, f in (stab.get("ref_images") or {}).items()}
                tabform = {
                    "title": (stab.get("title") or "Scheda").split(" (rigen)")[0] + " (copia)",
                    "ambientazione": prompt,
                    "soggetto": "", "extra": "",
                    "num": int(stab.get("num_images", 6) or 6),
                    "size": f"{snap_dim(stab.get('width', 1024))}x{snap_dim(stab.get('height', 1024))}",
                    "steps": int(stab.get("steps", 6) or 6),
                    "dest": stab.get("dest") or DEFAULT_DEST,
                    "loras": loras,
                    "imodel": stab.get("image_model", "klein"),
                    "literal": True,
                    "seed": str(seed) if seed else "",
                    "styles": styles,
                    "refs": refs,
                    "maxwords": int(stab.get("max_words", MAX_PROMPT_WORDS) or MAX_PROMPT_WORDS),
                    "tagmode": bool(stab.get("tagmode")),
                    "anatomy": bool(stab.get("anatomy", True)),
                    "evalon": bool(stab.get("eval_enabled")),
                    "evalevery": int(stab.get("eval_every", 0) or 0),
                    "evaldeep": bool(stab.get("eval_deep")),
                    "evalcriteria": stab.get("eval_criteria") or "",
                }
                self._send(200, {"tab": tabform})
            elif self.path.startswith("/api/job/") and self.path.endswith("/cancel"):
                jid = self.path.split("/")[3]
                if jid in jobs:
                    jobs[jid]["cancel"] = True
                    jobs[jid]["log"].append("Richiesta di annullamento...")
                    queue_event.set()
                    self._send(200, {"ok": True})
                else:
                    self._send(404, {"error": "job non trovato"})
            elif self.path == "/api/agent/warmup":
                if agent_warm["state"] not in ("loading", "ready") or \
                        (agent_warm["state"] == "ready" and not slot_alive(SLOT_AGENT)):
                    agent_warm["state"] = "loading"
                    threading.Thread(target=lambda: _safe_warmup(), daemon=True).start()
                self._send(200, {"state": agent_warm["state"]})
            elif self.path == "/api/agent/start":
                body = self._read_body()
                target = body.get("target", "klein")
                if target not in IMAGE_MODELS:
                    self._send(400, {"error": "modello di destinazione sconosciuto"})
                    return
                msg = (body.get("message") or "").strip()
                if not msg:
                    self._send(400, {"error": "scrivi cosa deve fare il workflow"})
                    return
                aid = uuid.uuid4().hex[:10]
                a = {
                    "id": aid, "target": target, "created": time.time(),
                    "status": "queued", "cancel": False, "iteration": 0, "max_iter": 6,
                    "chat": [{"role": "user", "text": msg}], "chat_log": [],
                    "llm_messages": [
                        {"role": "system", "content": agent_system_prompt(target)},
                        {"role": "user", "content": msg},
                    ],
                    "workflow": None, "preview": None,
                    "stream_text": None, "tps": None,
                    "test_steps": min(IMAGE_MODELS[target]["steps"], 12),
                }
                agents[aid] = a
                threading.Thread(target=run_agent_cycle, args=(a,), daemon=True).start()
                self._send(200, {"id": aid})
            elif self.path.startswith("/api/agent/") and self.path.endswith("/message"):
                aid = self.path.split("/")[3]
                a = agents.get(aid)
                if not a:
                    self._send(404, {"error": "sessione non trovata"})
                    return
                if a["status"] in ("queued", "loading", "working", "testing"):
                    self._send(400, {"error": "l'agente sta ancora lavorando"})
                    return
                msg = (self._read_body().get("message") or "").strip()
                if not msg:
                    self._send(400, {"error": "messaggio vuoto"})
                    return
                a["chat"].append({"role": "user", "text": msg})
                a["llm_messages"].append({"role": "user", "content": msg +
                    "\nRispondi con il workflow JSON COMPLETO aggiornato."})
                a["cancel"] = False
                threading.Thread(target=run_agent_cycle, args=(a,), daemon=True).start()
                self._send(200, {"ok": True})
            elif self.path.startswith("/api/agent/") and self.path.endswith("/cancel"):
                aid = self.path.split("/")[3]
                if aid in agents:
                    agents[aid]["cancel"] = True
                    self._send(200, {"ok": True})
                else:
                    self._send(404, {"error": "sessione non trovata"})
            elif self.path.startswith("/api/agent/") and self.path.endswith("/save"):
                aid = self.path.split("/")[3]
                a = agents.get(aid)
                body = self._read_body()
                name = (body.get("name") or "").strip()[:50]
                if not a or not a.get("workflow"):
                    self._send(400, {"error": "nessun workflow pronto da salvare"})
                    return
                if not name:
                    self._send(400, {"error": "dai un nome al workflow"})
                    return
                safe = re.sub(r'[^\w\- ]+', "_", name).strip() or "workflow"
                os.makedirs(WORKFLOWS_DIR, exist_ok=True)
                wf_json = json.dumps(a["workflow"])
                with open(os.path.join(WORKFLOWS_DIR, safe + ".json"), "w", encoding="utf-8") as f:
                    json.dump({"meta": {"name": name, "target": a["target"],
                                        "created": time.time(),
                                        "steps": IMAGE_MODELS[a["target"]]["steps"],
                                        "needs_ref": "{REF_IMAGE}" in wf_json},
                               "graph": a["workflow"]}, f, ensure_ascii=False, indent=1)
                self._send(200, {"ok": True, "key": "wf:" + safe})
            elif self.path == "/api/upload_ref":
                # riceve un'immagine campione (dataURL base64) e la salva nell'input di ComfyUI
                body = self._read_body()
                m = re.match(r"data:image/(png|jpe?g|webp);base64,(.+)$",
                             body.get("data") or "", re.S)
                if not m:
                    self._send(400, {"error": "immagine non valida: usa PNG, JPG o WEBP"})
                    return
                try:
                    blob = base64.b64decode(m.group(2))
                except Exception:
                    self._send(400, {"error": "dati immagine corrotti"})
                    return
                if len(blob) > 50 * 1024 * 1024:
                    self._send(400, {"error": "immagine troppo grande (max 50MB)"})
                    return
                ext = "jpg" if m.group(1).startswith("jp") else m.group(1)
                # nome dal contenuto: ricaricare la stessa foto non crea duplicati
                fname = "ps_ref_" + hashlib.sha1(blob).hexdigest()[:12] + "." + ext
                inp = os.path.join(COMFY_ROOT, "input")
                os.makedirs(inp, exist_ok=True)
                with open(os.path.join(inp, fname), "wb") as f:
                    f.write(blob)
                self._send(200, {"file": fname})
            elif self.path == "/api/characters/save":
                body = self._read_body()
                name = (body.get("name") or "").strip()[:40]
                sel = body.get("persona") or {}
                if not name or not isinstance(sel, dict):
                    self._send(400, {"error": "nome o configurazione mancante"})
                    return
                # tengo solo le scelte valide (sezione nota + opzione nota)
                clean = {k: str(v) for k, v in sel.items()
                         if k in _PERSONA_MAP and str(v) in _PERSONA_MAP[k]}
                chars = load_characters()
                chars[name] = clean
                save_characters(chars)
                self._send(200, {"ok": True, "characters": chars})
            elif self.path == "/api/characters/delete":
                body = self._read_body()
                chars = load_characters()
                chars.pop((body.get("name") or "").strip(), None)
                save_characters(chars)
                self._send(200, {"ok": True, "characters": chars})
            elif self.path == "/api/workflows/delete":
                body = self._read_body()
                fn = os.path.basename((body.get("file") or ""))
                fp = os.path.join(WORKFLOWS_DIR, fn if fn.endswith(".json") else fn + ".json")
                if os.path.exists(fp):
                    os.remove(fp)
                    self._send(200, {"ok": True})
                else:
                    self._send(404, {"error": "workflow non trovato"})
            elif self.path == "/api/presets/save":
                body = self._read_body()
                job = jobs.get(body.get("job"))
                name = (body.get("name") or "").strip()[:40]
                if not job or not name:
                    self._send(400, {"error": "nome o job mancante"})
                    return
                t = int(body.get("tab", -1))
                i = int(body.get("i", -1))
                if not (0 <= t < len(job["tabs"])) or not (0 <= i < len(job["tabs"][t]["images"])):
                    self._send(400, {"error": "indice non valido"})
                    return
                tab = job["tabs"][t]
                img = tab["images"][i]
                if not img["files"]:
                    self._send(400, {"error": "la foto non esiste piu'"})
                    return
                try:
                    thumb = make_thumb(img["files"][0])
                except Exception as e:
                    self._send(500, {"error": f"miniatura fallita: {e}"})
                    return
                preset = {
                    "id": uuid.uuid4().hex[:10], "name": name, "created": time.time(),
                    "seed": img["seed"], "image_model": tab.get("image_model", "klein"),
                    "width": tab["width"], "height": tab["height"], "steps": tab["steps"],
                    "loras": tab["loras"], "soggetto": tab.get("soggetto", ""),
                    "prompt": img["prompt"], "thumb": thumb,
                }
                with presets_lock:
                    presets = load_presets()
                    presets.append(preset)
                    save_presets(presets)
                self._send(200, {"ok": True, "id": preset["id"]})
            elif self.path == "/api/presets/delete":
                body = self._read_body()
                pid = body.get("id")
                with presets_lock:
                    presets = [p for p in load_presets() if p["id"] != pid]
                    save_presets(presets)
                self._send(200, {"ok": True})
            elif self.path == "/api/models_download":
                key = self._read_body().get("key")
                if key not in MODEL_CATALOG and key not in COMPONENT_CATALOG:
                    self._send(400, {"error": "voce sconosciuta"})
                    return
                with downloads_lock:
                    st = DOWNLOADS.get(key)
                    if st and st.get("status") == "downloading":
                        self._send(200, {"ok": True})
                        return
                    DOWNLOADS[key] = {"status": "downloading", "cancel": False, "file": "",
                                      "file_n": 0, "files_total": 0, "done": 0, "total": 0,
                                      "speed": 0, "error": None, "phase": "scarico"}
                worker = component_worker if key in COMPONENT_CATALOG else download_model_worker
                threading.Thread(target=worker, args=(key,), daemon=True).start()
                self._send(200, {"ok": True})
            elif self.path == "/api/models_cancel":
                key = self._read_body().get("key")
                if key in DOWNLOADS:
                    DOWNLOADS[key]["cancel"] = True
                self._send(200, {"ok": True})
            elif self.path == "/api/unload":
                # ferma e dimentica tutte le sessioni agente
                for a in agents.values():
                    a["cancel"] = True
                agents.clear()
                stop_llama_slot(SLOT_MAIN)
                stop_llama_slot(SLOT_AGENT)
                stop_llama_slot(SLOT_EVAL)
                agent_warm["state"] = "idle"
                try:
                    http_json(COMFY_URL + "/free", {"unload_models": True, "free_memory": True},
                              timeout=15)
                except Exception:
                    pass
                self._send(200, {"ok": True})
            elif self.path == "/api/update/apply":
                if _jobs_busy():
                    self._send(409, {"error": "Ci sono lavori in corso: attendi che finiscano prima di aggiornare."})
                    return
                chk = check_update()
                if not chk.get("update_available"):
                    self._send(400, {"error": chk.get("error") or "Nessun aggiornamento disponibile."})
                    return
                _UPDATE.update({"phase": "starting", "msg": "", "error": ""})
                threading.Thread(target=_do_update, daemon=True).start()
                self._send(200, {"ok": True, "latest": chk.get("latest")})
            elif self.path == "/api/update/config":
                body = self._read_body()
                cfg = get_config()
                if "github_repo" in body:
                    cfg["github_repo"] = str(body.get("github_repo") or "").strip().strip("/")
                if "auto_update" in body:
                    cfg["auto_update"] = bool(body.get("auto_update"))
                with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
                self._send(200, {"ok": True, "github_repo": cfg.get("github_repo", ""),
                                 "auto_update": bool(cfg.get("auto_update"))})
            elif self.path == "/api/shutdown":
                self._send(200, {"ok": True})
                threading.Thread(target=_shutdown_all, daemon=True).start()
            elif self.path == "/api/open_folder":
                body = self._read_body()
                # con job/tab/i apre Esplora risorse CON LA FOTO SELEZIONATA
                j2 = jobs.get(body.get("job") or "")
                if j2 is not None and body.get("tab") is not None:
                    fp = None
                    try:
                        fp = j2["tabs"][int(body["tab"])]["images"][int(body["i"])]["files"][0]
                    except (KeyError, IndexError, TypeError, ValueError):
                        pass
                    if fp and os.path.isfile(fp):
                        subprocess.Popen(["explorer", "/select,", os.path.normpath(fp)])
                        self._send(200, {"ok": True})
                    else:
                        self._send(404, {"error": "file della foto non trovato"})
                    return
                p = body.get("path", "")
                if os.path.isdir(p):
                    os.startfile(p)
                    self._send(200, {"ok": True})
                else:
                    self._send(400, {"error": "cartella inesistente"})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": str(e)})


def _shutdown_all():
    """Arresta tutto: LLM, ComfyUI e questo server (equivalente di Arresta_Prompt_Studio.bat)."""
    time.sleep(0.5)
    stop_llama_slot(SLOT_MAIN)
    stop_llama_slot(SLOT_AGENT)
    stop_llama_slot(SLOT_EVAL)
    if psutil:
        for p in psutil.process_iter(["name", "cmdline"]):
            try:
                cl = " ".join(p.info.get("cmdline") or [])
                if "main.py" in cl and "ComfyUI" in cl:
                    p.kill()
            except Exception:
                pass
    os._exit(0)


def _safe_warmup():
    try:
        ensure_agent_loaded([])
    except Exception:
        pass


def job_public(job, full=False):
    total = sum(len(t["images"]) for t in job["tabs"])
    done = sum(1 for t in job["tabs"] for im in t["images"] if im["status"] == "done")
    # Avanzamento e ETA A TEMPO STIMATO PER-ELEMENTO: ogni render pesa per la SUA durata
    # prevista, presa dalla calibrazione storica per (modello, risoluzione, step, frame).
    # Niente medie miste: una foto Klein da 15s e un video LTX da 6 min nello stesso job
    # hanno stime indipendenti. Il residuo si corregge col progresso reale degli step.
    now_t = time.time()

    def est_item(tab, idx):
        """Durata prevista del singolo render (senza caricamento modello)."""
        key = item_timing_key(tab, idx)
        est = timing_est(key)
        if est is None:
            # mai visto: media dei render con la STESSA chiave gia' finiti in questo job
            same = [x["dur"] for tt in job["tabs"] for j2, x in enumerate(tt["images"])
                    if x.get("dur") and item_timing_key(tt, j2) == key]
            est = (sum(same) / len(same)) if same else None
        if est is None:
            # prima stima (nessuna calibrazione): scala con frame/pixel/step invece di un
            # numero fisso, cosi' un video di 3s a 512 e uno di 10s a 1152 non hanno la
            # stessa attesa. Dopo il primo render vero, l'EMA per-chiave prende il sopravvento.
            if key.endswith("f"):
                try:
                    frames = int(key.rsplit("|", 1)[1][:-1])
                except Exception:
                    frames = 121
                mp = (tab["width"] * tab["height"]) / (640 * 960)
                est = 2.5 * frames * max(0.4, mp) * (tab.get("steps", 8) / 8.0)
            else:
                mp = (tab["width"] * tab["height"]) / (1024 * 1024)
                est = (8.0 + tab["steps"] * 1.4) * max(0.5, mp)
            est = max(4.0, est)
        return est

    def load_ov(tab, idx, prev_model):
        """Overhead di caricamento quando il modello cambia rispetto al render precedente."""
        model = item_timing_key(tab, idx).split("|")[1]
        ov = (timing_est("load|" + model) or 0.0) if model != prev_model else 0.0
        return model, ov

    # stima totale per ogni elemento (con overhead ai cambi di modello) nell'ordine reale
    est_map = {}
    prev_model = None
    for ti, t in enumerate(job["tabs"]):
        for i in range(len(t["images"])):
            prev_model, ov = load_ov(t, i, prev_model)
            est_map[(ti, i)] = est_item(t, i) + ov

    def im_frac(im, est_total):
        if im.get("status") != "generating":
            return None, None
        t0 = im.get("t_start")
        elapsed = max(0.0, now_t - t0) if t0 else 0.0
        p = progress_info(im.get("pid"))
        if p and p[2] is not None:
            remaining = float(p[2])          # ETA reale dagli step del render in corso
        else:
            remaining = max(0.6, est_total - elapsed)
        denom = elapsed + remaining
        fr = (100.0 * elapsed / denom) if denom > 0 else 0.0
        return round(max(0.0, min(99.0, fr)), 1), int(remaining)

    def img_pub(im, est_total):
        fr, et = im_frac(im, est_total)
        is_video = bool(im["files"]) and str(im["files"][0]).lower().endswith(
            (".mp4", ".webm", ".mov", ".mkv"))
        return {"prompt": im["prompt"], "status": im["status"], "seed": im["seed"],
                "error": im["error"], "has_file": bool(im["files"]), "video": is_video,
                "dur": round(im["dur"]) if im.get("dur") else None,     # solo render
                "wall": round(im["wall"]) if im.get("wall") else None,  # con caricamenti
                "final_prompt": im.get("final_prompt"),
                "score": im.get("score"), "score_note": im.get("score_note"),
                "preview": im.get("preview") if im["status"] == "generating" else None,
                "frac": fr, "eta": et,
                "prog": (progress_info(im.get("pid")) if im["status"] == "generating" else None)}
    # aggregato del job: ogni elemento pesa per la sua stima; il residuo dell'elemento
    # in corso arriva dagli step reali quando disponibili. steps_total/steps_done restano
    # i nomi dei campi ma ora sono SECONDI stimati (il frontend fa done/total, invariato).
    eta, eta_unknown = 0.0, False
    t_all = sum(est_map.values()) or 1.0
    t_done = 0.0
    for ti, t in enumerate(job["tabs"]):
        for i, im in enumerate(t["images"]):
            e = est_map[(ti, i)]
            st = im["status"]
            if st in ("done", "error", "cancelled", "deleted"):
                t_done += e
            elif st == "generating":
                fr, rem = im_frac(im, e)
                t_done += e * (fr or 0.0) / 100.0
                eta += rem if rem is not None else e
            elif st == "pending":
                eta += e
    steps_total = round(t_all)
    steps_done = round(min(t_done, t_all))
    d = {
        "id": job["id"], "name": job["name"], "created": job["created"], "ended": job["ended"],
        "phase": job["phase"], "error": job["error"], "launch_at": job["launch_at"],
        "total": total, "done": done,
        "steps_total": steps_total, "steps_done": steps_done,
        "eta_total": None if (eta_unknown or job["phase"] != "generating") else int(eta),
        "eval_total": job.get("eval_total", 0), "eval_done": job.get("eval_done", 0),
        "tab_titles": [t["title"] for t in job["tabs"]],
    }
    if full:
        d["log"] = job["log"][-40:]
        d["tabs"] = []
        for ti, t in enumerate(job["tabs"]):
            d["tabs"].append({
                "title": t["title"], "ambientazione": t["ambientazione"],
                "image_model": t.get("image_model", "klein"),
                "dest": t["dest"], "width": t["width"], "height": t["height"],
                "num_images": t["num_images"],
                "fixed_frags": t.get("fixed_frags") or [],
                # configurazione con cui e' stato lanciato il job (per il riepilogo in UI)
                "steps": t.get("steps"), "seed": t.get("seed"),
                "photo_steps": t.get("photo_steps"),
                "literal": bool(t.get("literal")), "tagmode": bool(t.get("tagmode")),
                "anatomy": bool(t.get("anatomy", True)), "max_words": t.get("max_words"),
                "auto_frames": bool(t.get("auto_frames")), "soggetto": t.get("soggetto") or "",
                "styles": t.get("styles") or [], "persona": t.get("persona") or {},
                "loras": t.get("loras") or [],
                "video_secs": t.get("video_secs"), "video_fps": t.get("video_fps"),
                "video_speed": t.get("video_speed"), "video_crf": t.get("video_crf"),
                "video_easing": bool(t.get("video_easing")),
                "video_detail": bool(t.get("video_detail")),
                "is_video": _wf_target_video(t),
                "images": [img_pub(im, est_map[(ti, i)])
                           for i, im in enumerate(t["images"])],
            })
    return d


if __name__ == "__main__":
    # il template jinja per Nemo viaggia con la app: copialo in models/ se manca
    _src = os.path.join(APP_DIR, "mistral-nemo.jinja")
    if not os.path.exists(NEMO_TEMPLATE) and os.path.exists(_src):
        os.makedirs(MODELS_DIR, exist_ok=True)
        shutil.copy2(_src, NEMO_TEMPLATE)
    cleanup_zombie_llamas()
    load_jobs()   # ripristina i job dell'ultima sessione (non spariscono al riavvio)
    load_timings()   # calibrazione tempi per modello/risoluzione (stime oneste)
    threading.Thread(target=worker_loop, daemon=True).start()
    threading.Thread(target=progress_monitor, daemon=True).start()
    threading.Thread(target=jobs_saver_loop, daemon=True).start()
    threading.Thread(target=_startup_update_check, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", APP_PORT), Handler)
    print(f"Prompt Studio: http://127.0.0.1:{APP_PORT}")
    server.serve_forever()
