@echo off
rem Da usare sul PC di sviluppo: rigenera app.zip prendendo l'interfaccia
rem corrente da E:\ComfyUI_windows_portable\LLM\app (esclusi i dati personali).
title Aggiorna app.zip
cd /d E:\
del "%~dp0app.zip" 2>nul
"C:\Program Files\7-Zip\7z.exe" a -tzip -mx=5 "%~dp0app.zip" "ComfyUI_windows_portable\LLM\app" -xr!__pycache__ "-x!ComfyUI_windows_portable\LLM\app\presets.json" "-xr!ComfyUI_windows_portable\LLM\app\workflows"
echo.
echo app.zip aggiornato con l'interfaccia corrente.
pause
