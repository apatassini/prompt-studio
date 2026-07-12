@echo off
title Arresta Prompt Studio
echo Arresto Prompt Studio, LLM e ComfyUI...
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'llama-server.exe') -or ($_.CommandLine -match 'ComfyUI.main\.py') -or ($_.CommandLine -match 'LLM.app.server\.py') } | ForEach-Object { Write-Host ('  chiudo: ' + $_.Name + ' (PID ' + $_.ProcessId + ')'); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
echo.
echo Fatto: tutto arrestato.
ping -n 4 127.0.0.1 >nul
