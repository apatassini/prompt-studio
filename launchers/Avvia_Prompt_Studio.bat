@echo off
title Prompt Studio
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "URL=http://127.0.0.1:8500"

rem Python: quello di ComfyUI se c'e', altrimenti il mini-python del bootstrap
set "PY=%ROOT%python_embeded\python.exe"
if not exist "%PY%" set "PY=%ROOT%minipython\python.exe"

echo ============================================================
echo    PROMPT STUDIO
echo.
echo    Indirizzo:  %URL%
echo ============================================================
echo.

rem ---- ComfyUI: avvialo (nascosto) solo se il runtime e' installato ----
if not exist "%ROOT%ComfyUI\main.py" goto :nocomfy
curl -s -o nul --max-time 2 http://127.0.0.1:8188/system_stats
if not errorlevel 1 goto :comfyok
echo  - Avvio ComfyUI in background...
powershell -NoProfile -Command "Start-Process -WindowStyle Hidden -FilePath '%ROOT%python_embeded\python.exe' -ArgumentList '-s','ComfyUI\main.py','--windows-standalone-build' -WorkingDirectory '%ROOT%.' -RedirectStandardOutput '%ROOT%comfyui.log' -RedirectStandardError '%ROOT%comfyui.err.log'"
goto :app
:comfyok
echo  - ComfyUI gia' attivo.
goto :app
:nocomfy
echo  - Runtime ComfyUI non ancora installato ^(si scarica dal pulsante "Modelli"^).

:app
rem ---- Prompt Studio: avvialo (nascosto) solo se non e' gia' attivo ----
curl -s -o nul --max-time 2 %URL%/api/status
if not errorlevel 1 goto :ready
echo  - Avvio Prompt Studio in background...
powershell -NoProfile -Command "Start-Process -WindowStyle Hidden -FilePath '%PY%' -ArgumentList '%ROOT%LLM\app\server.py' -WorkingDirectory '%ROOT%LLM\app' -RedirectStandardOutput '%ROOT%LLM\prompt_studio.log' -RedirectStandardError '%ROOT%LLM\prompt_studio.err.log'"

echo.
echo  Attendo che l'interfaccia sia pronta.
echo  Al primo avvio puo' servire qualche minuto ^(Windows controlla i file
echo  appena estratti^): lascia lavorare, i puntini indicano che sta partendo.
echo.

set /a tries=0
:waitloop
curl -s -o nul --max-time 3 %URL%/api/status
if not errorlevel 1 goto :ready
set /a tries+=1
set /a mod=tries %% 10
if %mod%==0 echo    ...ancora un momento, sto avviando...
if %tries% geq 120 goto :slow
ping -n 3 127.0.0.1 >nul
goto :waitloop

:ready
echo.
echo.
echo ============================================================
echo    PRONTO -^> apro il browser su  %URL%
echo ============================================================
start "" "%URL%"
echo.
echo  Se il browser non si apre da solo, apri Chrome/Edge e vai su:
echo        %URL%
echo  ^(oppure usa l'icona "Prompt Studio (browser)" sul desktop^)
echo.
echo  I server girano in background: questa finestra si puo' chiudere.
timeout /t 8 >nul
exit /b 0

:slow
echo.
echo.
echo ============================================================
echo    L'avvio sta impiegando piu' del solito.
echo    Provo comunque ad aprire il browser su:
echo        %URL%
echo ============================================================
start "" "%URL%"
echo.
echo  Se la pagina non carica subito, attendi un minuto e premi F5.
echo  Se dopo qualche minuto resta vuota, apri questo file e mandamelo:
echo        %ROOT%LLM\prompt_studio.err.log
echo.
pause
exit /b 0
