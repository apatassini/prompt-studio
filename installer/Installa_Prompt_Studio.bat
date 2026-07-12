@echo off
title Prompt Studio - Installazione
cd /d "%~dp0"

if exist "ComfyUI_windows_portable\LLM\app\server.py" goto :refresh

echo ============================================
echo  PROMPT STUDIO - prima installazione
echo ============================================
echo.
echo Estrazione dei file base...
tar -xf bootstrap.zip
if errorlevel 1 (
    echo ERRORE nell'estrazione. Serve Windows 10/11 con tar integrato.
    pause
    exit /b 1
)

rem Variante offline completa: se payload.7z e' presente, estrae anche il runtime
if exist "payload.7z" (
    echo Estrazione del runtime completo - alcuni minuti...
    tar -xf payload.7z
)

echo Creo le icone sul desktop...
powershell -NoProfile -Command "$ws=New-Object -ComObject WScript.Shell; $d=[Environment]::GetFolderPath('Desktop'); $root='%~dp0ComfyUI_windows_portable'; $s=$ws.CreateShortcut(\"$d\Prompt Studio.lnk\"); $s.TargetPath=\"$root\Avvia_Prompt_Studio.bat\"; $s.WorkingDirectory=$root; $s.Description='Avvia Prompt Studio'; $s.IconLocation=\"$root\play.ico,0\"; $s.Save(); $s2=$ws.CreateShortcut(\"$d\Arresta Prompt Studio.lnk\"); $s2.TargetPath=\"$root\Arresta_Prompt_Studio.bat\"; $s2.WorkingDirectory=$root; $s2.Description='Arresta Prompt Studio'; $s2.IconLocation=\"$root\stop.ico,0\"; $s2.Save(); Set-Content -Path \"$d\Prompt Studio (browser).url\" -Encoding ASCII -Value \"[InternetShortcut]`r`nURL=http://127.0.0.1:8500`r`nIconFile=$root\play.ico`r`nIconIndex=0\"; Write-Output $d" > "%TEMP%\ps_desk.txt"
set "DESKDIR="
for /f "usebackq delims=" %%D in ("%TEMP%\ps_desk.txt") do set "DESKDIR=%%D"
del "%TEMP%\ps_desk.txt" 2>nul

echo.
echo ============================================
echo  Installazione completata.
echo ============================================
echo.
echo  Icone create in questa cartella:
echo    %DESKDIR%
echo  ^(Prompt Studio / Arresta Prompt Studio / Prompt Studio browser^)
echo.
echo  IMPORTANTE - se non le trovi sul desktop:
echo  avvii e spegni il programma anche da qui, dentro la cartella
echo    %~dp0ComfyUI_windows_portable
echo  con  Avvia_Prompt_Studio.bat  e  Arresta_Prompt_Studio.bat
echo.
echo  Apro ora quella cartella: dentro trovi Avvia_Prompt_Studio (evidenziato).
echo.
pause
rem apre la cartella col programma d'avvio gia' evidenziato: niente caccia sul desktop
explorer /select,"%~dp0ComfyUI_windows_portable\Avvia_Prompt_Studio.bat"
echo.

:refresh
rem L'interfaccia web viaggia in app.zip: basta sostituirlo per aggiornarla
if exist "app.zip" tar -xf app.zip

call "ComfyUI_windows_portable\Avvia_Prompt_Studio.bat"
