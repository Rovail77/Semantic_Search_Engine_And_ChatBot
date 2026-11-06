@echo off
rem Запуск локального сервера Ollama для проекта.
rem Если Ollama установлена в стандартный каталог пользователя,
rem используется она; иначе вызывается команда ollama из PATH.

setlocal
set "OLLAMA_HOST=127.0.0.1:11434"
set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"

if exist "%OLLAMA_EXE%" (
    start "Ollama" "%OLLAMA_EXE%" serve
) else (
    start "Ollama" ollama serve
)

echo Ollama запускается на %OLLAMA_HOST%
endlocal
