@echo off
rem Запуск Ollama и веб-приложения проекта.
rem Требуется установленное виртуальное окружение .venv в каталоге проекта.

setlocal
cd /d "%~dp0.."

call "%~dp0start_ollama.bat"
timeout /t 3 /nobreak >nul

start "Legal Assistant" ".venv\Scripts\python.exe" app.py
echo Приложение запускается: http://localhost:8000
endlocal
