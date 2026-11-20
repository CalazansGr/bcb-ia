@echo off
rem Modo "pergunte por data": voce digita uma data e o mini-LLM escreve o boletim daquele dia.
setlocal
cd /d "%~dp0"
chcp 65001 >nul

if not exist ".venv\instalado.ok" (
  echo.
  echo A instalacao ainda nao terminou ^(ou falhou^).
  echo.
  echo  1. Rode o instalar.bat
  echo  2. ESPERE aparecer a palavra "Pronto!" na janela dele
  echo  3. So depois abra este conversar.bat
  echo.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" run.py llm-conversar
if errorlevel 1 (
  echo.
  echo Algo deu errado. Veja a mensagem acima.
)
pause
