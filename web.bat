@echo off
rem Abre a interface web do mini-LLM no navegador (calendario, grafico e boletim).
setlocal
cd /d "%~dp0"
chcp 65001 >nul

if not exist ".venv\instalado.ok" (
  echo.
  echo A instalacao ainda nao terminou ^(ou falhou^).
  echo.
  echo  1. Rode o instalar.bat
  echo  2. ESPERE aparecer a palavra "Pronto!" na janela dele
  echo  3. So depois abra este web.bat
  echo.
  pause
  exit /b 1
)

echo Abrindo a interface web... ^(para encerrar, feche esta janela^)
".venv\Scripts\python.exe" run.py web
if errorlevel 1 (
  echo.
  echo Algo deu errado. Veja a mensagem acima.
)
pause
