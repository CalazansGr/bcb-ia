@echo off
rem Instala tudo para rodar o bcb-ia. Basta dar dois cliques neste arquivo.
setlocal
cd /d "%~dp0"
echo ==========================================
echo   Instalando o bcb-ia (leva alguns minutos)
echo ==========================================
echo   NAO FECHE esta janela e NAO abra o rodar.bat
echo   ate aparecer a palavra "Pronto!" no final.
echo ==========================================

where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo Python nao encontrado. Instale em https://www.python.org/downloads/
  echo e marque a opcao "Add python.exe to PATH". Depois rode este arquivo de novo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Criando ambiente isolado...
  python -m venv .venv
  if errorlevel 1 goto erro
)
if exist ".venv\instalado.ok" del ".venv\instalado.ok"

echo Instalando bibliotecas...
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
if errorlevel 1 goto erro

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo Sem placa NVIDIA: instalando o PyTorch para CPU ^(o treino fica mais lento^)...
  ".venv\Scripts\python.exe" -m pip install --quiet torch
) else (
  echo Placa NVIDIA encontrada: instalando o PyTorch com GPU ^(download grande, ~2,5 GB^)...
  echo Esta etapa demora e a janela fica parada sem barra de progresso. E normal.
  ".venv\Scripts\python.exe" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
)
if errorlevel 1 goto erro

rem marca que a instalacao terminou de verdade (o rodar.bat confere este arquivo)
echo ok> ".venv\instalado.ok"

echo.
echo ==========================================
echo   Pronto! Agora de dois cliques em rodar.bat
echo ==========================================
pause
exit /b 0

:erro
echo.
echo Algo deu errado na instalacao. Dica: se o caminho desta pasta for muito longo,
echo mova a pasta para algo curto, como C:\projetos\bcb-ia, e tente de novo.
pause
exit /b 1
