@echo off
rem Instala tudo para rodar o bcb-ia. Basta dar dois cliques neste arquivo.
setlocal
cd /d "%~dp0"
echo ==========================================
echo   Instalando o bcb-ia (leva alguns minutos)
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

echo Instalando bibliotecas...
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
if errorlevel 1 goto erro

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo Sem placa NVIDIA: instalando o PyTorch para CPU ^(o treino fica mais lento^)...
  ".venv\Scripts\python.exe" -m pip install --quiet torch
) else (
  echo Placa NVIDIA encontrada: instalando o PyTorch com GPU ^(download grande, ~2,5 GB^)...
  ".venv\Scripts\python.exe" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
)
if errorlevel 1 goto erro

echo.
echo Pronto! Agora de dois cliques em rodar.bat
pause
exit /b 0

:erro
echo.
echo Algo deu errado na instalacao. Dica: se o caminho desta pasta for muito longo,
echo mova a pasta para algo curto, como C:\projetos\bcb-ia, e tente de novo.
pause
exit /b 1
