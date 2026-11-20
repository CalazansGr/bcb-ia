@echo off
rem Atualiza os dados do Banco Central e faz o mini-LLM escrever sobre o dia mais recente.
setlocal
cd /d "%~dp0"
chcp 65001 >nul

if not exist ".venv\Scripts\python.exe" (
  echo Rode instalar.bat primeiro.
  pause
  exit /b 1
)

echo === 1/3 Baixando dados atualizados do Banco Central ===
".venv\Scripts\python.exe" run.py atualizar
if errorlevel 1 goto erro

if not exist "modelos\minillm.pt" (
  echo.
  echo === 2/3 Treinando o mini-LLM ^(so na primeira vez^) ===
  ".venv\Scripts\python.exe" run.py llm-treinar
  if errorlevel 1 goto erro
) else (
  echo.
  echo === 2/3 Modelo ja treinado: pulando o treino ===
)

echo.
echo === 3/3 O mini-LLM escreve sobre o dia mais recente ===
".venv\Scripts\python.exe" run.py llm-escrever
if errorlevel 1 goto erro

pause
exit /b 0

:erro
echo.
echo Algo deu errado. Veja a mensagem acima.
pause
exit /b 1
