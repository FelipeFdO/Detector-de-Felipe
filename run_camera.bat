@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   Identificador de Felipe - Inicializacao
echo ============================================
echo.

set "VENV_PY=.venv\Scripts\python.exe"

if exist "%VENV_PY%" goto :CHECK_LIBS

echo [1/4] Ambiente virtual (.venv) nao encontrado. Verificando Python...
call :FIND_PYTHON
if defined PYTHON_CMD goto :CREATE_VENV

echo Python 3 nao encontrado. Tentando instalar Python 3.12 automaticamente...
where winget >nul 2>&1
if %errorlevel%==0 (
    echo Instalando via winget...
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements --silent
)

call :FIND_PYTHON
if defined PYTHON_CMD goto :CREATE_VENV

echo Winget indisponivel ou falhou. Baixando instalador oficial do Python 3.12...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile '%TEMP%\python-3.12.10-amd64.exe'"
if exist "%TEMP%\python-3.12.10-amd64.exe" (
    echo Executando instalador silencioso do Python...
    "%TEMP%\python-3.12.10-amd64.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
    del "%TEMP%\python-3.12.10-amd64.exe" >nul 2>&1
)

call :FIND_PYTHON
if not defined PYTHON_CMD (
    echo.
    echo [ERRO] Nao foi possivel instalar o Python automaticamente.
    echo Instale manualmente em https://www.python.org/downloads/ marcando "Add Python to PATH"
    echo e execute este arquivo novamente.
    pause
    exit /b 1
)

:CREATE_VENV
echo Usando Python: %PYTHON_CMD%
echo [2/4] Criando ambiente virtual (.venv)...
%PYTHON_CMD% -m venv .venv
if errorlevel 1 (
    echo [ERRO] Falha ao criar o ambiente virtual.
    pause
    exit /b 1
)

:CHECK_LIBS
echo [3/4] Verificando bibliotecas necessarias...
"%VENV_PY%" -c "import cv2, sklearn, PIL, joblib, numpy, tqdm" >nul 2>&1
if errorlevel 1 (
    echo Instalando/atualizando dependencias...
    "%VENV_PY%" -m pip install --upgrade pip
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERRO] Falha ao instalar as bibliotecas. Verifique sua conexao com a internet.
        pause
        exit /b 1
    )
) else (
    echo Bibliotecas OK!
)

echo [4/4] Verificando modelos de IA...
if not exist "models" mkdir "models"

if not exist "models\face_detection_yunet_2023mar.onnx" (
    echo Baixando modelo YuNet...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx' -OutFile 'models\face_detection_yunet_2023mar.onnx'"
)

if not exist "models\face_recognition_sface_2021dec.onnx" (
    echo Baixando modelo SFace...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx' -OutFile 'models\face_recognition_sface_2021dec.onnx'"
)

if not exist "models\felipe_classifier.joblib" (
    echo Classificador treinado nao encontrado. Executando train.py...
    "%VENV_PY%" train.py
    if errorlevel 1 (
        echo [ERRO] Falha no treinamento.
        pause
        exit /b 1
    )
)

echo.
echo Iniciando camera...
"%VENV_PY%" webcam.py %*
if errorlevel 1 (
    echo.
    echo Ocorreu um erro ao executar o identificador.
    pause
    exit /b 1
)
exit /b 0

:FIND_PYTHON
set "PYTHON_CMD="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
    exit /b 0
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON_CMD=python"
    exit /b 0
)
for %%P in (
    "%LocalAppData%\Programs\Python\Python313\python.exe"
    "%LocalAppData%\Programs\Python\Python312\python.exe"
    "%LocalAppData%\Programs\Python\Python311\python.exe"
    "%LocalAppData%\Programs\Python\Python310\python.exe"
    "C:\Program Files\Python312\python.exe"
    "C:\Program Files\Python311\python.exe"
) do (
    if exist %%P (
        set "PYTHON_CMD=%%~P"
        exit /b 0
    )
)
exit /b 0
