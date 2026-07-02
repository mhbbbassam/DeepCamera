@echo off
setlocal

echo Deploying icebag-counter skill on Windows...
if not exist .venv (
  python -m venv .venv
)
.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements_cuda.txt

REM Try to make TRT engine if trtexec available
set MODEL_ONNX=models\icebag_yolo26.onnx
set ENGINE_OUT=%MODEL_ONNX:.onnx=.trt%
where trtexec >nul 2>&1
if %ERRORLEVEL%==0 (
  echo Attempting trtexec build...
  trtexec --onnx="%MODEL_ONNX%" --saveEngine="%ENGINE_OUT%" --explicitBatch --fp16 || echo trtexec failed
)

echo Deploy complete.
endlocal
