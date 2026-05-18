@echo off
REM Start the CASI AI API server on Windows
echo Starting CASI AI server on port 8001...
python -m uvicorn serving.main:app --host 0.0.0.0 --port 8001
