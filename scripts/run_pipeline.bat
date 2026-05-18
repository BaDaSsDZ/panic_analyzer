@echo off
REM CASI AI - Full training pipeline (Windows)
REM Run this once to extract data, preprocess, and train the model.

echo === CASI AI Training Pipeline ===
echo.

echo [1/3] Extracting labeled data from database...
python -m data.extract
if errorlevel 1 goto error

echo.
echo [2/3] Preprocessing and building training splits...
python -m data.preprocess
if errorlevel 1 goto error

echo.
echo [3/3] Training the model...
python -m training.train
if errorlevel 1 goto error

echo.
echo === Pipeline complete! ===
echo Model saved to: .\model\saved\
echo.
echo To evaluate on test set:  python -m training.evaluate
echo To start API server:      python -m uvicorn serving.main:app --host 0.0.0.0 --port 8001
goto end

:error
echo.
echo === ERROR: Pipeline failed at step above ===
exit /b 1

:end
