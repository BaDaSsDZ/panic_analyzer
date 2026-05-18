#!/bin/bash
# CASI AI - Full training pipeline (Mac/Linux)

set -e

echo "=== CASI AI Training Pipeline ==="

echo "[1/3] Extracting labeled data from database..."
python -m data.extract

echo "[2/3] Preprocessing and building training splits..."
python -m data.preprocess

echo "[3/3] Training the model..."
python -m training.train

echo ""
echo "=== Pipeline complete! ==="
echo "Model saved to: ./model/saved/"
echo ""
echo "To evaluate:    python -m training.evaluate"
echo "To start API:   uvicorn serving.main:app --host 0.0.0.0 --port 8001"
