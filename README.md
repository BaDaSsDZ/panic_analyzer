# CASI AI — Panic Incident Tagger

Custom DistilBERT-based multi-label classifier that reads a CASI security incident and suggests the correct tags based on what happened.

## How it works

1. Reads panic data from the casi-dashboard PostgreSQL database
2. Assembles a structured text input: `[META] → [PROCEDURES] → [LOGS] → [COMMENTS] → [FORM]`
3. Runs through a fine-tuned DistilBERT classifier
4. Returns ranked tag suggestions with confidence scores

Supports both **completed incidents** (full data) and **active incidents** (partial data — whatever exists at time of call).

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env with your DB credentials
```

## Training

```bash
# Windows — run full pipeline in one step
scripts\run_pipeline.bat

# Mac/Linux
bash scripts/run_pipeline.sh

# Or step by step:
python -m data.extract        # Pull labeled panics from DB
python -m data.preprocess     # Build train/val/test splits
python -m training.train      # Fine-tune DistilBERT
python -m training.evaluate   # Per-tag F1 report on test set
```

## Running the API

```bash
# Windows
scripts\start_server.bat

# Mac/Linux
uvicorn serving.main:app --host 0.0.0.0 --port 8001
```

API docs at: `http://localhost:8001/docs`

## API Usage

### Tag a completed panic
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret" \
  -d '{"panic_id": 286965, "mode": "completed"}'
```

### Tag an active (in-progress) incident
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret" \
  -d '{"panic_id": 286965, "mode": "active"}'
```

### Submit feedback (human corrections)
```bash
curl -X POST http://localhost:8001/feedback \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret" \
  -d '{
    "panic_id": 286965,
    "accepted_tag_ids": ["uuid-1"],
    "rejected_tag_ids": ["uuid-2"],
    "added_tag_ids": ["uuid-3"],
    "corrected_by_user_id": 150240
  }'
```

## Project structure

```
casi-ai/
├── data/
│   ├── extract.py        # Pull labeled data from DB
│   ├── preprocess.py     # Build training splits
│   └── dataset.py        # PyTorch Dataset
├── model/
│   └── classifier.py     # DistilBERT + classification head
├── training/
│   ├── train.py          # Fine-tuning loop
│   └── evaluate.py       # Per-tag evaluation report
├── serving/
│   ├── main.py           # FastAPI server
│   ├── predictor.py      # Model inference singleton
│   ├── panic_context.py  # DB fetcher + text assembler
│   └── schemas.py        # Request/response models
├── feedback/
│   └── collector.py      # Record human corrections
├── scripts/
│   ├── run_pipeline.bat  # Windows: extract + train
│   ├── run_pipeline.sh   # Mac/Linux: extract + train
│   └── start_server.bat  # Windows: start API
└── docs/
    └── panic-analysis.md # Real panic data analysis
```

## Retraining

Run the pipeline again after collecting more tagged panics or feedback corrections:
```bash
scripts\run_pipeline.bat
```

The model improves with each cycle as more labeled data accumulates.

## Integration with casi-dashboard (Laravel)

The Laravel app calls this service via:
- `POST /predict` after panic status → COMPLETE (via a Laravel Job)
- `POST /predict` on-demand when operator clicks "Re-run AI" button  
- `POST /feedback` when operator accepts/rejects/edits AI suggestions
- Tags are saved to `panic_tags` with `is_ai_suggested=true` and `ai_confidence` score
