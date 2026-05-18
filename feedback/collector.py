"""
Records human corrections to AI tag suggestions.

Every accept/reject/add is a labeled training example.
Stored as JSONL — one line per feedback event.
Accumulates until next retraining cycle.
"""

import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

FEEDBACK_DIR = Path(os.getenv("DATA_DIR", "./data/output")) / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_FILE = FEEDBACK_DIR / "corrections.jsonl"


def record_feedback(data: dict):
    """
    Append a feedback record to corrections.jsonl.

    Expected keys:
      panic_id, accepted_tag_ids, rejected_tag_ids, added_tag_ids, corrected_by_user_id
    """
    record = {
        "recorded_at":          datetime.now(timezone.utc).isoformat(),
        "panic_id":             data.get("panic_id"),
        "accepted_tag_ids":     data.get("accepted_tag_ids", []),
        "rejected_tag_ids":     data.get("rejected_tag_ids", []),
        "added_tag_ids":        data.get("added_tag_ids", []),
        "corrected_by_user_id": data.get("corrected_by_user_id"),
    }

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    log.info(
        "Feedback recorded: panic=%s accepted=%d rejected=%d added=%d",
        record["panic_id"],
        len(record["accepted_tag_ids"]),
        len(record["rejected_tag_ids"]),
        len(record["added_tag_ids"]),
    )


def load_feedback():
    """Load all feedback records. Used during retraining to incorporate corrections."""
    if not FEEDBACK_FILE.exists():
        return []
    records = []
    with open(FEEDBACK_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def feedback_stats():
    records = load_feedback()
    return {
        "total_corrections": len(records),
        "panics_corrected":  len(set(r["panic_id"] for r in records)),
        "tags_added":        sum(len(r["added_tag_ids"]) for r in records),
        "tags_rejected":     sum(len(r["rejected_tag_ids"]) for r in records),
    }
