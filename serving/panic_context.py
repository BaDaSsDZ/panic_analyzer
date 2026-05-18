"""
Fetches and assembles panic context from the casi-dashboard PostgreSQL DB.

Used by the predictor for both completed and active incident modes.
Active mode works on partial data — whatever exists at time of call.
"""

import os
import re
import logging
from dotenv import load_dotenv
import sqlalchemy as sa

load_dotenv()
log = logging.getLogger(__name__)

CREATION_METHOD_MAP = {
    0: "alerter", 1: "server", 2: "journey", 3: "dispatcher",
    4: "occurrence", 5: "extension", 100: "api",
    101: "api_site_dispatch", 102: "api_custom_dispatch",
    200: "personal_tracker",
}

OCCURRENCE_TYPE_MAP = {
    0: "meet", 1: "ride", 2: "test_panic",
    3: "iot_mandown", 4: "iot_outside_zone", 5: "posting",
}

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        url = sa.engine.URL.create(
            drivername="postgresql+psycopg2",
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            database=os.getenv("DB_NAME"),
            username=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )
        _engine = sa.create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
    return _engine


def _clean_comment(text):
    if not text or not text.strip():
        return None
    cleaned = re.sub(r'https?://\S+', '', text).strip()
    if not cleaned or re.match(r'^[.\s]+$', cleaned):
        return None
    return cleaned


def fetch_panic_context(panic_id: int) -> str:
    """
    Fetch all relevant data for a panic and assemble into a single input string.
    Works for both completed and active (in-progress) panics.
    Returns the assembled text string.
    """
    engine = get_engine()

    with engine.connect() as conn:
        # Core panic fields
        panic_row = conn.execute(
            sa.text("""
                SELECT creation_method, is_false_alarm, organisation_id, alerter_id
                FROM panics WHERE id = :id AND deleted_at IS NULL
            """),
            {"id": panic_id}
        ).fetchone()

        if not panic_row:
            raise ValueError(f"Panic {panic_id} not found")

        cm = CREATION_METHOD_MAP.get(panic_row.creation_method, str(panic_row.creation_method))
        false_alarm = "yes" if panic_row.is_false_alarm else "no"

        # Services dispatched
        services = conn.execute(
            sa.text("""
                SELECT s.name FROM panic_service ps
                JOIN services s ON s.id = ps.service_id
                WHERE ps.panic_id = :id AND ps.deleted_at IS NULL
            """),
            {"id": panic_id}
        ).fetchall()
        service_str = ",".join(r.name for r in services) if services else ""

        # Occurrence type (Bolt/ride context)
        occ = conn.execute(
            sa.text("""
                SELECT o.type, o.creation_method FROM occurrences o
                JOIN rides r ON r.occurrence_id = o.id
                WHERE r.panic_id = :id
                UNION
                SELECT o.type, o.creation_method FROM occurrences o
                JOIN meets m ON m.occurrence_id = o.id
                WHERE m.panic_id = :id
                LIMIT 1
            """),
            {"id": panic_id}
        ).fetchone()
        occ_str = ""
        if occ:
            occ_type = OCCURRENCE_TYPE_MAP.get(occ.type, str(occ.type))
            occ_str = f"occurrence_type={occ_type} occurrence_source={occ.creation_method}"

        meta = f"creation={cm} false_alarm={false_alarm} services={service_str} {occ_str}".strip()

        # Special procedures
        sp_rows = conn.execute(
            sa.text("""
                SELECT DISTINCT sp.name, sp.description, sp.critical
                FROM special_procedures sp
                WHERE sp.deleted_at IS NULL AND sp.id IN (
                    SELECT csp.special_procedure_id FROM cover_users cu
                    JOIN cover_special_procedure csp ON csp.cover_id = cu.cover_id
                    WHERE cu.user_id = :alerter_id
                    UNION
                    SELECT osp.special_procedure_id FROM organisation_special_procedure osp
                    WHERE osp.organisation_id = :org_id
                )
                ORDER BY sp.critical DESC, sp.name
            """),
            {"alerter_id": panic_row.alerter_id, "org_id": panic_row.organisation_id or 0}
        ).fetchall()

        sp_parts = []
        for sp in sp_rows:
            desc = re.sub(r'https?://\S+', '', sp.description or '').strip()
            sp_parts.append(f"{sp.name}: {desc}" if desc else sp.name)
        procedure_str = " | ".join(sp_parts)

        # Logs — signal types only, chronological, no noise
        log_rows = conn.execute(
            sa.text("""
                SELECT description FROM panic_logs
                WHERE panic_id = :id
                  AND description IS NOT NULL
                  AND (type IS NULL OR type NOT IN (32, 60))
                  AND description NOT LIKE '%acknowledged the panic for monitoring%'
                ORDER BY created_at ASC
            """),
            {"id": panic_id}
        ).fetchall()
        log_str = " | ".join(r.description.strip() for r in log_rows if r.description)

        # Comments — cleaned
        comment_rows = conn.execute(
            sa.text("""
                SELECT text FROM panic_comments
                WHERE panic_id = :id AND deleted_at IS NULL AND text IS NOT NULL
                ORDER BY created_at ASC
            """),
            {"id": panic_id}
        ).fetchall()
        cleaned_comments = [_clean_comment(r.text) for r in comment_rows]
        comment_str = " | ".join(c for c in cleaned_comments if c)

        # Dynamic form answers — completed forms only, text answers only
        form_rows = conn.execute(
            sa.text("""
                SELECT dfq.question_title AS question, dfa.answer_text AS answer
                FROM dynamic_form_responses dfr
                JOIN dynamic_form_answers dfa
                    ON dfa.dynamic_form_response_id = dfr.id AND dfa.deleted_at IS NULL
                JOIN dynamic_form_questions dfq ON dfq.id = dfa.dynamic_form_question_id
                WHERE dfr.panic_id = :id
                  AND dfr.form_completed = true
                  AND dfa.answer_text IS NOT NULL AND dfa.answer_text != ''
                  AND dfq.question_title NOT ILIKE '%image%'
                  AND dfq.question_title NOT ILIKE '%photo%'
                  AND dfq.question_title NOT ILIKE '%signature%'
                ORDER BY dfa.created_at ASC
            """),
            {"id": panic_id}
        ).fetchall()
        form_parts = [f"{r.question}: {r.answer}" for r in form_rows if r.answer]
        form_str = " | ".join(form_parts)

    # Assemble input text
    parts = []
    if meta:
        parts.append(f"[META] {meta}")
    if procedure_str:
        parts.append(f"[PROCEDURES] {procedure_str}")
    if log_str:
        parts.append(f"[LOGS] {log_str}")
    if comment_str:
        parts.append(f"[COMMENTS] {comment_str}")
    if form_str:
        parts.append(f"[FORM] {form_str}")

    return " ".join(parts)
