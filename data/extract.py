"""
Extract labeled panic data from the casi-dashboard PostgreSQL database.

Pulls completed panics with at least one human-applied tag. Joins logs,
comments, services, dynamic form answers, special procedures, and occurrence
data into one row per panic. Saves labeled_panics.csv and tags.csv.

Run: python -m data.extract
"""

import os
import sys
import logging
import re
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import sqlalchemy as sa

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.getenv("DATA_DIR", "./data/output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Log types that carry semantic signal — confirmed from panic analysis
SIGNAL_LOG_TYPES = {
    17, 4, 38, 2, 19, 6, 7, 21, 18, 15, 25, 26, 27, 31, 39, 66, 74, 20,
    # null-type logs are excluded by the query
}

# Log types that are pure noise — exclude entirely
NOISE_LOG_TYPES = {32, 60}

CREATION_METHOD_MAP = {
    0: "alerter",
    1: "server",
    2: "journey",
    3: "dispatcher",
    4: "occurrence",
    5: "extension",
    100: "api",
    101: "api_site_dispatch",
    102: "api_custom_dispatch",
    200: "personal_tracker",
}

OCCURRENCE_TYPE_MAP = {
    0: "meet",
    1: "ride",
    2: "test_panic",
    3: "iot_mandown",
    4: "iot_outside_zone",
    5: "posting",
}


def get_engine():
    url = sa.engine.URL.create(
        drivername="postgresql+psycopg2",
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME"),
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )
    return sa.create_engine(url, pool_pre_ping=True)


def fetch_panics_auto(engine, months: int = 8):
    """
    Pull completed panics from the last N months that show tagging quality signals.

    Quality signals (proxy for "operator was paying attention when they tagged"):
      - Tags exist and none were soft-deleted (operator didn't correct themselves)
      - Panic is completed (status 3 or 5)
      - Has at least one comment OR dynamic form answer (operator engaged enough to write)

    This is NOT filtering by what happened in the panic — it's filtering by how
    carefully the operator handled the tagging. High-volume panics tagged then
    immediately corrected are excluded; panics with thoughtful operator notes are included.
    """
    query = sa.text(f"""
        SELECT DISTINCT
            p.id                        AS panic_id,
            p.creation_method,
            p.status,
            p.panic_type,
            p.tier,
            p.is_false_alarm,
            COALESCE(p.comment_alerter, '') AS comment_alerter,
            COALESCE(p.comment_responder, '') AS comment_responder,
            p.created_at,
            p.updated_at
        FROM panics p
        -- Must have at least one active (non-deleted) tag
        WHERE EXISTS (
            SELECT 1 FROM panic_tags pt
            WHERE pt.panic_id = p.id AND pt.deleted_at IS NULL
        )
        -- No corrected/soft-deleted tags — operator was confident in their tagging
        AND NOT EXISTS (
            SELECT 1 FROM panic_tags pt
            WHERE pt.panic_id = p.id AND pt.deleted_at IS NOT NULL
        )
        -- Operator left a comment OR filled in a form (shows engagement)
        AND (
            EXISTS (
                SELECT 1 FROM panic_comments pc
                WHERE pc.panic_id = p.id AND pc.deleted_at IS NULL AND pc.text IS NOT NULL
            )
            OR EXISTS (
                SELECT 1 FROM dynamic_form_responses dfr
                JOIN dynamic_form_answers dfa ON dfa.dynamic_form_response_id = dfr.id
                WHERE dfr.panic_id = p.id AND dfr.form_completed = true
                  AND dfa.answer_text IS NOT NULL AND dfa.answer_text != ''
                  AND dfa.deleted_at IS NULL
            )
        )
        AND p.deleted_at IS NULL
        AND p.status IN (3, 5)
        AND p.created_at >= NOW() - INTERVAL '{months} months'
        ORDER BY p.id
    """)
    log.info("Fetching completed panics from last %d months with quality signals...", months)
    df = pd.read_sql(query, engine)
    log.info("Auto-pulled %d training panics", len(df))
    return df


def fetch_panics_by_ids(engine, curated_ids: list):
    """Fetch specific panic IDs (from curated_panic_ids.txt override list)."""
    ids_str = ",".join(str(i) for i in curated_ids)
    query = sa.text(f"""
        SELECT
            p.id                        AS panic_id,
            p.creation_method,
            p.status,
            p.panic_type,
            p.tier,
            p.is_false_alarm,
            COALESCE(p.comment_alerter, '') AS comment_alerter,
            COALESCE(p.comment_responder, '') AS comment_responder,
            p.created_at,
            p.updated_at
        FROM panics p
        WHERE
            p.id IN ({ids_str})
            AND p.deleted_at IS NULL
            AND p.status IN (3, 5)
        ORDER BY p.id
    """)
    df = pd.read_sql(query, engine)
    missing = set(curated_ids) - set(df["panic_id"].tolist())
    if missing:
        log.warning(
            "%d curated IDs not found or not complete in DB: %s",
            len(missing), sorted(missing)
        )
    return df


def fetch_panics(engine, curated_ids: list):
    """
    Combine auto-pulled panics (last 8 months, quality-filtered) with any
    manually curated IDs from curated_panic_ids.txt.

    Auto pull is the primary source of volume. Curated IDs let you force-include
    specific panics that are known-good training examples regardless of the
    quality heuristics (e.g., panics where a tag was corrected but the final
    state is verified correct by a supervisor).
    """
    months = int(os.getenv("TRAINING_MONTHS", 8))
    auto_df = fetch_panics_auto(engine, months)

    extra_df = pd.DataFrame()
    if curated_ids:
        log.info("Also fetching %d explicitly curated panic IDs...", len(curated_ids))
        extra_df = fetch_panics_by_ids(engine, curated_ids)

    if len(extra_df) > 0:
        combined = pd.concat([auto_df, extra_df]).drop_duplicates(subset="panic_id")
    else:
        combined = auto_df

    log.info(
        "Total training set: %d panics (%d auto + %d curated-only)",
        len(combined),
        len(auto_df),
        len(combined) - len(auto_df),
    )
    return combined


def fetch_ordered_logs(engine, panic_ids):
    """Fetch signal logs in chronological order, excluding noise types."""
    ids_str = ",".join(str(i) for i in panic_ids)
    query = sa.text(f"""
        SELECT
            panic_id,
            type,
            description,
            created_at
        FROM panic_logs
        WHERE
            panic_id IN ({ids_str})
            AND description IS NOT NULL
            AND (type IS NULL OR type NOT IN (32, 60))
        ORDER BY panic_id, created_at ASC
    """)
    return pd.read_sql(query, engine)


def fetch_comments(engine, panic_ids):
    ids_str = ",".join(str(i) for i in panic_ids)
    query = sa.text(f"""
        SELECT panic_id, text, created_at
        FROM panic_comments
        WHERE
            panic_id IN ({ids_str})
            AND deleted_at IS NULL
            AND text IS NOT NULL
        ORDER BY panic_id, created_at ASC
    """)
    return pd.read_sql(query, engine)


def fetch_services(engine, panic_ids):
    ids_str = ",".join(str(i) for i in panic_ids)
    query = sa.text(f"""
        SELECT
            ps.panic_id,
            s.name AS service_name,
            ps.diagnosis,
            ps.location_description
        FROM panic_service ps
        JOIN services s ON s.id = ps.service_id
        WHERE
            ps.panic_id IN ({ids_str})
            AND ps.deleted_at IS NULL
        ORDER BY ps.panic_id, ps.created_at ASC
    """)
    return pd.read_sql(query, engine)


def fetch_form_answers(engine, panic_ids):
    """Fetch only completed forms with non-null text answers."""
    ids_str = ",".join(str(i) for i in panic_ids)
    query = sa.text(f"""
        SELECT
            dfr.panic_id,
            dfq.question_title AS question,
            dfa.answer_text    AS answer
        FROM dynamic_form_responses dfr
        JOIN dynamic_form_answers dfa
            ON dfa.dynamic_form_response_id = dfr.id
            AND dfa.deleted_at IS NULL
        JOIN dynamic_form_questions dfq
            ON dfq.id = dfa.dynamic_form_question_id
        WHERE
            dfr.panic_id IN ({ids_str})
            AND dfr.form_completed = true
            AND dfa.answer_text IS NOT NULL
            AND dfa.answer_text != ''
            AND dfq.question_title NOT ILIKE '%image%'
            AND dfq.question_title NOT ILIKE '%photo%'
            AND dfq.question_title NOT ILIKE '%signature%'
        ORDER BY dfr.panic_id, dfa.created_at ASC
    """)
    return pd.read_sql(query, engine)


def fetch_special_procedures(engine, panic_ids):
    """
    Fetch special procedures linked to each panic via alerter's cover,
    alerter's organisation, or the panic's own organisation.
    """
    ids_str = ",".join(str(i) for i in panic_ids)
    query = sa.text(f"""
        SELECT DISTINCT
            p.id AS panic_id,
            sp.name,
            sp.description,
            sp.critical
        FROM panics p
        JOIN (
            -- Via alerter's cover
            SELECT cu.user_id, csp.special_procedure_id
            FROM cover_users cu
            JOIN cover_special_procedure csp ON csp.cover_id = cu.cover_id

            UNION

            -- Via alerter's organisation
            SELECT u.id AS user_id, osp.special_procedure_id
            FROM users u
            JOIN organisation_special_procedure osp ON osp.organisation_id = u.organisation_id
            WHERE u.organisation_id IS NOT NULL

            UNION

            -- Via panic's own organisation
            SELECT p2.alerter_id AS user_id, osp.special_procedure_id
            FROM panics p2
            JOIN organisation_special_procedure osp ON osp.organisation_id = p2.organisation_id
            WHERE p2.id IN ({ids_str})
        ) sp_map ON sp_map.user_id = p.alerter_id
        JOIN special_procedures sp ON sp.id = sp_map.special_procedure_id
        WHERE
            p.id IN ({ids_str})
            AND sp.deleted_at IS NULL
        ORDER BY p.id, sp.critical DESC, sp.name
    """)
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        log.warning("Could not fetch special procedures: %s", e)
        return pd.DataFrame(columns=["panic_id", "name", "description", "critical"])


def fetch_occurrences(engine, panic_ids):
    """Fetch occurrence type for panics created via rides/meets."""
    ids_str = ",".join(str(i) for i in panic_ids)
    query = sa.text(f"""
        SELECT p.id AS panic_id, o.type AS occurrence_type, o.creation_method AS occurrence_creation_method
        FROM panics p
        JOIN rides r ON r.panic_id = p.id
        JOIN occurrences o ON o.id = r.occurrence_id
        WHERE p.id IN ({ids_str})

        UNION

        SELECT p.id AS panic_id, o.type AS occurrence_type, o.creation_method AS occurrence_creation_method
        FROM panics p
        JOIN meets m ON m.panic_id = p.id
        JOIN occurrences o ON o.id = m.occurrence_id
        WHERE p.id IN ({ids_str})
    """)
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        log.warning("Could not fetch occurrences: %s", e)
        return pd.DataFrame(columns=["panic_id", "occurrence_type", "occurrence_creation_method"])


def fetch_tags(engine, panic_ids):
    ids_str = ",".join(str(i) for i in panic_ids)
    query = sa.text(f"""
        SELECT pt.panic_id, pt.tag_id::text AS tag_id
        FROM panic_tags pt
        WHERE pt.panic_id IN ({ids_str}) AND pt.deleted_at IS NULL
        ORDER BY pt.panic_id, pt.created_at ASC
    """)
    return pd.read_sql(query, engine)


def fetch_all_tags(engine):
    query = sa.text("""
        SELECT id::text AS tag_id, name, controller_advice, tag_weight
        FROM tags
        WHERE deleted_at IS NULL
        ORDER BY tag_weight DESC NULLS LAST, name
    """)
    df = pd.read_sql(query, engine)
    log.info("Found %d active tags", len(df))
    return df


def clean_comment(text):
    """Remove URLs, dot-only, and empty comments."""
    if not text or not text.strip():
        return None
    cleaned = re.sub(r'https?://\S+', '', text).strip()
    if not cleaned or re.match(r'^[.\s]+$', cleaned):
        return None
    return cleaned


def aggregate_logs(logs_df, panic_ids):
    result = {}
    for pid in panic_ids:
        rows = logs_df[logs_df["panic_id"] == pid]
        # Exclude generic null-type admin ack lines (contain "acknowledged the panic for monitoring")
        descriptions = []
        for _, row in rows.iterrows():
            desc = row["description"]
            if desc and "acknowledged the panic for monitoring" in desc:
                continue
            if desc:
                descriptions.append(desc.strip())
        result[pid] = " | ".join(descriptions)
    return result


def aggregate_comments(comments_df, panic_ids):
    result = {}
    for pid in panic_ids:
        rows = comments_df[comments_df["panic_id"] == pid]
        cleaned = [clean_comment(r["text"]) for _, r in rows.iterrows()]
        cleaned = [c for c in cleaned if c]
        result[pid] = " | ".join(cleaned)
    return result


def aggregate_services(services_df, panic_ids):
    result = {}
    for pid in panic_ids:
        rows = services_df[services_df["panic_id"] == pid]
        names = rows["service_name"].dropna().unique().tolist()
        result[pid] = ",".join(names)
    return result


def aggregate_form_answers(forms_df, panic_ids):
    result = {}
    for pid in panic_ids:
        rows = forms_df[forms_df["panic_id"] == pid]
        parts = []
        for _, row in rows.iterrows():
            q = (row["question"] or "").strip()
            a = (row["answer"] or "").strip()
            if a:
                parts.append(f"{q}: {a}")
        result[pid] = " | ".join(parts)
    return result


def aggregate_special_procedures(sp_df, panic_ids):
    result = {}
    for pid in panic_ids:
        rows = sp_df[sp_df["panic_id"] == pid]
        parts = []
        for _, row in rows.iterrows():
            name = (row["name"] or "").strip()
            desc = (row["description"] or "").strip()
            # Strip URLs from procedure descriptions
            desc = re.sub(r'https?://\S+', '', desc).strip()
            if name:
                parts.append(f"{name}: {desc}" if desc else name)
        result[pid] = " | ".join(parts)
    return result


def aggregate_tags(tags_df, panic_ids):
    result = {}
    for pid in panic_ids:
        rows = tags_df[tags_df["panic_id"] == pid]
        result[pid] = ",".join(rows["tag_id"].tolist())
    return result


def load_curated_ids(path: Path) -> list:
    if not path.exists():
        log.info("No curated_panic_ids.txt found — using auto-pull only.")
        return []
    ids = []
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            try:
                ids.append(int(line))
            except ValueError:
                log.warning("Skipping non-integer line in curated_panic_ids.txt: %r", line)
    if ids:
        log.info("Loaded %d curated panic IDs from %s (will be merged with auto-pull)", len(ids), path)
    return ids


def main():
    curated_ids_path = Path(__file__).parent / "curated_panic_ids.txt"
    curated_ids = load_curated_ids(curated_ids_path)

    engine = get_engine()
    try:
        panics_df = fetch_panics(engine, curated_ids)
        if len(panics_df) == 0:
            log.error("No panics found for curated IDs. Check DB connection and status filters.")
            sys.exit(1)

        panic_ids = panics_df["panic_id"].tolist()
        log.info("Fetching related data for %d panics...", len(panic_ids))

        # Fetch all related data
        logs_df       = fetch_ordered_logs(engine, panic_ids)
        comments_df   = fetch_comments(engine, panic_ids)
        services_df   = fetch_services(engine, panic_ids)
        forms_df      = fetch_form_answers(engine, panic_ids)
        sp_df         = fetch_special_procedures(engine, panic_ids)
        occurrences_df = fetch_occurrences(engine, panic_ids)
        tags_df       = fetch_tags(engine, panic_ids)
        all_tags_df   = fetch_all_tags(engine)

        log.info("Aggregating per-panic text...")
        log_texts     = aggregate_logs(logs_df, panic_ids)
        comment_texts = aggregate_comments(comments_df, panic_ids)
        service_texts = aggregate_services(services_df, panic_ids)
        form_texts    = aggregate_form_answers(forms_df, panic_ids)
        sp_texts      = aggregate_special_procedures(sp_df, panic_ids)
        tag_map       = aggregate_tags(tags_df, panic_ids)

        # Build occurrence lookup
        occ_lookup = {}
        for _, row in occurrences_df.iterrows():
            occ_lookup[row["panic_id"]] = {
                "type": OCCURRENCE_TYPE_MAP.get(row["occurrence_type"], str(row["occurrence_type"])),
                "method": row["occurrence_creation_method"],
            }

        rows = []
        active_tag_ids = set(all_tags_df["tag_id"].tolist())

        for _, panic in panics_df.iterrows():
            pid = panic["panic_id"]

            tag_ids_raw = tag_map.get(pid, "")
            tag_ids = [t for t in tag_ids_raw.split(",") if t and t in active_tag_ids]
            if not tag_ids:
                continue  # skip if all applied tags are since deleted

            cm = CREATION_METHOD_MAP.get(panic["creation_method"], str(panic["creation_method"]))
            false_alarm = "yes" if panic["is_false_alarm"] else "no"
            occ = occ_lookup.get(pid)
            occ_str = f"occurrence_type={occ['type']} occurrence_source={occ['method']}" if occ else ""

            meta = f"creation={cm} false_alarm={false_alarm} services={service_texts.get(pid, '')} {occ_str}".strip()

            rows.append({
                "panic_id":          pid,
                "creation_method":   panic["creation_method"],
                "is_false_alarm":    panic["is_false_alarm"],
                "meta_text":         meta,
                "log_text":          log_texts.get(pid, ""),
                "comment_text":      comment_texts.get(pid, ""),
                "form_text":         form_texts.get(pid, ""),
                "procedure_text":    sp_texts.get(pid, ""),
                "tag_ids":           ",".join(tag_ids),
                "tag_count":         len(tag_ids),
            })

        df = pd.DataFrame(rows)
        log.info("Final dataset: %d labeled panics", len(df))

        # Save outputs
        out_path = OUTPUT_DIR / "labeled_panics.csv"
        df.to_csv(out_path, index=False)
        log.info("Saved → %s", out_path)

        tags_path = OUTPUT_DIR / "tags.csv"
        all_tags_df.to_csv(tags_path, index=False)
        log.info("Saved → %s", tags_path)

        # Tag distribution report
        from collections import Counter
        all_applied = [t for row in df["tag_ids"] for t in row.split(",") if t]
        counts = Counter(all_applied)
        name_map = dict(zip(all_tags_df["tag_id"], all_tags_df["name"]))

        log.info("\n--- Tag distribution ---")
        for tag_id, count in counts.most_common():
            log.info("  %-40s  %d", name_map.get(tag_id, tag_id), count)

        rare = [(name_map.get(t, t), c) for t, c in counts.items() if c < 15]
        if rare:
            log.warning("\n--- Tags with <15 examples (will train poorly) ---")
            for name, count in rare:
                log.warning("  %-40s  %d", name, count)

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
