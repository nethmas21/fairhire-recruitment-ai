"""
FairHire - Shared Database Module
------------------------------------
Every agent (Reader, Matcher, Explainer, Dashboard, Notification, JD
Fairness) imports and uses THIS module to read/write data - nobody writes
raw SQL directly in their own agent file. This keeps the schema consistent
no matter who on the team is coding which agent.

Setup:
    No installation needed - sqlite3 is built into Python.

Usage example (from any agent):
    from database.db import init_db, save_job, save_candidate, save_ranking

    init_db()  # run once, creates tables if they don't exist
    job_id = save_job("Data Analyst", "Looking for...")
    save_candidate("CAND_001", job_id, skills=["python","sql"])
"""

import sqlite3
import json
import os
import warnings
from contextlib import contextmanager
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "fairhire.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

# ---------------------------------------------------------------
# Encryption at rest for the PII vault
# ---------------------------------------------------------------
# Security feature: name/email/phone are encrypted before being written to
# disk, so a stolen/leaked copy of fairhire.db does not expose candidate
# PII in plaintext - only the running application, with the correct key,
# can read it back.
#
# Setup (.env):
#     FAIRHIRE_ENCRYPTION_KEY=<a Fernet key>
#
# Generate one with:
#     python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
#
# If no key is configured, PII is stored in PLAINTEXT and a warning is
# printed at import time - this keeps local development/testing working
# without requiring setup, but should never be left this way for anything
# beyond a demo.
_ENCRYPTION_KEY = os.environ.get("FAIRHIRE_ENCRYPTION_KEY", "")
if _ENCRYPTION_KEY:
    _fernet = Fernet(_ENCRYPTION_KEY.encode())
else:
    _fernet = None
    warnings.warn(
        "FAIRHIRE_ENCRYPTION_KEY is not set - PII (name/email/phone) will "
        "be stored in PLAINTEXT in fairhire.db. Set FAIRHIRE_ENCRYPTION_KEY "
        "in your .env before treating this as production-ready.",
        stacklevel=2,
    )


def _encrypt_field(value):
    if value is None or _fernet is None:
        return value
    return _fernet.encrypt(value.encode()).decode()


def _decrypt_field(value):
    if value is None or _fernet is None:
        return value
    try:
        return _fernet.decrypt(value.encode()).decode()
    except InvalidToken:
        # Value was stored before encryption was enabled (legacy plaintext
        # row), or the wrong key is configured - fail safe by returning the
        # raw stored value rather than crashing the whole request.
        return value


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Run once at startup - creates all tables if they don't already exist."""
    with get_connection() as conn:
        with open(SCHEMA_PATH, "r") as f:
            conn.executescript(f.read())
        _migrate_jobs_table(conn)
        _migrate_candidates_table(conn)


def _migrate_candidates_table(conn):
    """
    Adds 'cv_text_anonymized', 'projects_text', and 'cv_summary' to the
    candidates table if they don't already exist - same safe-migration
    pattern as _migrate_jobs_table below, so this works on a database
    created before these columns existed. cv_text_anonymized stores the
    full PII-stripped CV text; projects_text stores just the extracted
    "Projects" section; cv_summary stores a short recruiter-friendly
    summary (the NLP Summarization feature - see
    reader_agent.py's generate_cv_summary()).
    """
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(candidates)").fetchall()}
    if "cv_text_anonymized" not in existing_cols:
        conn.execute("ALTER TABLE candidates ADD COLUMN cv_text_anonymized TEXT")
    if "projects_text" not in existing_cols:
        conn.execute("ALTER TABLE candidates ADD COLUMN projects_text TEXT")
    if "cv_summary" not in existing_cols:
        conn.execute("ALTER TABLE candidates ADD COLUMN cv_summary TEXT")


def _migrate_jobs_table(conn):
    """
    Adds 'status' and 'closed_at' to the jobs table if they don't already
    exist, without requiring a change to schema.sql itself - this keeps
    init_db() safe to call on a database that was created before these
    columns existed (SQLite's ALTER TABLE ADD COLUMN is safe to run once;
    the existing-columns check keeps it safe to run every startup too).
    'status' defaults to 'open'; a job becomes 'closed' via close_job()
    below, which is also the point at which "delete on close" candidates
    actually get their PII erased.
    """
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "status" not in existing_cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN status TEXT DEFAULT 'open'")
    if "closed_at" not in existing_cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN closed_at TIMESTAMP")


# ---------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------
def save_job(title, description, fairness_check_flags=None):
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (title, description, fairness_check_flags) VALUES (?, ?, ?)",
            (title, description, json.dumps(fairness_check_flags or [])),
        )
        return cur.lastrowid


def get_job(job_id):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------
# Candidates (anonymized data only)
# ---------------------------------------------------------------
def save_candidate(candidate_id, job_id, skills=None, organizations=None, proxy_fields=None,
                    cv_text_anonymized=None, projects_text=None, cv_summary=None):
    """
    IMPORTANT: this is called from more than one place in the pipeline -
    the Reader Agent calls it first (with the full CV-derived data,
    including cv_text_anonymized/projects_text/cv_summary), and the
    Matcher Agent calls it again later (only to update skills), for the
    SAME candidate_id. Because this used INSERT OR REPLACE, that second
    call used to silently WIPE OUT the CV-derived fields back to NULL,
    since the Matcher Agent's call never passed them (they default to
    None) - no error, just silent data loss on every re-save.

    Fix: when cv_text_anonymized/projects_text/cv_summary aren't
    explicitly passed (i.e. still None), preserve whatever is already in
    the database for this candidate instead of blindly overwriting it
    with NULL. Any caller that HAS real CV data (the Reader Agent) still
    saves it normally; callers that don't (the Matcher Agent) no longer
    destroy it.
    """
    with get_connection() as conn:
        if cv_text_anonymized is None or projects_text is None or cv_summary is None:
            existing = conn.execute(
                "SELECT cv_text_anonymized, projects_text, cv_summary FROM candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if existing:
                if cv_text_anonymized is None:
                    cv_text_anonymized = existing["cv_text_anonymized"]
                if projects_text is None:
                    projects_text = existing["projects_text"]
                if cv_summary is None:
                    cv_summary = existing["cv_summary"]

        conn.execute(
            """INSERT OR REPLACE INTO candidates
               (candidate_id, job_id, skills, organizations, proxy_fields,
                cv_text_anonymized, projects_text, cv_summary)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (candidate_id, job_id, json.dumps(skills or []),
             json.dumps(organizations or []), json.dumps(proxy_fields or {}),
             cv_text_anonymized, projects_text, cv_summary),
        )


def get_candidates_for_job(job_id):
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM candidates WHERE job_id = ?", (job_id,)).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["skills"] = json.loads(d["skills"] or "[]")
            d["organizations"] = json.loads(d["organizations"] or "[]")
            d["proxy_fields"] = json.loads(d["proxy_fields"] or "{}")
            results.append(d)
        return results


def get_candidate(candidate_id):
    """
    Fetches a single candidate's anonymized profile - including their
    anonymized CV text - so the Explainer Agent can generate interview
    questions grounded in what the candidate actually wrote (specific
    projects, tools, metrics), not just their flat skill-keyword list.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["skills"] = json.loads(d["skills"] or "[]")
        d["organizations"] = json.loads(d["organizations"] or "[]")
        d["proxy_fields"] = json.loads(d["proxy_fields"] or "{}")
        return d


# ---------------------------------------------------------------
# PII vault - access-controlled, every read is logged
# ---------------------------------------------------------------
def save_pii(candidate_id, name=None, email=None, phone=None, age=None,
             gender=None, data_retention_choice="delete"):
    """
    name/email/phone are encrypted before being written (see
    _encrypt_field above) - age/gender are never populated by the Reader
    Agent by design (see reader_agent.py's extract_pii docstring), so
    they're stored as-is if ever provided.
    """
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO pii_vault
               (candidate_id, name, email, phone, age, gender, data_retention_choice)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (candidate_id, _encrypt_field(name), _encrypt_field(email),
             _encrypt_field(phone), age, gender, data_retention_choice),
        )


def get_pii(candidate_id, accessed_by, purpose):
    """
    Every call to this function is logged - this is the audit trail that
    supports the "Accountability" layer. Never call this from the Matcher
    or Explainer's ranking logic - only from the Dashboard (human review)
    or an aggregate-only bias audit.

    name/email/phone are decrypted here before being returned, so callers
    never need to know encryption is happening - they get plain strings
    back exactly as before.
    """
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO vault_access_log (candidate_id, accessed_by, purpose) VALUES (?, ?, ?)",
            (candidate_id, accessed_by, purpose),
        )
        row = conn.execute(
            "SELECT * FROM pii_vault WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["name"] = _decrypt_field(d["name"])
        d["email"] = _decrypt_field(d["email"])
        d["phone"] = _decrypt_field(d["phone"])
        return d


VALID_RETENTION_CHOICES = ("delete", "similar_roles", "indefinite")


def update_data_retention_choice(candidate_id, choice):
    """
    Lets a candidate's actual data-retention preference be recorded, e.g.
    when they click one of the "keep me in your talent pool" / "delete my
    data" links included in their notification email. Before this
    existed, save_pii()'s "delete" default was the only value ever
    stored - candidates were never actually given the chance to choose,
    even though earlier email copy implied they had (a bug fixed
    alongside this function). Returns True if a row was updated, False if
    no PII record exists for this candidate_id.
    """
    if choice not in VALID_RETENTION_CHOICES:
        raise ValueError(f"Invalid choice '{choice}' - must be one of {VALID_RETENTION_CHOICES}")

    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE pii_vault SET data_retention_choice = ? WHERE candidate_id = ?",
            (choice, candidate_id),
        )
        return cur.rowcount > 0


def close_job(job_id):
    """
    Marks a job as closed and immediately deletes the PII vault row (name,
    email, phone) for every candidate on this job who chose 'delete' as
    their retention preference - this is the step that makes the email's
    promise ("we will delete your data once this role is closed") actually
    true, instead of update_data_retention_choice() just recording a
    preference that nothing ever acted on.

    Only pii_vault is touched. The candidates/rankings/decisions_log rows
    are deliberately left alone - they were already anonymized by design
    (that's the whole point of the PII vault separation) and are still
    needed for the bias-audit trail, so there's no privacy reason to erase
    them too.

    Returns a dict listing which candidate_ids had their PII deleted, so
    the caller (e.g. the Dashboard) can show the recruiter what happened.
    """
    with get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'closed', closed_at = CURRENT_TIMESTAMP WHERE job_id = ?",
            (job_id,),
        )
        candidate_rows = conn.execute(
            "SELECT candidate_id FROM candidates WHERE job_id = ?", (job_id,)
        ).fetchall()

        deleted = []
        for row in candidate_rows:
            cid = row["candidate_id"]
            pii_row = conn.execute(
                "SELECT data_retention_choice FROM pii_vault WHERE candidate_id = ?", (cid,)
            ).fetchone()
            if pii_row and pii_row["data_retention_choice"] == "delete":
                conn.execute("DELETE FROM pii_vault WHERE candidate_id = ?", (cid,))
                deleted.append(cid)

        return {"job_id": job_id, "pii_deleted_for": deleted}


def purge_expired_retention(similar_roles_days=180):
    """
    Catch-all cleanup, meant to run periodically (in production this
    should be a scheduled task - e.g. a daily cron job or APScheduler -
    rather than only running when someone happens to click a button).
    Handles two cases close_job() alone doesn't cover:
      1. A candidate who chose 'delete' AFTER their job was already
         closed (close_job() only sweeps at the moment of closing).
      2. A candidate who chose 'similar_roles' and whose ~6-month window
         has now elapsed since their job closed.
    Returns the list of candidate_ids whose PII was deleted this run.
    """
    with get_connection() as conn:
        closed_jobs = conn.execute(
            "SELECT job_id, closed_at FROM jobs WHERE status = 'closed'"
        ).fetchall()

        deleted = []
        for job in closed_jobs:
            candidate_rows = conn.execute(
                "SELECT candidate_id FROM candidates WHERE job_id = ?", (job["job_id"],)
            ).fetchall()

            for row in candidate_rows:
                cid = row["candidate_id"]
                pii_row = conn.execute(
                    "SELECT data_retention_choice FROM pii_vault WHERE candidate_id = ?", (cid,)
                ).fetchone()
                if not pii_row:
                    continue

                choice = pii_row["data_retention_choice"]
                if choice == "delete":
                    conn.execute("DELETE FROM pii_vault WHERE candidate_id = ?", (cid,))
                    deleted.append(cid)
                elif choice == "similar_roles" and job["closed_at"]:
                    days_row = conn.execute(
                        "SELECT (julianday('now') - julianday(?)) AS days", (job["closed_at"],)
                    ).fetchone()
                    elapsed = days_row["days"] if days_row else None
                    if elapsed is not None and elapsed >= similar_roles_days:
                        conn.execute("DELETE FROM pii_vault WHERE candidate_id = ?", (cid,))
                        deleted.append(cid)

        return {"pii_deleted_for": deleted}


# ---------------------------------------------------------------
# Rankings (Matcher + Explainer output)
# ---------------------------------------------------------------
def save_ranking(candidate_id, job_id, semantic_score, keyword_score, final_score,
                  matched_skills=None, missing_skills=None, explanation=None,
                  consistency_check_passed=None):
    """
    Saves or UPDATES a candidate's ranking for a job. If the Matcher Agent
    already created a row for this candidate+job, this updates it (adding
    the explanation later) instead of creating a duplicate row - this is
    what lets the Matcher Agent and Explainer Agent both write to the same
    ranking record at different points in the pipeline.
    """
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT ranking_id FROM rankings WHERE candidate_id = ? AND job_id = ?",
            (candidate_id, job_id),
        ).fetchone()

        matched_json = json.dumps(matched_skills) if matched_skills is not None else None
        missing_json = json.dumps(missing_skills) if missing_skills is not None else None
        consistency_int = int(consistency_check_passed) if consistency_check_passed is not None else None

        if existing:
            # Update only the fields that were actually provided this call -
            # so the Explainer Agent adding an explanation later doesn't
            # accidentally wipe out the scores the Matcher Agent already saved.
            fields, values = [], []
            for col, val in [
                ("semantic_score", semantic_score), ("keyword_score", keyword_score),
                ("final_score", final_score), ("matched_skills", matched_json),
                ("missing_skills", missing_json), ("explanation", explanation),
                ("consistency_check_passed", consistency_int),
            ]:
                if val is not None:
                    fields.append(f"{col} = ?")
                    values.append(val)
            if fields:
                values.append(existing["ranking_id"])
                conn.execute(f"UPDATE rankings SET {', '.join(fields)} WHERE ranking_id = ?", values)
        else:
            conn.execute(
                """INSERT INTO rankings
                   (candidate_id, job_id, semantic_score, keyword_score, final_score,
                    matched_skills, missing_skills, explanation, consistency_check_passed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (candidate_id, job_id, semantic_score, keyword_score, final_score,
                 matched_json or "[]", missing_json or "[]", explanation, consistency_int),
            )


def get_rankings_for_job(job_id):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM rankings WHERE job_id = ? ORDER BY final_score DESC", (job_id,)
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["matched_skills"] = json.loads(d["matched_skills"] or "[]")
            d["missing_skills"] = json.loads(d["missing_skills"] or "[]")
            results.append(d)
        return results


# ---------------------------------------------------------------
# Decisions log (human accept/reject - Accountability layer)
# ---------------------------------------------------------------
def log_decision(candidate_id, job_id, decision, stage="shortlist", reason=None,
                  interview_date=None, interview_time=None, interview_location=None,
                  decided_by=None):
    """
    stage='shortlist' -> the recruiter's first decision, before any interview
        decision='accepted' -> candidate gets an interview invite
        decision='rejected' -> this is an EARLY rejection (never interviewed)
    stage='final' -> the decision AFTER the interview happened
        decision='accepted' -> candidate is hired
        decision='rejected' -> this is a POST-INTERVIEW rejection

    notified always starts at 0 - the actual email is sent separately
    (either immediately, or in a batch later) and mark_notified() is
    called once it's actually sent.
    """
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO decisions_log
               (candidate_id, job_id, stage, decision, reason, interview_date,
                interview_time, interview_location, decided_by, notified)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
            (candidate_id, job_id, stage, decision, reason, interview_date,
             interview_time, interview_location, decided_by),
        )


def get_pending_notifications(job_id, decision_filter=None):
    """
    Returns decisions that haven't been notified yet for this job.
    decision_filter can be 'accepted' or 'rejected' to get just one type -
    this is what lets the Dashboard offer a "send all pending rejection
    emails" batch action instead of sending one at a time.
    """
    with get_connection() as conn:
        if decision_filter:
            rows = conn.execute(
                """SELECT * FROM decisions_log
                   WHERE job_id = ? AND notified = 0 AND decision = ?""",
                (job_id, decision_filter),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM decisions_log WHERE job_id = ? AND notified = 0",
                (job_id,),
            ).fetchall()
        return [dict(row) for row in rows]


def mark_notified(decision_id):
    """Call this once the Notification Agent has actually generated/sent
    the email for this decision, so it doesn't get sent again."""
    with get_connection() as conn:
        conn.execute("UPDATE decisions_log SET notified = 1 WHERE decision_id = ?", (decision_id,))


def get_decisions_for_job(job_id):
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM decisions_log WHERE job_id = ?", (job_id,)).fetchall()
        return [dict(row) for row in rows]


def get_notification_status(candidate_id, job_id):
    """
    Looks at this candidate's decision history for this job and works out
    which of the 3 email types they should receive right now - so nobody
    has to manually specify the status, it's derived from real decisions.

    Returns one of: 'shortlisted', 'early_rejected', 'interview_rejected',
    or None if no decision has been made yet (don't send anything).
    """
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM decisions_log WHERE candidate_id = ? AND job_id = ?
               ORDER BY created_at DESC, decision_id DESC""",
            (candidate_id, job_id),
        ).fetchall()

    if not rows:
        return None, None

    latest = dict(rows[0])

    if latest["stage"] == "final":
        if latest["decision"] == "rejected":
            return "interview_rejected", latest
        else:
            return None, latest  # accepted at final stage = hired, not a rejection email
    else:  # stage == 'shortlist'
        if latest["decision"] == "rejected":
            return "early_rejected", latest
        else:
            return "shortlisted", latest


# ---------------------------------------------------------------
# Bias audit log
# ---------------------------------------------------------------
def log_bias_audit(job_id, flagged, avg_with, avg_without, note):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO bias_audit_log
               (job_id, flagged, avg_score_with_proxy_field, avg_score_without_proxy_field, note)
               VALUES (?, ?, ?, ?, ?)""",
            (job_id, int(flagged), avg_with, avg_without, note),
        )


if __name__ == "__main__":
    # Quick self-test: create the DB and do a round-trip save/load
    init_db()
    job_id = save_job("Data Analyst", "Looking for Python, SQL, Power BI skills")
    save_candidate("CAND_001", job_id, skills=["python", "sql"])
    save_ranking("CAND_001", job_id, 0.65, 1.0, 0.79,
                 matched_skills=["python", "sql"], missing_skills=["power bi"],
                 explanation="Strong match on core skills.", consistency_check_passed=True)
    log_decision("CAND_001", job_id, "accepted", decided_by="test_recruiter")

    print("Job:", get_job(job_id))
    print("Candidates:", get_candidates_for_job(job_id))
    print("Rankings:", get_rankings_for_job(job_id))
    print("Decisions:", get_decisions_for_job(job_id))
    print("\nDatabase self-test passed - fairhire.db created successfully.")