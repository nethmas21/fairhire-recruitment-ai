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
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "fairhire.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


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
def save_candidate(candidate_id, job_id, skills=None, organizations=None, proxy_fields=None):
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO candidates
               (candidate_id, job_id, skills, organizations, proxy_fields)
               VALUES (?, ?, ?, ?, ?)""",
            (candidate_id, job_id, json.dumps(skills or []),
             json.dumps(organizations or []), json.dumps(proxy_fields or {})),
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


# ---------------------------------------------------------------
# PII vault - access-controlled, every read is logged
# ---------------------------------------------------------------
def save_pii(candidate_id, name=None, email=None, phone=None, age=None,
             gender=None, data_retention_choice="delete"):
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO pii_vault
               (candidate_id, name, email, phone, age, gender, data_retention_choice)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (candidate_id, name, email, phone, age, gender, data_retention_choice),
        )


def get_pii(candidate_id, accessed_by, purpose):
    """
    Every call to this function is logged - this is the audit trail that
    supports the "Accountability" layer. Never call this from the Matcher
    or Explainer's ranking logic - only from the Dashboard (human review)
    or an aggregate-only bias audit.
    """
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO vault_access_log (candidate_id, accessed_by, purpose) VALUES (?, ?, ?)",
            (candidate_id, accessed_by, purpose),
        )
        row = conn.execute(
            "SELECT * FROM pii_vault WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        return dict(row) if row else None


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
def log_decision(candidate_id, job_id, decision, reason=None,
                  interview_date=None, interview_time=None, decided_by=None):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO decisions_log
               (candidate_id, job_id, decision, reason, interview_date, interview_time, decided_by)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (candidate_id, job_id, decision, reason, interview_date, interview_time, decided_by),
        )


def get_decisions_for_job(job_id):
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM decisions_log WHERE job_id = ?", (job_id,)).fetchall()
        return [dict(row) for row in rows]


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