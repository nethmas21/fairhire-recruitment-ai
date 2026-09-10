-- FairHire Database Schema
-- SQLite for development. Upgrade path to PostgreSQL for production.
-- Every table here is shared across all agents - everyone on the team
-- reads/writes through database/db.py, not directly through raw SQL,
-- so the schema stays consistent no matter who's coding which agent.

CREATE TABLE IF NOT EXISTS jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    fairness_check_flags TEXT,      -- JSON: flags from the JD Fairness Agent
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY, -- e.g. "CAND_001" - anonymous ID only
    job_id INTEGER NOT NULL,
    skills TEXT,                    -- JSON list, e.g. ["python","sql"]
    organizations TEXT,             -- JSON list
    proxy_fields TEXT,               -- JSON, e.g. {"employment_gap_detected": true}
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

-- Personal identifying information - kept SEPARATE from the candidates
-- table on purpose. Only accessed by the Dashboard (human review) or the
-- Explainer Agent's aggregate-only bias audit - never by the Matcher.
CREATE TABLE IF NOT EXISTS pii_vault (
    candidate_id TEXT PRIMARY KEY,
    name TEXT,
    email TEXT,
    phone TEXT,
    age TEXT,
    gender TEXT,
    data_retention_choice TEXT DEFAULT 'delete',  -- 'delete' | 'similar_roles' | 'indefinite'
    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
);

CREATE TABLE IF NOT EXISTS rankings (
    ranking_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    job_id INTEGER NOT NULL,
    semantic_score REAL,
    keyword_score REAL,
    final_score REAL,
    matched_skills TEXT,    -- JSON list
    missing_skills TEXT,    -- JSON list
    explanation TEXT,
    consistency_check_passed INTEGER,  -- 0 or 1
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS decisions_log (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    job_id INTEGER NOT NULL,
    stage TEXT NOT NULL DEFAULT 'shortlist',  -- 'shortlist' | 'final'
    decision TEXT NOT NULL,           -- 'accepted' | 'rejected'
    reason TEXT,                       -- optional recruiter note
    interview_date TEXT,
    interview_time TEXT,
    interview_location TEXT,
    decided_by TEXT,                   -- recruiter username
    notified INTEGER DEFAULT 0,        -- 0 = email not sent yet, 1 = sent
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
);

-- Every time the PII vault is queried, it's logged here - this is the
-- audit trail supporting the "Accountability" layer of the fairness plan.
CREATE TABLE IF NOT EXISTS vault_access_log (
    access_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT,
    accessed_by TEXT,          -- which agent or user
    purpose TEXT,              -- e.g. "recruiter review", "bias audit aggregate"
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'recruiter',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS bias_audit_log (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    flagged INTEGER,          -- 0 or 1
    avg_score_with_proxy_field REAL,
    avg_score_without_proxy_field REAL,
    note TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);