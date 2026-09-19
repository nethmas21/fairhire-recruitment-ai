"""
FairHire - Untested Feature Verification Script
---------------------------------------------------
Exercises four features that exist in code but have never been run in
this project's testing so far:
  1. JD Fairness Checker  (port 8004)
  2. Bias Audit            (port 8003)
  3. Near-miss skills-gap feedback  (port 8003)
  4. Agentic threshold-widening     (port 8002)

Run this AFTER all 5 agents are up (reader/matcher/explainer/jd-fairness/
notification), from anywhere - it finds the shared .env automatically via
python-dotenv's upward search, as long as this script sits somewhere
inside the project folder tree.

    python test_untested_features.py
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("FAIRHIRE_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY}

if not API_KEY:
    print("WARNING: FAIRHIRE_API_KEY not found in .env - requests below will "
          "likely get 401 Unauthorized if the agents have auth enabled.\n")


def show(label, response):
    print(f"--- {label} ---")
    print("Status:", response.status_code)
    try:
        print(json.dumps(response.json(), indent=2))
    except Exception:
        print(response.text)
    print()


# ---------------------------------------------------------------
# TEST 1: JD Fairness Checker - deliberately biased job description,
# should trigger multiple keyword_flags plus LLM commentary.
# ---------------------------------------------------------------
biased_jd = (
    "We're looking for a rockstar Data Analyst who can dominate the "
    "competition. Must be a young and energetic self-starter with "
    "10+ years of experience in every BI tool imaginable."
)
try:
    r = requests.post(
        "http://127.0.0.1:8004/check",
        json={"job_description": biased_jd, "title": "Test Role"},
        headers=HEADERS,
    )
    show("TEST 1: JD Fairness Checker (expect several keyword_flags)", r)
except requests.exceptions.ConnectionError:
    print("TEST 1 SKIPPED - could not reach JD Fairness Checker on port 8004\n")


# ---------------------------------------------------------------
# TEST 2: Bias Audit - two candidates with an employment gap scoring
# notably lower than two without one. Should flag=True.
# ---------------------------------------------------------------
audit_candidates = [
    {"candidate_id": "A1", "skills": ["python", "sql"], "final_score": 0.85,
     "proxy_fields": {"employment_gap_detected": False}},
    {"candidate_id": "A2", "skills": ["python", "sql"], "final_score": 0.82,
     "proxy_fields": {"employment_gap_detected": False}},
    {"candidate_id": "A3", "skills": ["python", "sql"], "final_score": 0.40,
     "proxy_fields": {"employment_gap_detected": True}},
    {"candidate_id": "A4", "skills": ["python", "sql"], "final_score": 0.38,
     "proxy_fields": {"employment_gap_detected": True}},
]
try:
    r = requests.post(
        "http://127.0.0.1:8003/audit",
        json={"job_description": "Data Analyst role", "candidates": audit_candidates},
        headers=HEADERS,
    )
    show("TEST 2: Bias Audit (expect flagged: true)", r)
except requests.exceptions.ConnectionError:
    print("TEST 2 SKIPPED - could not reach Explainer Agent on port 8003\n")


# ---------------------------------------------------------------
# TEST 3: Near-miss skills-gap feedback - final_score 0.40 falls inside
# NEAR_MISS_LOWER_BOUND (0.30) / NEAR_MISS_UPPER_BOUND (0.50), so
# skills_gap_feedback should be populated (not null).
# ---------------------------------------------------------------
near_miss_candidate = {
    "candidate_id": "NM1",
    "skills": ["sql", "excel"],
    "final_score": 0.40,
    "missing_skills": ["python", "power bi"],
}
try:
    r = requests.post(
        "http://127.0.0.1:8003/explain",
        json={
            "job_description": "Looking for a Data Analyst with Python, SQL, and Power BI experience",
            "candidates": [near_miss_candidate],
        },
        headers=HEADERS,
    )
    show("TEST 3: Near-miss feedback (expect skills_gap_feedback to be non-null)", r)
except requests.exceptions.ConnectionError:
    print("TEST 3 SKIPPED - could not reach Explainer Agent on port 8003\n")


# ---------------------------------------------------------------
# TEST 4: Agentic threshold-widening - only 2 weak candidates, neither
# should clear MIN_MATCH_THRESHOLD (0.5), so fewer than
# MIN_CANDIDATES_REQUIRED (3) qualify -> widening_note should mention
# "automatically widened".
# ---------------------------------------------------------------
weak_candidates = [
    {"candidate_id": "W1", "skills": ["java"], "organizations": []},
    {"candidate_id": "W2", "skills": ["excel"], "organizations": []},
]
try:
    r = requests.post(
        "http://127.0.0.1:8002/match",
        json={
            "job_description": "Looking for a Data Analyst with Python, SQL, and Power BI experience",
            "candidates": weak_candidates,
        },
        headers=HEADERS,
    )
    show("TEST 4: Agentic threshold-widening (expect widening_note to mention widening)", r)
except requests.exceptions.ConnectionError:
    print("TEST 4 SKIPPED - could not reach Matcher Agent on port 8002\n")

print("Done. Review each section above against the 'expect' comment in this script.")