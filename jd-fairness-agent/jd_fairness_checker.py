"""
FairHire - Job Description Fairness Checker 
----------------------------------------------
Job: This is the NEW "Pre-Prevention" layer - runs BEFORE any candidates are
processed. Analyzes the job description itself for language patterns known
to correlate with reduced application rates from underrepresented groups,
and flags suggestions for the recruiter - who can revise or proceed anyway.

This extends the three-layer fairness approach into four layers:
  0. Pre-Prevention (NEW) - catch bias in the job posting itself
  1. Prevention        - PII blinding at CV intake 
  2. Monitoring         - proxy-field bias auditing on rankings
  3. Accountability     - human-in-the-loop + logging 

Setup:
    pip install google-generativeai fastapi uvicorn

Run standalone (sample data):
    python jd_fairness_checker.py

Run as an API:
    uvicorn jd_fairness_checker:app --port 8004 --reload
"""

import os
import re
import json
import sys
from typing import List

from dotenv import load_dotenv
load_dotenv()  # reads the .env file in the project root, if present

# Import the shared database module (lives in ../database relative to this file)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database"))
from db import init_db, save_job
from shared.auth import verify_api_key

import google.generativeai as genai
from fastapi import FastAPI, Depends
from pydantic import BaseModel

API_KEY = os.environ.get("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-3.5-flash-lite")
else:
    model = None

#1st layer: keyword scan - fast, deterministic, no LLM needed
def call_llm(prompt: str) -> str:
    if model is None:
        return "[LLM not configured - set GEMINI_API_KEY to enable real responses]"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        # Never let an LLM failure (quota limit, model renamed, network
        # issue, etc.) crash the whole request - fall back to a clear
        # message so the keyword-scan half of this agent still works.
        return f"[LLM check unavailable right now: {type(e).__name__}]"


# A small, known-pattern keyword list as a fast, non-LLM first pass -
# genuine research (Textio, Gender Decoder, and similar studies) has shown
# certain word categories correlate with reduced application rates.
MASCULINE_CODED_WORDS = [
    "dominant", "dominate", "aggressive", "competitive", "ninja", "rockstar",
    "crush", "fearless", "superior", "relentless",
]
AGE_CODED_PHRASES = [
    "young and energetic", "recent graduate preferred", "digital native",
]
VAGUE_EXCESSIVE_REQUIREMENTS = [
    "10+ years", "must be an expert in everything", "unicorn",
]


def keyword_scan(job_description: str) -> List[dict]:
    """Fast, deterministic first pass - no LLM needed, catches obvious cases instantly."""
    text_lower = job_description.lower()
    flags = []

    for word in MASCULINE_CODED_WORDS:
        if word in text_lower:
            flags.append({
                "term": word,
                "category": "competitive/masculine-coded language",
                "note": "Research (e.g. Gender Decoder, Textio) associates this "
                        "kind of language with reduced application rates from women.",
            })
    for phrase in AGE_CODED_PHRASES:
        if phrase in text_lower:
            flags.append({
                "term": phrase,
                "category": "age-coded language",
                "note": "This phrasing may discourage older applicants and can "
                        "carry age-discrimination risk.",
            })
    for phrase in VAGUE_EXCESSIVE_REQUIREMENTS:
        if phrase in text_lower:
            flags.append({
                "term": phrase,
                "category": "excessive/vague requirement",
                "note": "Overly broad requirements can discourage qualified "
                        "candidates (especially women) who tend to under-apply "
                        "unless they meet nearly all listed criteria.",
            })
    return flags


def llm_deep_check(job_description: str) -> str:
    """Second pass - LLM catches subtler patterns the keyword list would miss."""
    prompt = f"""Analyze this job description for language patterns that could
reduce application rates from underrepresented groups (e.g. subtly gendered
wording, exclusionary jargon, unnecessary requirements). Job description:

{job_description}

List up to 3 specific concerns in plain English, or say "No significant
concerns found" if none apply. Do not rewrite the job description, just
identify concerns."""
    return call_llm(prompt)


def check_job_description(job_description: str) -> dict:
    keyword_flags = keyword_scan(job_description)
    llm_notes = llm_deep_check(job_description)

    return {
        "keyword_flags": keyword_flags,
        "llm_analysis": llm_notes,
        "recommendation": (
            "Consider revising the flagged language before posting."
            if keyword_flags else
            "No obvious exclusionary patterns detected by the keyword scan; "
            "see LLM analysis above for subtler concerns."
        ),
        "note": "This check informs the recruiter - it does not block posting. "
                "The recruiter decides whether to revise or proceed.",
    }

##2
# ---------------------------------------------------------------
# FastAPI wrapper
# ---------------------------------------------------------------
app = FastAPI(title="FairHire Job Description Fairness Checker")
init_db()  # creates tables if they don't exist yet - safe to call every startup


class JDRequest(BaseModel):
    job_description: str
    title: str = "Untitled role"

##3
class ConfirmJobRequest(BaseModel):
    title: str
    job_description: str
    proceed_despite_flags: bool = False  # recruiter's explicit choice

##4
@app.get("/health")
def health():
    return {"status": "ok", "agent": "jd_fairness_checker", "llm_configured": model is not None}

##5
@app.post("/check", dependencies=[Depends(verify_api_key)])
def check_endpoint(request: JDRequest):
    """
    Step 1: check the job description BEFORE it's saved as a real posting.
    This does NOT save anything to the database yet - it just returns
    flags so the recruiter can decide whether to revise or proceed.
    """
    return check_job_description(request.job_description)


@app.post("/confirm-and-post", dependencies=[Depends(verify_api_key)])
def confirm_and_post_endpoint(request: ConfirmJobRequest):
    """
    Step 2: once the recruiter has seen the /check results and decided to
    revise or proceed anyway, THIS is what actually creates the job in the
    database - with the fairness check result permanently attached, so
    there's a record of what was flagged (or not) when the job went live.
    """
    check_result = check_job_description(request.job_description)
    job_id = save_job(request.title, request.job_description,
                       fairness_check_flags=check_result["keyword_flags"])
    return {
        "job_id": job_id,
        "fairness_check": check_result,
        "posted_despite_flags": bool(check_result["keyword_flags"]) and request.proceed_despite_flags,
    }


if __name__ == "__main__":
    sample_jd = (
        "We're looking for a rockstar Data Analyst who can dominate the "
        "competition. Must be a young and energetic self-starter with "
        "10+ years of experience in every BI tool imaginable."
    )
    print(json.dumps(check_job_description(sample_jd), indent=2))
 