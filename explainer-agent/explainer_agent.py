"""
FairHire - Explainer Agent
----------------------------
Job: This agent has FOUR responsibilities:
  1. Explain why each candidate ranked where they did (LLM)
  2. Audit the ranking results for bias patterns (proxy-field monitoring)
  3. Generate tailored interview questions once a candidate is accepted (LLM)
  4. Generate skills-gap feedback for "near-miss" candidates - people who were
     close but didn't qualify, telling them specifically what to improve
     (this is the differentiator feature - most systems just reject silently)

Setup:
    pip install google-generativeai fastapi uvicorn

Get a free API key:
    1. Go to aistudio.google.com
    2. Sign in, click "Get API key" - free, no credit card needed
    3. Set it as an environment variable before running:
       Windows (PowerShell):  $env:GEMINI_API_KEY="your-key-here"
       Mac/Linux:             export GEMINI_API_KEY="your-key-here"

Run standalone (sample data):
    python explainer_agent.py

Run as an API other agents can call:
    uvicorn explainer_agent:app --port 8003 --reload
"""

import os
import json
import warnings
from typing import List, Optional, Dict

warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel

# --- Config ---
NEAR_MISS_LOWER_BOUND = 0.30   # scores in this range count as "near miss"
NEAR_MISS_UPPER_BOUND = 0.50   # (below MIN_MATCH_THRESHOLD from matcher_agent)
BIAS_FLAG_GAP_THRESHOLD = 0.15  # how much difference between groups triggers a flag

API_KEY = os.environ.get("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-3.6-flash")
else:
    model = None  # allows the file to still be imported/tested without a key


def call_llm(prompt: str) -> str:
    """Central place all LLM calls go through - makes it easy to swap providers."""
    if model is None:
        return "[LLM not configured - set GEMINI_API_KEY to enable real responses]"
    response = model.generate_content(prompt)
    return response.text.strip()


# ---------------------------------------------------------------
# 1. EXPLANATION - why did this candidate rank where they did?
# ---------------------------------------------------------------
def generate_explanation(candidate: dict, job_description: str) -> dict:
    """
    Generates the LLM explanation, then runs a lightweight NLP consistency
    check: does the explanation actually mention skills the candidate really
    has? This is the Explainer Agent's NLP component - it doesn't trust the
    LLM's output blindly, it verifies it against the real structured data.
    """
    prompt = f"""You are explaining a candidate ranking to a recruiter.

Job description: {job_description}
Candidate skills: {', '.join(candidate.get('skills', []))}
Candidate's match score: {candidate.get('final_score', 'N/A')}

Write a 2-sentence, plain-English explanation of why this candidate received
this ranking, mentioning specific matched skills. Do not mention name, age,
or gender - this data was not provided and should never be referenced."""
    explanation_text = call_llm(prompt)

    # --- NLP consistency check (no LLM needed - fast, deterministic) ---
    actual_skills = [s.lower() for s in candidate.get("skills", [])]
    explanation_lower = explanation_text.lower()
    mentioned_real_skills = [s for s in actual_skills if s in explanation_lower]
    consistency_ok = len(mentioned_real_skills) > 0 if actual_skills else True

    return {
        "explanation": explanation_text,
        "consistency_check": {
            "passed": consistency_ok,
            "skills_confirmed_in_text": mentioned_real_skills,
            "note": (
                "Explanation references at least one of the candidate's actual "
                "skills - grounded in real data."
                if consistency_ok else
                "Warning: explanation does not clearly reference the candidate's "
                "actual listed skills - worth a manual check before trusting it."
            ),
        },
        # --- THE VERIFIABILITY FIX ---
        # This raw data is shown to the recruiter ALONGSIDE the explanation
        # (not hidden), so the explanation is never something they have to
        # trust blindly - they can check it against the actual numbers and
        # matched/missing skills the Matcher Agent produced.
        "verifiable_data": {
            "final_score": candidate.get("final_score"),
            "semantic_score": candidate.get("semantic_score"),
            "keyword_score": candidate.get("keyword_score"),
            "matched_skills": candidate.get("skills", []),
            "missing_skills": candidate.get("missing_skills", []),
        },
    }


# ---------------------------------------------------------------
# 2. BIAS AUDIT - proxy-field monitoring across the ranked results
# ---------------------------------------------------------------
def audit_bias(ranked_candidates: List[dict]) -> dict:
    """
    Checks whether candidates with a flagged proxy field (e.g. an employment
    gap) score systematically lower than those without it, even though the
    ranking never saw age/gender directly. This is the "Monitoring" layer
    of the three-layer fairness approach.
    """
    with_gap = [c["final_score"] for c in ranked_candidates
                if c.get("proxy_fields", {}).get("employment_gap_detected")]
    without_gap = [c["final_score"] for c in ranked_candidates
                   if not c.get("proxy_fields", {}).get("employment_gap_detected")]

    if not with_gap or not without_gap:
        return {"flagged": False, "note": "Not enough data in both groups to compare."}

    avg_with = sum(with_gap) / len(with_gap)
    avg_without = sum(without_gap) / len(without_gap)
    gap_difference = avg_without - avg_with

    flagged = gap_difference > BIAS_FLAG_GAP_THRESHOLD
    return {
        "flagged": flagged,
        "avg_score_with_employment_gap": round(avg_with, 3),
        "avg_score_without_employment_gap": round(avg_without, 3),
        "difference": round(gap_difference, 3),
        "note": (
            "Candidates with an employment gap scored notably lower on average "
            "- worth a manual review to check this isn't proxy discrimination."
            if flagged else
            "No significant score difference detected between groups."
        ),
    }


# ---------------------------------------------------------------
# 3. INTERVIEW QUESTIONS - generated once a candidate is accepted
# ---------------------------------------------------------------
def generate_interview_questions(candidate: dict, job_description: str) -> List[str]:
    prompt = f"""Generate 5 tailored interview questions for this candidate.

Job description: {job_description}
Candidate skills: {', '.join(candidate.get('skills', []))}

Include at least one question probing a specific skill they claim, and one
addressing any gap between their skills and the job requirements. Return
just the 5 questions, one per line, no extra commentary."""
    response = call_llm(prompt)
    return [q.strip("- ").strip() for q in response.split("\n") if q.strip()]


# ---------------------------------------------------------------
# 4. SKILLS-GAP FEEDBACK - the differentiator feature
#    For "near-miss" candidates: tell them specifically what to improve,
#    instead of a silent/generic rejection.
# ---------------------------------------------------------------
def generate_skills_gap_feedback(candidate: dict, job_description: str) -> Optional[str]:
    score = candidate.get("final_score", 0)
    if not (NEAR_MISS_LOWER_BOUND <= score < NEAR_MISS_UPPER_BOUND):
        return None  # only applies to near-miss candidates, not clear rejects

    prompt = f"""This candidate was a NEAR MISS for a job - close, but not
quite qualified. Job description: {job_description}
Candidate's current skills: {', '.join(candidate.get('skills', []))}

Write a short, encouraging 2-3 sentence note identifying 1-2 SPECIFIC skills
or areas they should develop to be a stronger fit for similar roles in the
future. Be constructive and kind, not clinical. Do not mention their score
or ranking number."""
    return call_llm(prompt)


# ---------------------------------------------------------------
# FastAPI wrapper
# ---------------------------------------------------------------
app = FastAPI(title="FairHire Explainer Agent")


class Candidate(BaseModel):
    candidate_id: str
    skills: List[str]
    final_score: Optional[float] = 0.0
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    missing_skills: Optional[List[str]] = []
    proxy_fields: Optional[Dict] = {}


class ExplainRequest(BaseModel):
    job_description: str
    candidates: List[Candidate]


@app.get("/health")
def health():
    return {"status": "ok", "agent": "explainer", "llm_configured": model is not None}


@app.post("/explain")
def explain_endpoint(request: ExplainRequest):
    """Generates an explanation + skills-gap feedback (if applicable) for each candidate."""
    results = []
    for c in request.candidates:
        c_dict = c.dict()
        results.append({
            "candidate_id": c.candidate_id,
            "explanation": generate_explanation(c_dict, request.job_description),
            "skills_gap_feedback": generate_skills_gap_feedback(c_dict, request.job_description),
        })
    return {"explanations": results}


@app.post("/audit")
def audit_endpoint(request: ExplainRequest):
    """Runs the bias audit across all ranked candidates."""
    candidates_as_dicts = [c.dict() for c in request.candidates]
    return audit_bias(candidates_as_dicts)


@app.post("/interview-questions")
def interview_questions_endpoint(candidate: Candidate, job_description: str):
    """Called when a recruiter accepts a specific candidate for interview."""
    return {"candidate_id": candidate.candidate_id,
            "questions": generate_interview_questions(candidate.dict(), job_description)}


if __name__ == "__main__":
    job_description = "Looking for a Data Analyst with Python, SQL, and Power BI experience"

    sample_candidates = [
        {"candidate_id": "CAND_001", "skills": ["python", "sql", "power bi"],
         "final_score": 0.79, "proxy_fields": {"employment_gap_detected": False}},
        {"candidate_id": "CAND_002", "skills": ["sql", "excel"],
         "final_score": 0.40, "proxy_fields": {"employment_gap_detected": True}},
        {"candidate_id": "CAND_003", "skills": ["java", "project management"],
         "final_score": 0.11, "proxy_fields": {"employment_gap_detected": False}},
    ]

    print("=== Explanations (with NLP consistency check) ===")
    for c in sample_candidates:
        result = generate_explanation(c, job_description)
        print(c["candidate_id"], "->", result["explanation"])
        print("   consistency check:", result["consistency_check"]["note"])

    print("\n=== Bias Audit ===")
    print(json.dumps(audit_bias(sample_candidates), indent=2))

    print("\n=== Skills-gap feedback (near-miss only) ===")
    for c in sample_candidates:
        feedback = generate_skills_gap_feedback(c, job_description)
        if feedback:
            print(c["candidate_id"], "->", feedback)

    print("\n=== Interview questions (top candidate) ===")
    print(generate_interview_questions(sample_candidates[0], job_description))