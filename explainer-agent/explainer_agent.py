"""
FairHire - Explainer Agent
----------------------------
Job: This agent has FOUR responsibilities:
  1. Explain why each candidate ranked where they did (LLM)
  2. Audit the ranking results for bias patterns (proxy-field monitoring)
  3. Generate tailored interview questions once a candidate is accepted (LLM) -
     grounded in the candidate's actual anonymized CV text (projects, tools,
     specific experience), not just their flat skill-keyword list
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
import sys
from typing import List, Optional, Dict

from dotenv import load_dotenv
load_dotenv()  # reads the .env file in the project root, if present

warnings.filterwarnings("ignore", category=FutureWarning)

# Import the shared database module (lives in ../database relative to this file)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database"))
from db import init_db, save_ranking, log_bias_audit, get_job, get_candidate, get_rankings_for_job
from auth import verify_api_key

import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

# --- Config ---
NEAR_MISS_LOWER_BOUND = 0.30   # scores in this range count as "near miss"
NEAR_MISS_UPPER_BOUND = 0.50   # (below MIN_MATCH_THRESHOLD from matcher_agent)
BIAS_FLAG_GAP_THRESHOLD = 0.15  # how much difference between groups triggers a flag

API_KEY = os.environ.get("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-3.5-flash-lite")
else:
    model = None  # allows the file to still be imported/tested without a key


def call_llm(prompt: str) -> str:
    """Central place all LLM calls go through - makes it easy to swap providers."""
    if model is None:
        return "[LLM not configured - set GEMINI_API_KEY to enable real responses]"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        # Never let an LLM failure (quota limit, model renamed, network
        # issue, etc.) crash the whole request - this is the exact bug
        # that caused a 500 error when the free-tier quota was exceeded.
        return f"[LLM explanation unavailable right now: {type(e).__name__}]"


# ---------------------------------------------------------------
# 1. EXPLANATION - why did this candidate rank where they did?
# ---------------------------------------------------------------
# Phrases that mean the LLM is speculating about WHY the scoring system or
# algorithm behaved a certain way (e.g. "parsing failed to recognize skills")
# rather than just describing the skill overlap. The Explainer Agent has no
# actual visibility into the Matcher Agent's internals, so any such claim is
# an unverifiable guess dressed up as fact - never something to trust.
SCORE_MECHANICS_PHRASES = [
    "parsing", "algorithm failed", "algorithm struggled", "failed to recognize",
    "failed to properly recognize", "scoring algorithm", "automated parsing",
    "evaluation system", "system failed", "system struggled", "system's",
]

# Phrases implying a poor/low match - checked against the actual matched vs
# missing skill counts to catch the LLM asserting a narrative the real data
# contradicts (e.g. calling it a "low match" when 11 of 12 skills matched).
LOW_MATCH_PHRASES = ["low score", "low match", "poor match", "poor fit", "weak match"]


def generate_explanation(candidate: dict, job_description: str) -> dict:
    """
    Generates the LLM explanation, then runs a lightweight NLP consistency
    check: does the explanation actually mention skills the candidate really
    has, avoid speculating about the scoring system's internals, and avoid
    contradicting the candidate's actual matched/missing skill counts? This
    is the Explainer Agent's NLP component - it doesn't trust the LLM's
    output blindly, it verifies it against the real structured data.
    """
    prompt = f"""You are explaining a candidate ranking to a recruiter.

Job description: {job_description}
Candidate skills: {', '.join(candidate.get('skills', []))}
Candidate's match score: {candidate.get('final_score', 'N/A')}

Write a 2-sentence, plain-English explanation of why this candidate received
this ranking, mentioning specific matched skills.

Base your explanation ONLY on the skills listed above and the score given -
describe what the skill overlap looks like, not why the scoring system or
algorithm behaved a certain way. Do NOT speculate about parsing errors,
algorithm failures, or system limitations - you have no visibility into how
the score was actually computed, so any claim about the scoring mechanics
would be an unverifiable guess, not a fact. If the score seems surprising
given the skill list, simply describe the skills and score as given without
inventing a reason for the apparent mismatch.

Do not mention name, age, or gender - this data was not provided and should
never be referenced."""
    explanation_text = call_llm(prompt)

    # --- NLP consistency check (no LLM needed - fast, deterministic) ---
    actual_skills = [s.lower() for s in candidate.get("skills", [])]
    missing_skills = candidate.get("missing_skills", []) or []
    final_score = candidate.get("final_score")
    explanation_lower = explanation_text.lower()

    mentioned_real_skills = [s for s in actual_skills if s in explanation_lower]
    skills_grounded = len(mentioned_real_skills) > 0 if actual_skills else True

    mechanics_speculation = any(p in explanation_lower for p in SCORE_MECHANICS_PHRASES)

    implies_low_match = any(p in explanation_lower for p in LOW_MATCH_PHRASES)
    matched_count = len(actual_skills)
    missing_count = len(missing_skills)
    strong_actual_match = (
        (final_score is not None and final_score >= 0.5) or
        (matched_count > 0 and missing_count <= max(1, matched_count // 4))
    )
    contradicts_data = implies_low_match and strong_actual_match

    consistency_ok = skills_grounded and not mechanics_speculation and not contradicts_data

    notes = []
    if not skills_grounded:
        notes.append("Explanation does not clearly reference the candidate's actual listed skills.")
    if mechanics_speculation:
        notes.append("Explanation speculates about WHY the scoring system/algorithm behaved a "
                      "certain way (e.g. parsing or system failures) - this can't be verified from "
                      "the data available and should be treated as unreliable commentary, not fact.")
    if contradicts_data:
        notes.append(f"Explanation implies a low/poor match, but the actual data shows "
                      f"{matched_count} matched vs {missing_count} missing skills "
                      f"(final_score={final_score}) - this looks like a contradiction worth a "
                      f"manual check before trusting the explanation.")

    note = " ".join(notes) if notes else (
        "Explanation references at least one of the candidate's actual skills and is "
        "consistent with the scoring data - grounded in real data."
    )

    return {
        "explanation": explanation_text,
        "consistency_check": {
            "passed": consistency_ok,
            "skills_confirmed_in_text": mentioned_real_skills,
            "note": note,
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
    """
    Older, skill-list-only version - kept for backward compatibility with
    any caller that only has scores/skills on hand and no stored CV text
    (e.g. sample/test data that never went through the Reader Agent).
    Prefer generate_interview_questions_from_cv() below whenever real
    anonymized CV text is available - it produces far more specific,
    genuinely CV-grounded questions.
    """
    prompt = f"""Generate 5 tailored interview questions for this candidate.

Job description: {job_description}
Candidate skills: {', '.join(candidate.get('skills', []))}

Include at least one question probing a specific skill they claim, and one
addressing any gap between their skills and the job requirements. Return
just the 5 questions, one per line, no extra commentary."""
    response = call_llm(prompt)
    return [q.strip("- ").strip() for q in response.split("\n") if q.strip()]


def generate_interview_questions_from_cv(job_title: str, job_description: str,
                                          cv_text: str, matched_skills: List[str],
                                          missing_skills: List[str],
                                          projects_text: Optional[str] = None) -> List[str]:
    """
    The CV-grounded version: instead of only knowing a flat skill list, this
    gives the LLM the candidate's actual (PII-stripped) CV text - project
    names, tools used, metrics claimed, responsibilities described - and
    asks for questions that reference those SPECIFIC details. This is what
    makes the questions "really based on the CV" rather than generic
    per-skill boilerplate that could apply to any candidate with the same
    skill list.

    When projects_text is available (extracted separately by the Reader
    Agent via extract_projects_section), it's surfaced to the LLM as its
    own clearly-labeled block and the prompt REQUIRES a guaranteed number
    of questions tied to it - this is what makes project-specific
    questions reliable rather than hoping the LLM notices project details
    on its own inside a much larger, undifferentiated CV text block.

    Falls back to the skill-list-only prompt if no CV text was stored for
    this candidate (e.g. they were added via the sample-data path rather
    than a real uploaded CV).
    """
    if not cv_text:
        return generate_interview_questions(
            {"skills": matched_skills, "missing_skills": missing_skills}, job_description
        )

    projects_block = ""
    project_instruction = (
        "- At least 3 questions must reference a SPECIFIC project, tool, metric, or "
        "responsibility actually mentioned in the CV text above - be concrete "
        "(e.g. ask them to walk through a named project's approach, a specific "
        "result they claimed, or a tool they say they used), not generic."
    )
    if projects_text:
        projects_block = f"""

The candidate's PROJECTS section, extracted separately (this is the most
important source for your questions - draw heavily on it):

---
{projects_text}
---"""
        project_instruction = (
            "- At least 3 questions MUST each be about a DIFFERENT project from the "
            "PROJECTS section above - name the specific project (or its core "
            "technology/technique) directly in the question, and ask about a "
            "concrete choice they made, a result they claimed, a challenge they "
            "likely faced, or a trade-off in their approach. Do not write a "
            "generic question that could apply to any project - it must be clear "
            "from the question text which specific project you mean."
        )

    prompt = f"""You are preparing interview questions for a hiring panel.

Job title: {job_title}
Job description: {job_description}

Below is the candidate's CV text (personal identifying details like name,
email, and phone have already been removed - do not attempt to guess or
reference who this is, only what they wrote about their work):

---
{cv_text}
---{projects_block}

Matched skills (present in both the CV and the job requirements): {', '.join(matched_skills) or 'none recorded'}
Missing skills (in the job requirements but not clearly in the CV): {', '.join(missing_skills) or 'none recorded'}

Write 6 interview questions for this specific candidate:
{project_instruction}
- At least 1 question should probe a gap between the missing skills and
  the role, framed constructively (e.g. how they'd approach ramping up).
- Keep each question to one or two sentences.
- Do not mention or guess the candidate's name, age, gender, or any other
  personal identifying detail.

Return just the 6 questions, one per line, numbered 1-6, no extra
commentary before or after the list."""

    response = call_llm(prompt)
    questions = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]
    # Strip any leading "1. " / "1)" numbering the LLM added, since the
    # PDF/dashboard renders its own numbering.
    cleaned = []
    for q in questions:
        stripped = q
        for prefix_len in (2, 3, 4):
            if len(stripped) > prefix_len and stripped[:prefix_len].rstrip(". )").isdigit():
                stripped = stripped[prefix_len:].lstrip(". )").strip()
                break
        cleaned.append(stripped)
    return cleaned


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
init_db()  # creates tables if they don't exist yet - safe to call every startup


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
    job_id: Optional[int] = None   # if provided, results update the saved ranking rows


class InterviewQuestionsRequest(BaseModel):
    candidate_id: str
    job_id: int


@app.get("/health")
def health():
    return {"status": "ok", "agent": "explainer", "llm_configured": model is not None}


@app.post("/explain", dependencies=[Depends(verify_api_key)])
def explain_endpoint(request: ExplainRequest):
    """
    Generates an explanation + skills-gap feedback for each candidate.
    If job_id is provided, the explanation and consistency-check result are
    saved back onto that candidate's ranking row in the database, so the
    Dashboard can read the full picture (score + explanation together).
    """
    results = []
    for c in request.candidates:
        c_dict = c.dict()
        explanation_result = generate_explanation(c_dict, request.job_description)
        skills_gap = generate_skills_gap_feedback(c_dict, request.job_description)

        results.append({
            "candidate_id": c.candidate_id,
            "explanation": explanation_result["explanation"],
            "consistency_check": explanation_result["consistency_check"],
            "verifiable_data": explanation_result["verifiable_data"],
            "skills_gap_feedback": skills_gap,
        })

        if request.job_id is not None:
            save_ranking(
                c.candidate_id, request.job_id,
                c.semantic_score, c.keyword_score, c.final_score,
                matched_skills=c.skills, missing_skills=c.missing_skills,
                explanation=explanation_result["explanation"],
                consistency_check_passed=explanation_result["consistency_check"]["passed"],
            )

    return {"explanations": results}


@app.post("/audit", dependencies=[Depends(verify_api_key)])
def audit_endpoint(request: ExplainRequest):
    """
    Runs the bias audit across all ranked candidates. If job_id is provided,
    the result is logged to bias_audit_log for a permanent audit trail.
    """
    candidates_as_dicts = [c.dict() for c in request.candidates]
    result = audit_bias(candidates_as_dicts)

    if request.job_id is not None and "avg_score_with_employment_gap" in result:
        log_bias_audit(
            request.job_id, result["flagged"],
            result["avg_score_with_employment_gap"],
            result["avg_score_without_employment_gap"],
            result["note"],
        )
    return result


@app.post("/interview-questions", dependencies=[Depends(verify_api_key)])
def interview_questions_endpoint(request: InterviewQuestionsRequest):
    """
    Called when a recruiter accepts a specific candidate for interview.
    Looks everything up from the shared database itself - the job's
    description, the candidate's matched/missing skills from their
    ranking, and (crucially) their anonymized CV text - rather than
    requiring the caller to assemble and pass all of that manually. This
    is what makes the questions genuinely grounded in the real CV instead
    of whatever skill list happens to get passed in.
    """
    job = get_job(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"No job found with job_id={request.job_id}")

    candidate = get_candidate(request.candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail=f"No candidate found with candidate_id={request.candidate_id}")

    rankings = get_rankings_for_job(request.job_id)
    ranking = next((r for r in rankings if r["candidate_id"] == request.candidate_id), None)
    matched_skills = ranking["matched_skills"] if ranking else candidate.get("skills", [])
    missing_skills = ranking["missing_skills"] if ranking else []

    questions = generate_interview_questions_from_cv(
        job_title=job["title"],
        job_description=job["description"],
        cv_text=candidate.get("cv_text_anonymized") or "",
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        projects_text=candidate.get("projects_text") or None,
    )

    return {
        "candidate_id": request.candidate_id,
        "job_id": request.job_id,
        "job_title": job["title"],
        "questions": questions,
        "cv_based": bool(candidate.get("cv_text_anonymized")),
        "projects_found": bool(candidate.get("projects_text")),
    }


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

    print("\n=== Interview questions (top candidate, skill-list-only fallback) ===")
    print(generate_interview_questions(sample_candidates[0], job_description))