"""
FairHire - Matcher Agent
--------------------------
Job: Rank candidates against a job description using HYBRID RETRIEVAL:
  - Semantic similarity (sentence embeddings) - understands meaning/context
  - Keyword overlap - a DIRECT, verifiable ratio of required skills the
    candidate actually has, out of all required skills for the role
Final score = weighted average of both.

NOTE ON A FIX APPLIED HERE: the keyword component originally used BM25
scored against the full job-description sentence (including filler words
like "looking", "for", "a"), with a tokenizer bug that silently truncated
every multi-word skill ("power bi" -> just "power") in the per-candidate
corpus. This produced final scores that could look badly mismatched
against the matched/missing skill lists shown to the recruiter right next
to them - e.g. a candidate matching 12 of 13 required skills scoring only
~0.35. The keyword score is now simply (# required skills matched) / (#
required skills total) - the same numbers already shown in the
matched/missing lists, so the score is always numerically consistent with
what the recruiter can see and verify themselves.

Also implements the "agentic" behavior from the master spec: if too few
candidates clear the minimum threshold, the agent autonomously widens its
matching criteria and re-ranks once before returning results.

Setup:
    pip install sentence-transformers numpy fastapi uvicorn

Run standalone (sample data):
    python matcher_agent.py

Run as an API other agents can call:
    uvicorn matcher_agent:app --port 8002 --reload
    Then POST to http://localhost:8002/match
"""

import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"

import json
import re
import sys
import os
import warnings

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning)

# Import the shared database module (lives in ../database relative to this file)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database"))
from db import init_db, save_candidate, save_ranking, get_job
from shared.auth import verify_api_key

from typing import List, Optional
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

API_KEY = os.environ.get("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
    llm_model = genai.GenerativeModel("gemini-3.5-flash-lite")
else:
    llm_model = None


def call_llm(prompt: str) -> str:
    if llm_model is None:
        return None
    try:
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return None


def tokenize(text):
    """Lowercase and strip punctuation so 'Python,' matches 'python'."""
    return re.findall(r"[a-z0-9\+\#]+", text.lower())

# --- Config ---
SEMANTIC_WEIGHT = 0.5          # how much weight semantic similarity gets
KEYWORD_WEIGHT = 0.5           # how much weight required-skill overlap gets
# Rebalanced to an even 50/50 split (previously 0.6/0.4 favoring semantic
# similarity). Required-skill overlap is the objective, directly verifiable
# signal - a candidate either has a required skill or doesn't - so giving
# it at least equal weight to the semantic similarity score (computed over
# a bag of skill-keywords rather than a natural sentence, which sentence
# embedding models weren't trained to compare as reliably) produces scores
# that align much better with what the matched/missing lists actually show.
# This is a tunable policy choice - feel free to adjust for your report.
MIN_MATCH_THRESHOLD = 0.5      # minimum final_score to be considered a real match
MIN_CANDIDATES_REQUIRED = 3    # if fewer than this clear the threshold, widen search

model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, good enough for a demo


def compute_semantic_scores(job_description, candidates):
    """Embed the job description and each candidate's skill text, compare similarity."""
    job_embedding = model.encode(job_description, convert_to_tensor=True)

    candidate_texts = [
        " ".join(c["skills"] + c.get("organizations", []))
        for c in candidates
    ]
    candidate_embeddings = model.encode(candidate_texts, convert_to_tensor=True)

    scores = util.cos_sim(job_embedding, candidate_embeddings)[0]
    return [float(s) for s in scores]


def expand_skills_with_llm(job_description, known_skills):
    """
    LLM component of the Matcher Agent: query expansion. If a job says
    "data visualization" but never says "Power BI" explicitly, a candidate
    whose CV literally says "Power BI" would otherwise be under-counted on
    keyword matching, even though they clearly have the skill the job
    wants. The LLM suggests which of our KNOWN skill keywords are closely
    related to what the job actually describes, so the required-skill list
    can credit candidates for related skills too - not just literal
    keyword overlap in the job description text.

    Falls back to an empty list (no expansion) if the LLM isn't configured
    or fails - matching still works correctly without this, just less
    generously on paraphrased skills.
    """
    if llm_model is None:
        return []

    prompt = (
        f"Job description: {job_description}\n\n"
        f"From this list of skills, which ones are CLOSELY RELATED to what "
        f"this job needs, even if not explicitly named in the description? "
        f"Skills list: {', '.join(known_skills)}\n"
        f"Reply with a comma-separated list only, no explanation. If none "
        f"apply, reply 'none'."
    )
    result = call_llm(prompt)
    if not result or result.strip().lower() == "none":
        return []
    suggested = [s.strip().lower() for s in result.split(",") if s.strip()]
    # Only keep suggestions that are actually in our known skill list -
    # guards against the LLM inventing skills we don't track.
    return [s for s in suggested if s in known_skills]


def extract_required_skills(job_description):
    """
    Extraction of likely required skills from the job description: first
    a fast, exact word-boundary match (no LLM needed), THEN an LLM query
    expansion pass adds closely-related skills the JD implies but never
    states outright (e.g. "data visualization" implying "power bi").
    """
    known_skills = [
        "python", "sql", "java", "javascript", "r", "excel", "power bi",
        "tableau", "machine learning", "data analysis", "pandas", "numpy",
        "scikit-learn", "tensorflow", "aws", "azure", "docker", "git",
    ]
    jd_lower = job_description.lower()
    found = []
    for skill in known_skills:
        # \b = word boundary, so "r" matches "R" but not the "r" inside "for"
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, jd_lower):
            found.append(skill)

    expanded = expand_skills_with_llm(job_description, known_skills)
    for skill in expanded:
        if skill not in found:
            found.append(skill)

    return found


def rank_candidates(job_description, candidates, relaxed=False):
    """
    Core ranking function. If relaxed=True, this represents the "widened
    search" pass - in a fuller version this would relax a specific constraint
    (e.g. treat related skills as synonyms); here we keep the scoring the
    same but lower the threshold, which is the simplest honest version of
    "widening" for a student project - document this choice in your report.
    """
    semantic_scores = compute_semantic_scores(job_description, candidates)
    required_skills = extract_required_skills(job_description)

    results = []
    for cand, sem in zip(candidates, semantic_scores):
        candidate_skills_lower = [s.lower() for s in cand.get("skills", [])]
        matched = [s for s in required_skills if s in candidate_skills_lower]
        missing = [s for s in required_skills if s not in candidate_skills_lower]

        # Keyword score = direct required-skill overlap ratio. This is the
        # SAME matched/missing data shown to the recruiter, so the score is
        # always numerically consistent with what they can see and verify -
        # a candidate matching 12 of 13 required skills always gets a
        # keyword_score of exactly 12/13 (~0.923), never a disconnected
        # number from an opaque retrieval-ranking formula.
        keyword_score = len(matched) / len(required_skills) if required_skills else 0.0

        final_score = (SEMANTIC_WEIGHT * sem) + (KEYWORD_WEIGHT * keyword_score)

        results.append({
            "candidate_id": cand["candidate_id"],
            "semantic_score": round(sem, 3),
            "keyword_score": round(keyword_score, 3),
            "final_score": round(final_score, 3),
            # --- New: feeds the Explainer Agent's verifiable_data field ---
            "matched_skills": matched,
            "missing_skills": missing,
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


def match(job_description, candidates, job_id=None):
    """
    Main entry point. Implements the agentic threshold-widening behavior
    from the master spec: if too few candidates clear MIN_MATCH_THRESHOLD,
    automatically re-rank with a lower threshold once.

    If job_id is provided, each candidate and their ranking is saved to the
    shared database - this is what lets the Explainer Agent and Dashboard
    read this data later, instead of it only existing in this response.
    """
    results = rank_candidates(job_description, candidates)

    qualifying = [r for r in results if r["final_score"] >= MIN_MATCH_THRESHOLD]

    if len(qualifying) < MIN_CANDIDATES_REQUIRED:
        # Agentic decision: widen the search rather than return too few results
        widened_threshold = MIN_MATCH_THRESHOLD - 0.15
        qualifying = [r for r in results if r["final_score"] >= widened_threshold]
        note = (f"Fewer than {MIN_CANDIDATES_REQUIRED} candidates met the "
                f"standard threshold ({MIN_MATCH_THRESHOLD}); search was "
                f"automatically widened to {widened_threshold}.")
    else:
        note = "Standard threshold applied - no widening needed."

    if job_id is not None:
        candidates_by_id = {c["candidate_id"]: c for c in candidates}
        for r in results:
            cand = candidates_by_id[r["candidate_id"]]
            # NOTE: this call deliberately omits cv_text_anonymized/
            # projects_text - it only ever has skills/organizations
            # available here. save_candidate() in db.py now preserves
            # whatever the Reader Agent already saved for those two
            # fields instead of overwriting them with NULL when they're
            # not passed - this used to silently wipe out CV text and
            # extracted project data on every re-match.
            save_candidate(r["candidate_id"], job_id, skills=cand.get("skills", []),
                            organizations=cand.get("organizations", []))
            save_ranking(r["candidate_id"], job_id, r["semantic_score"],
                         r["keyword_score"], r["final_score"],
                         matched_skills=r["matched_skills"],
                         missing_skills=r["missing_skills"])

    return {
        "ranked_candidates": results,
        "qualifying_candidates": qualifying,
        "widening_note": note,
    }


"""
------------------------------------------------------------
FastAPI wrapper - this is what makes the Matcher Agent a real
"agent" other agents can call over HTTP, instead of just a script.
------------------------------------------------------------
"""

app = FastAPI(title="FairHire Matcher Agent")
init_db()  # creates tables if they don't exist yet - safe to call every startup


class Candidate(BaseModel):
    candidate_id: str
    skills: List[str]
    organizations: Optional[List[str]] = []


class MatchRequest(BaseModel):
    job_description: str
    candidates: List[Candidate]
    job_id: Optional[int] = None   # if provided, results are saved to the database


@app.get("/health")
def health():
    """Quick check that the agent is alive - useful when debugging integration."""
    return {"status": "ok", "agent": "matcher"}


@app.post("/match", dependencies=[Depends(verify_api_key)])
def match_endpoint(request: MatchRequest):
    """
    This is the endpoint the Dashboard (or Reader Agent) calls. Requires
    the shared API key (see auth.py).
    Example call from another agent:

        import requests
        response = requests.post(
            "http://localhost:8002/match",
            json={"job_description": job_text, "candidates": candidate_list, "job_id": 1},
            headers={"X-API-Key": API_KEY},
        )
        ranked = response.json()
    """
    candidates_as_dicts = [c.dict() for c in request.candidates]
    return match(request.job_description, candidates_as_dicts, job_id=request.job_id)


if __name__ == "__main__":
    # Running "python matcher_agent.py" still does the standalone sample-data
    # test - useful for quick checks without needing to start the API server.
    job_description = "Looking for a Data Analyst with Python, SQL, and Power BI experience"

    sample_candidates = [
        {"candidate_id": "CAND_001", "skills": ["python", "sql", "power bi"], "organizations": ["Company A"]},
        {"candidate_id": "CAND_002", "skills": ["sql", "excel"], "organizations": ["Company B"]},
        {"candidate_id": "CAND_003", "skills": ["java", "project management"], "organizations": ["Company C"]},
    ]

    output = match(job_description, sample_candidates)
    print(json.dumps(output, indent=2))