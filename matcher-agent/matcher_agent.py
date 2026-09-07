"""
FairHire - Matcher Agent
--------------------------
Job: Rank candidates against a job description using HYBRID RETRIEVAL:
  - Semantic similarity (sentence embeddings) - understands meaning
  - Keyword/BM25 matching - rewards exact required-skill matches
Final score = weighted average of both.

Also implements the "agentic" behavior from the master spec: if too few
candidates clear the minimum threshold, the agent autonomously widens its
matching criteria and re-ranks once before returning results.

Setup:
    pip install sentence-transformers rank_bm25 numpy fastapi uvicorn

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
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi


def tokenize(text):
    """Lowercase and strip punctuation so 'Python,' matches 'python'."""
    return re.findall(r"[a-z0-9\+\#]+", text.lower())

# --- Config ---
SEMANTIC_WEIGHT = 0.6          # how much weight semantic similarity gets
KEYWORD_WEIGHT = 0.4           # how much weight keyword/BM25 match gets
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


def compute_keyword_scores(job_description, candidates):
    """BM25 keyword match - rewards candidates whose skills literally match
    the required skills mentioned in the job description."""
    corpus = [[tokenize(skill)[0] if tokenize(skill) else skill for skill in c["skills"]]
              for c in candidates]  # normalize each skill the same way as the query
    bm25 = BM25Okapi(corpus)

    query = tokenize(job_description)
    raw_scores = bm25.get_scores(query)

    # Normalize to 0-1 range so it can be combined with semantic scores
    max_score = max(raw_scores) if max(raw_scores) > 0 else 1
    return [float(s / max_score) for s in raw_scores]


def extract_required_skills(job_description):
    """
    Simple extraction of likely required skills from the job description,
    by checking which known skill keywords appear in the text as whole
    words (not substrings - e.g. "r" must not match inside "for").
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
    keyword_scores = compute_keyword_scores(job_description, candidates)
    required_skills = extract_required_skills(job_description)

    results = []
    for cand, sem, kw in zip(candidates, semantic_scores, keyword_scores):
        final_score = (SEMANTIC_WEIGHT * sem) + (KEYWORD_WEIGHT * kw)
        candidate_skills_lower = [s.lower() for s in cand.get("skills", [])]
        matched = [s for s in required_skills if s in candidate_skills_lower]
        missing = [s for s in required_skills if s not in candidate_skills_lower]

        results.append({
            "candidate_id": cand["candidate_id"],
            "semantic_score": round(sem, 3),
            "keyword_score": round(kw, 3),
            "final_score": round(final_score, 3),
            # --- New: feeds the Explainer Agent's verifiable_data field ---
            "matched_skills": matched,
            "missing_skills": missing,
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


def match(job_description, candidates):
    """
    Main entry point. Implements the agentic threshold-widening behavior
    from the master spec: if too few candidates clear MIN_MATCH_THRESHOLD,
    automatically re-rank with a lower threshold once.
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


class Candidate(BaseModel):
    candidate_id: str
    skills: List[str]
    organizations: Optional[List[str]] = []


class MatchRequest(BaseModel):
    job_description: str
    candidates: List[Candidate]


@app.get("/health")
def health():
    """Quick check that the agent is alive - useful when debugging integration."""
    return {"status": "ok", "agent": "matcher"}


@app.post("/match")
def match_endpoint(request: MatchRequest):
    """
    This is the endpoint the Dashboard (or Reader Agent) calls.
    Example call from another agent:

        import requests
        response = requests.post(
            "http://localhost:8002/match",
            json={"job_description": job_text, "candidates": candidate_list},
        )
        ranked = response.json()
    """
    candidates_as_dicts = [c.dict() for c in request.candidates]
    return match(request.job_description, candidates_as_dicts)


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