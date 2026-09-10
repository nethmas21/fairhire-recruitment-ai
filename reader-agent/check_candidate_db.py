import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database"))
from db import get_candidate, get_candidates_for_job

job_id = 1  # <-- change to the actual job_id you're testing with

print("=== All candidates for this job ===")
for c in get_candidates_for_job(job_id):
    print(c["candidate_id"], "| projects_text present:", bool(c.get("projects_text")))

print("\n=== Full detail for CAND_001 ===")
candidate = get_candidate("CAND_001")  # match to whichever ID you generated questions for
print("cv_text_anonymized present?", bool(candidate.get("cv_text_anonymized")))
print("projects_text present?", bool(candidate.get("projects_text")))
print(candidate.get("projects_text"))