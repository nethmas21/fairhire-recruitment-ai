import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database"))
import sqlite3

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "fairhire.db")
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

print("=== All jobs ===")
for row in conn.execute("SELECT job_id, title, status FROM jobs"):
    print(dict(row))

print("\n=== All candidates (across all jobs) ===")
for row in conn.execute("SELECT candidate_id, job_id, cv_text_anonymized IS NOT NULL AS has_cv_text, projects_text IS NOT NULL AS has_projects FROM candidates"):
    print(dict(row))

conn.close()
#commit
#commit