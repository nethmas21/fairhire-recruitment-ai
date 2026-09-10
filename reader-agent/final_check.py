import sqlite3, os

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "fairhire.db")
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT candidate_id, job_id, skills, cv_text_anonymized, projects_text FROM candidates WHERE job_id = 20"
).fetchall()

if not rows:
    print("NO ROWS FOUND for job_id=20")
else:
    for row in rows:
        d = dict(row)
        print("candidate_id:", d["candidate_id"])
        print("skills:", d["skills"])
        print("cv_text_anonymized:", d["cv_text_anonymized"])
        print("projects_text:", d["projects_text"])
        print("-----")

conn.close()