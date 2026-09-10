import sqlite3, os
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "fairhire.db")
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

for row in conn.execute("SELECT candidate_id, job_id, skills, cv_text_anonymized, projects_text FROM candidates WHERE job_id = 20"):
    d = dict(row)
    print("candidate_id:", d["candidate_id"])
    print("skills:", d["skills"])
    print("cv_text_anonymized is None?", d["cv_text_anonymized"] is None)
    print("projects_text is None?", d["projects_text"] is None)
    print("---")
conn.close()