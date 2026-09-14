"""
FairHire - Reader Agent
------------------------
Job: Take a CV file, extract skills/experience/education using NLP (NER),
strip out personal details (name, gender, age) into a separate protected
vault, and save the clean, anonymized profile to the shared database.

Setup:
    pip install spacy pdfplumber python-docx fastapi uvicorn python-dotenv
    python -m spacy download en_core_web_sm

Run standalone (sample data):
    python reader_agent.py path/to/cv.pdf <job_id>

Run as an API:
    uvicorn reader_agent:app --port 8001 --reload
"""

import sys
import os
import re
import json
from collections import Counter
import warnings
import tempfile

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning)

# Import the shared database module
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database"))
from db import init_db, save_candidate, save_pii
from shared.auth import verify_api_key


import spacy
import pdfplumber
from docx import Document
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from pydantic import BaseModel

nlp = spacy.load("en_core_web_sm")

API_KEY = os.environ.get("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-3.5-flash-lite")
else:
    model = None


def call_llm(prompt: str) -> str:
    if model is None:
        return None
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return None

# Skill keyword list - expand this with skills relevant to your target roles.
SKILL_KEYWORDS = [
    "python", "sql", "java", "javascript", "r", "excel", "power bi",
    "tableau", "machine learning", "data analysis", "pandas", "numpy",
    "scikit-learn", "tensorflow", "aws", "azure", "docker", "git",
    "communication", "project management", "leadership"
]

# Common CV section headers used to find the boundaries of the "Projects"
# section specifically - so it can be extracted and handed to the
# interview-question generator directly, rather than hoping the LLM
# notices project details buried somewhere in a large block of CV text.
KNOWN_CV_HEADERS = [
    "professional summary", "summary", "objective", "profile",
    "experience", "work experience", "employment history", "internship",
    "education", "academic background",
    "projects", "personal projects", "academic projects", "key projects",
    "notable projects", "selected projects",
    "skills", "technical skills", "core competencies",
    "certifications", "certificates", "licenses",
    "awards", "achievements", "honors",
    "publications", "research",
    "references", "contact", "hobbies", "interests",
    "languages", "volunteer", "volunteering", "extracurricular",
]
PROJECT_HEADER_KEYWORDS = ("project",)  # matches "projects", "key projects", etc.


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "as", "by", "from", "it", "its",
    "their", "his", "her", "they", "he", "she", "i", "we", "you", "your",
    "our", "will", "would", "can", "could", "should", "has", "have", "had",
    "not", "no", "so", "if", "then", "than", "also", "into", "over",
    "such", "which", "who", "what", "when", "where", "while", "about",
    "each", "other", "more", "most", "some", "any", "all", "there", "here",
}


def extractive_summarize(text, num_sentences=4):
    """
    NLP TECHNIQUE: classic frequency-based extractive summarization -
    no LLM required, always available. Splits the text into sentences,
    scores each sentence by the summed frequency of its non-stopword
    terms (normalized by sentence length so long sentences don't win
    purely on word count), and returns the top-scoring sentences in
    their ORIGINAL document order (not ranked order) so the result still
    reads coherently. This is the same family of technique as classic
    extractive summarizers (e.g. Luhn's algorithm) - a genuine NLP
    method distinct from named-entity recognition and distinct from
    LLM prompting.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if not sentences:
        return ""
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    word_freq = Counter()
    for sentence in sentences:
        for word in re.findall(r"[a-z']+", sentence.lower()):
            if word not in STOPWORDS and len(word) > 2:
                word_freq[word] += 1

    scored = []
    for i, sentence in enumerate(sentences):
        words = re.findall(r"[a-z']+", sentence.lower())
        if not words:
            continue
        score = sum(word_freq.get(w, 0) for w in words) / len(words)
        scored.append((i, score))

    top = sorted(scored, key=lambda x: x[1], reverse=True)[:num_sentences]
    top_indices_in_order = sorted(i for i, _ in top)
    return " ".join(sentences[i] for i in top_indices_in_order)


def generate_cv_summary(anonymized_text):
    """
    NLP Summarization feature: produces a short, recruiter-friendly
    summary of the candidate's (already PII-stripped) CV.

    Two layers, following the same pattern used elsewhere in this
    pipeline (fast deterministic pass first, LLM refinement second):
      1. extractive_summarize() above - a genuine NLP technique that
         always works, with or without an LLM configured.
      2. LLM abstractive polish - rewrites the extracted sentences into
         a smooth 2-3 sentence summary. Falls back to the raw extractive
         result if no LLM is configured or the call fails, so this
         feature never blocks CV processing.
    """
    extractive = extractive_summarize(anonymized_text, num_sentences=4)
    if not extractive:
        return ""

    if model is None:
        return extractive

    prompt = (
        "Rewrite the following sentences extracted from a CV into a "
        "smooth, professional 2-3 sentence summary a recruiter can read "
        "in a few seconds to understand this candidate's background. "
        "Do not invent any detail not present in the text. Do not guess "
        "or mention a name - personal details have already been removed."
        f"\n\nExtracted sentences:\n{extractive}"
    )
    result = call_llm(prompt)
    return result or extractive


def extract_projects_with_llm(anonymized_text):
    """
    Reads the WHOLE CV and asks the LLM to identify specific projects or
    significant pieces of technical/academic/professional work the
    candidate describes - regardless of whether there's a labeled
    "Projects" heading at all. Real CVs are inconsistent about this: some
    describe a project inline inside a work-experience bullet, some have
    no headings at all, some use a heading extract_projects_section()
    below wouldn't recognize. Rather than trying to pattern-match every
    possible layout, this lets the LLM actually reason about the content.

    Falls back to None if no LLM is configured, the call fails, or the
    LLM genuinely finds nothing - the caller then tries the heading-based
    extract_projects_section() as a deterministic backstop.
    """
    if model is None:
        return None

    prompt = (
        "Read the following CV/resume text (personal identifying details "
        "have already been removed). Identify any specific PROJECTS, "
        "systems, or significant pieces of technical, academic, or "
        "professional work the candidate describes - whether or not "
        "they appear under a labeled 'Projects' heading. This includes "
        "work described inline inside a job/experience entry, a summary "
        "paragraph, or anywhere else in the text, as long as it's a real, "
        "specific piece of work (not just a bare skill or tool mention "
        "with no project attached).\n\n"
        "For each one you find, write exactly one line in this form:\n"
        "<short project name> - <1-2 sentence description of what they "
        "built or did, including any specific technology, approach, or "
        "result they mentioned>\n\n"
        "If you genuinely find no specific projects or pieces of work "
        "described, reply with exactly: none\n\n"
        f"CV text:\n{anonymized_text}"
    )
    result = call_llm(prompt)
    if not result or result.strip().lower() == "none":
        return None
    return result.strip()


def extract_projects_section(text):
    """
    Pulls out just the "Projects" section of a CV (under whatever heading
    variant the candidate used - "Projects", "Key Projects", "Academic
    Projects", etc.), so interview questions can be guaranteed to
    reference the candidate's own named projects specifically, instead of
    relying on the LLM to notice them inside a long undifferentiated block
    of CV text.

    Returns None if no recognizable Projects heading is found - the
    question generator falls back to using the full CV text in that case.
    """
    lines = text.split("\n")

    def _matched_header(line):
        stripped = line.strip().strip(":").lower()
        if len(stripped) > 40:
            return None
        for header in KNOWN_CV_HEADERS:
            if stripped == header:
                return header
        return None

    header_indices = []  # (line_index, header_text)
    for i, line in enumerate(lines):
        header = _matched_header(line)
        if header:
            header_indices.append((i, header))

    project_start = None
    for i, header in header_indices:
        if any(kw in header for kw in PROJECT_HEADER_KEYWORDS):
            project_start = i
            break

    if project_start is None:
        return None

    # Find the next header after the projects one to know where it ends.
    project_end = len(lines)
    for i, _ in header_indices:
        if i > project_start:
            project_end = i
            break

    section_lines = lines[project_start + 1:project_end]
    section_text = "\n".join(line for line in section_lines if line.strip())
    return section_text.strip() or None


def extract_text(file_path):
    """Extract raw text from a PDF or DOCX CV."""
    if file_path.lower().endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text
    elif file_path.lower().endswith(".docx"):
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Only .pdf and .docx files are supported")


def extract_skills(text):
    """Match known skill keywords against the CV text, whole-word only
    (so 'r' doesn't match inside 'for', same fix as the Matcher Agent)."""
    text_lower = text.lower()
    found = []
    for skill in SKILL_KEYWORDS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


def verify_organizations_with_llm(candidate_orgs, known_skills):
    """
    LLM component of the Reader Agent: spaCy's NER sometimes misclassifies
    skill/tool names (like "SQL" or "Power BI") as organizations, since a
    general-purpose NER model isn't trained specifically on CVs. Rather
    than trusting spaCy blindly, we ask the LLM to double-check the
    ambiguous cases - this is the "fallback verification" role the LLM
    plays here, while spaCy still does the fast, main extraction pass.

    Falls back to spaCy's original list unchanged if no LLM is configured,
    so this never blocks the pipeline.
    """
    if model is None or not candidate_orgs:
        return candidate_orgs

    prompt = (
        f"From this list, return ONLY the ones that are genuinely company "
        f"or organization names (not skills, tools, or technologies): "
        f"{', '.join(candidate_orgs)}. "
        f"For reference, here are known skill/tool names to exclude if "
        f"they appear: {', '.join(known_skills)}. "
        f"Reply with a comma-separated list only, no explanation. If none "
        f"are real organizations, reply with 'none'."
    )
    result = call_llm(prompt)
    if not result or result.strip().lower() == "none":
        return [] if result else candidate_orgs  # None on error = keep original, "none" = empty

    verified = [org.strip() for org in result.split(",") if org.strip()]
    # Only keep verified names that were actually in the original list -
    # guards against the LLM inventing something not in the CV at all.
    return [org for org in candidate_orgs if org in verified]


def extract_entities(text):
    """Use spaCy NER to pull out organizations (work experience) and dates,
    then verify the organizations with the LLM to catch skill/tool names
    spaCy sometimes misclassifies (e.g. "SQL", "Power BI")."""
    doc = nlp(text)
    orgs_raw = list({ent.text for ent in doc.ents if ent.label_ == "ORG"})
    dates = list({ent.text for ent in doc.ents if ent.label_ == "DATE"})
    orgs_verified = verify_organizations_with_llm(orgs_raw, SKILL_KEYWORDS)
    return {"organizations": orgs_verified, "organizations_before_llm_check": orgs_raw, "dates": dates}


# Words that commonly appear as CV section/field labels or tool names near
# the top of a document, which spaCy's small NER model sometimes misreads
# as a PERSON entity (e.g. "Email:", "Postman" the API-testing tool). None
# of these are ever a real candidate name, so they're always excluded.
NAME_EXCLUDE_TERMS = [
    "email", "phone", "contact", "address", "linkedin", "github", "portfolio",
    "objective", "summary", "profile", "resume", "cv", "curriculum vitae",
    "postman", "swagger", "jira", "confluence", "slack", "notion", "figma",
]


def _is_skill_like(candidate_text, known_skills):
    """
    True if candidate_text is (or contains, as a WHOLE WORD) a known
    skill/tool term or common CV field label, rather than a real person
    name - e.g. spaCy sometimes tags "Machine Learning" or bare label
    words like "Email" as PERSON, the same misclassification problem
    already handled for organizations above.

    Uses word-boundary matching (same fix already applied in
    extract_skills) rather than plain substring matching - a naive
    substring check would wrongly flag names like "Weerasinghe" as
    skill-like just because the single-letter skill "r" happens to
    appear inside the word.
    """
    normalized = candidate_text.strip().lower()
    all_excluded = known_skills + NAME_EXCLUDE_TERMS
    return any(re.search(r"\b" + re.escape(term) + r"\b", normalized) for term in all_excluded)


def _regex_labeled_name(text):
    """
    Deterministic first pass: many CVs (especially template-based ones)
    literally have a line like "Name: John Doe" or "Full Name - John Doe".
    Checking for this explicit label avoids relying on NER at all when the
    document just tells us directly.
    """
    match = re.search(r"(?im)^\s*(?:full\s*name|name)\s*[:\-]\s*(.+)$", text)
    if match:
        candidate = match.group(1).strip()
        # Guard against the label itself being blank or absurdly long
        # (e.g. it accidentally matched across an unrelated paragraph).
        if candidate and len(candidate) < 60 and not _is_skill_like(candidate, SKILL_KEYWORDS):
            return candidate
    return None


def _format_name_case(name):
    """
    Many CVs put the candidate's name in ALL CAPS as a header (e.g.
    "NETHMA WEERASINGHE"), which reads oddly in a "Dear ___," greeting -
    and which spaCy's NER often fails to recognize as a PERSON at all,
    since it relies heavily on standard capitalization patterns. Convert
    all-caps or all-lowercase names to Title Case for display; leave
    already-mixed-case names untouched so we don't mangle something like
    "McDonald" or "O'Brien".
    """
    if name.isupper() or name.islower():
        return name.title()
    return name


def _first_line_name_heuristic(text):
    """
    Deterministic second pass (after the explicit "Name:" label check):
    the very first non-empty line of a CV is, in the overwhelming
    majority of real CVs, just the candidate's name with no label at all
    - often in a larger font or ALL CAPS, which is exactly what trips up
    spaCy's NER and is handled explicitly here rather than left to chance.

    Only accepts the line if it looks plausibly like a name: short, no
    email/phone/URL characters, and made up of a small number of
    alphabetic words - so a CV that opens with a tagline or section
    header instead of a name is correctly rejected rather than misused.
    """
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Reject anything with digits, @, or URL-like punctuation - a
        # name line never contains these.
        if re.search(r"[\d@/]", line):
            return None
        if len(line) > 50:
            return None
        words = line.split()
        if not (1 <= len(words) <= 5):
            return None
        if not all(re.match(r"^[A-Za-z'\-\.]+$", w) for w in words):
            return None
        if _is_skill_like(line, SKILL_KEYWORDS):
            return None
        return _format_name_case(line)
    return None


def _llm_verify_name(text_snippet, ner_candidates):
    """
    LLM component, mirroring verify_organizations_with_llm: asks the model
    directly what the candidate's name is, using only the top of the CV
    (where names live) plus whatever NER guessed, as a sanity check. Falls
    back to the best NER guess (or None) if no LLM is configured or the
    LLM call fails - this never blocks the pipeline.
    """
    if model is None:
        return ner_candidates[0] if ner_candidates else None

    prompt = (
        f"Here is the top portion of a CV/resume:\n\n{text_snippet}\n\n"
        f"What is the candidate's full name? Reply with ONLY the name, "
        f"nothing else. If you cannot confidently determine a real "
        f"person's name (for example, if only skills, tools, or section "
        f"labels like 'Email' or 'Objective' appear), reply with exactly: "
        f"unknown"
    )
    result = call_llm(prompt)
    if result is None:
        # The LLM call itself failed (quota/network/etc.), as opposed to
        # succeeding and saying "unknown" - fall back to NER rather than
        # silently returning no name at all.
        return ner_candidates[0] if ner_candidates else None
    if result.strip().lower() == "unknown":
        return None
    candidate = result.strip()
    if len(candidate) < 60 and not _is_skill_like(candidate, SKILL_KEYWORDS):
        return candidate
    return ner_candidates[0] if ner_candidates else None


def extract_person_name(text, known_skills):
    """
    Picks the candidate's real name using four layers, from most to
    least reliable:
      1. An explicit "Name: ..." label, if the CV states one directly.
      2. The first non-empty line of the document, if it looks plausibly
         like a name - this is where the overwhelming majority of CVs put
         the candidate's name (often in ALL CAPS as a header), which is
         exactly the case spaCy's NER tends to get wrong.
      3. LLM verification against the top of the document, for CVs where
         the name isn't cleanly on the first line (e.g. below a photo or
         logo placeholder).
      4. spaCy NER on the top lines, filtered against known skills/tools
         and common CV field labels, as a last resort if no LLM is
         configured.

    Returns None (rather than a wrong guess) if nothing usable is found -
    the notification templates already handle a missing name gracefully
    by falling back to the candidate ID.
    """
    labeled_name = _regex_labeled_name(text)
    if labeled_name:
        return labeled_name

    first_line_name = _first_line_name_heuristic(text)
    if first_line_name:
        return first_line_name

    top_lines = "\n".join(text.strip().split("\n")[:8])
    top_doc = nlp(top_lines)
    ner_candidates = [ent.text.strip() for ent in top_doc.ents if ent.label_ == "PERSON"]
    ner_candidates = [c for c in ner_candidates if c and not _is_skill_like(c, known_skills)]

    if model is not None:
        return _llm_verify_name(top_lines, ner_candidates)

    if ner_candidates:
        return ner_candidates[0]

    # Last-resort fallback: scan the whole document if nothing was found
    # up top and no LLM is available to verify.
    full_doc = nlp(text)
    all_candidates = [ent.text.strip() for ent in full_doc.ents if ent.label_ == "PERSON"]
    all_candidates = [c for c in all_candidates if c and not _is_skill_like(c, known_skills)]
    return all_candidates[0] if all_candidates else None


def extract_pii(text):
    """
    Pulls out personal identifying info so it can be stored SEPARATELY in
    the PII vault, never passed to the Matcher/Explainer agents. This is
    the "Prevention" layer of the Responsible AI fairness plan.
    """
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    phone_match = re.search(r"(\+?\d[\d\-\s]{7,}\d)", text)

    name = extract_person_name(text, SKILL_KEYWORDS)

    return {
        "name": name,
        "email": email_match.group(0) if email_match else None,
        "phone": phone_match.group(0) if phone_match else None,
        # age/gender are NOT extracted at all - deliberately never captured,
        # not even for the vault, since most CVs don't state them explicitly
        # and guessing them would itself be a bias risk.
    }


def detect_proxy_fields(dates):
    """
    Very simple employment-gap heuristic for the Explainer Agent's bias
    audit to use later. A real version would parse date ranges properly -
    this is a documented simplification for a student project.
    """
    return {"employment_gap_detected": len(dates) < 2}


def anonymize_for_matching(text):
    """Remove PII from the text BEFORE it's used for skill/entity extraction
    that feeds the Matcher/Explainer agents."""
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "[EMAIL REMOVED]", text)
    text = re.sub(r"(\+?\d[\d\-\s]{7,}\d)", "[PHONE REMOVED]", text)
    return text


def process_cv(file_path, candidate_id, job_id=None):
    """
    Main pipeline: extract -> split into (a) anonymized profile for
    matching and (b) PII for the separate vault -> save both to the
    database if job_id is provided.
    """
    raw_text = extract_text(file_path)
    pii = extract_pii(raw_text)
    anonymized_text = anonymize_for_matching(raw_text)

    skills = extract_skills(anonymized_text)
    entities = extract_entities(anonymized_text)
    proxy_fields = detect_proxy_fields(entities["dates"])

    profile = {
        "candidate_id": candidate_id,
        "skills": skills,
        "organizations": entities["organizations"],
        "proxy_fields": proxy_fields,
        # Deliberately NOT included here: name, email, phone, age, gender
    }

    if job_id is not None:
        # Cap the stored length - keeps the DB row and later LLM prompts
        # (interview-question generation) reasonably sized even for very
        # long CVs, while still covering far more than a flat skill list.
        cv_text_for_storage = anonymized_text[:8000]
        # Try the LLM first - it can find project descriptions regardless
        # of CV layout/heading conventions. Fall back to the heading-based
        # regex extraction only if no LLM is configured or it found
        # nothing, as a deterministic backstop.
        projects_text = extract_projects_with_llm(anonymized_text) or extract_projects_section(anonymized_text)
        cv_summary = generate_cv_summary(anonymized_text)
        save_candidate(candidate_id, job_id, skills=skills,
                        organizations=entities["organizations"],
                        proxy_fields=proxy_fields,
                        cv_text_anonymized=cv_text_for_storage,
                        projects_text=projects_text,
                        cv_summary=cv_summary)
        save_pii(candidate_id, name=pii["name"], email=pii["email"], phone=pii["phone"])

    return profile


# ---------------------------------------------------------------
# FastAPI wrapper
# ---------------------------------------------------------------
app = FastAPI(title="FairHire Reader Agent")
init_db()

# Security feature: input sanitization on uploaded CVs. Files are capped
# at 5MB (prevents a huge/malicious file from hanging text extraction or
# exhausting server resources), and the actual file CONTENT is checked
# against known magic bytes for PDF/DOCX - not just the filename extension,
# which anyone could rename to bypass a naive "ends with .pdf" check.
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


def validate_uploaded_file(file_bytes: bytes, filename: str):
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB",
        )

    ext = os.path.splitext(filename)[1].lower()
    is_pdf_content = file_bytes[:5] == b"%PDF-"
    is_docx_content = file_bytes[:4] == b"PK\x03\x04"  # DOCX files are zip archives

    if ext == ".pdf" and not is_pdf_content:
        raise HTTPException(status_code=400, detail="File has a .pdf extension but its content is not a valid PDF")
    if ext == ".docx" and not is_docx_content:
        raise HTTPException(status_code=400, detail="File has a .docx extension but its content is not a valid DOCX")
    if ext not in (".pdf", ".docx"):
        raise HTTPException(status_code=400, detail="Only .pdf and .docx files are supported")


@app.get("/health")
def health():
    return {"status": "ok", "agent": "reader"}


@app.post("/read-cv", dependencies=[Depends(verify_api_key)])
async def read_cv_endpoint(candidate_id: str, job_id: int, file: UploadFile = File(...)):
    """
    Accepts an uploaded CV file, processes it, and saves the anonymized
    profile + PII to the database. Returns the anonymized profile only.

    Requires the shared API key (see auth.py) and validates the uploaded
    file's actual content before processing it (see validate_uploaded_file
    above) - both are security features specifically required by the
    assignment brief (authentication + input sanitization).

    Uses Python's tempfile module (not a hardcoded "/tmp/" path) so this
    works correctly on Windows too - "/tmp/" only exists by default on
    Linux/Mac, which caused a real FileNotFoundError on Windows.
    """
    file_bytes = await file.read()
    validate_uploaded_file(file_bytes, file.filename)

    suffix = os.path.splitext(file.filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        result = process_cv(temp_path, candidate_id, job_id)
    finally:
        os.remove(temp_path)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reader_agent.py <path_to_cv> [job_id]")
        sys.exit(1)

    cv_path = sys.argv[1]
    job_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
    result = process_cv(cv_path, candidate_id="CAND_001", job_id=job_id)
    print(json.dumps(result, indent=2))
    #commit
    #coomit