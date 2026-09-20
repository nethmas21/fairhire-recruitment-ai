"""
FairHire - Notification Agent
-------------------------------
Job: Generate the right email text for each outcome, actually send it via
SMTP to the candidate's real email address (extracted from their CV by the
Reader Agent), and let candidates confirm their own data-retention
preference via a link in the email instead of the email falsely claiming
a choice they never made.

Three email types:
    1. Shortlisted + interview invite (includes date/time)
    2. Early rejection (not shortlisted at all)
    3. Post-interview rejection

Setup:
    pip install fastapi uvicorn python-dotenv

Run standalone (sample data):
    python notification_agent.py

Run as an API:
    uvicorn notification_agent:app --port 8005 --reload

SMTP setup (.env):
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=your_address@gmail.com
    SMTP_PASSWORD=your_16_char_app_password
    FROM_NAME=The Hiring Team

    If SMTP_USER / SMTP_PASSWORD are not set, sending is skipped and the
    agent still returns the generated email text (preview-only mode) -
    this keeps the demo working even without real credentials configured.

Public URL (.env):
    NOTIFICATION_PUBLIC_URL=http://127.0.0.1:8005

    This is the base URL embedded in the "keep me in your talent pool" /
    "delete my data" links inside rejection emails. For a real deployment
    this MUST be a public HTTPS URL the candidate's email client can
    actually reach - localhost only works while testing on your own
    machine.
"""

import sys
import os
import re
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database"))
from db import (init_db, get_pii, get_job, get_notification_status, get_candidates_for_job,
                 mark_notified, get_pending_notifications, get_rankings_for_job,
                 update_data_retention_choice, VALID_RETENTION_CHOICES)
from shared.auth import verify_api_key

import google.generativeai as genai
from fastapi import FastAPI, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

API_KEY = os.environ.get("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-3.5-flash-lite")
else:
    model = None


def call_llm(prompt: str) -> str:
    if model is None:
        return None  # fall back to the plain template if no key is configured
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return None  # never let an LLM failure block sending the email


# ---------------------------------------------------------------
# SMTP sending
# ---------------------------------------------------------------
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
FROM_NAME = os.environ.get("FROM_NAME", "The Hiring Team")
PUBLIC_URL = os.environ.get("NOTIFICATION_PUBLIC_URL", "http://127.0.0.1:8005").rstrip("/")


def send_email(to_address: str, email_text: str) -> dict:
    """
    Parses the 'Subject: ...' line off the top of the generated email_text
    and sends the rest as the body to the candidate's real email address
    (pulled from the PII vault, which the Reader Agent populated from the
    candidate's own CV).

    Returns a result dict instead of raising, so one bad address or SMTP
    hiccup never crashes a batch send for every other candidate.
    """
    if not SMTP_USER or not SMTP_PASSWORD:
        return {"sent": False, "error": "SMTP credentials not configured (SMTP_USER/SMTP_PASSWORD)"}

    if not to_address:
        return {"sent": False, "error": "No email address found for this candidate (their CV may not have had one)"}

    lines = email_text.strip().split("\n", 1)
    if lines[0].lower().startswith("subject:"):
        subject = lines[0][len("subject:"):].strip()
        body = lines[1].lstrip("\n") if len(lines) > 1 else ""
    else:
        subject = "Update on your application"
        body = email_text

    msg = MIMEMultipart()
    msg["From"] = f"{FROM_NAME} <{SMTP_USER}>"
    msg["To"] = to_address
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, [to_address], msg.as_string())
        return {"sent": True, "error": None}
    except Exception as e:
        return {"sent": False, "error": str(e)}


# ---------------------------------------------------------------
# NLP component: a simple, fast tone-check on generated email text -
# no LLM needed for this part, just keyword matching. Flags harsh/
# clinical-sounding words so a rejection email never accidentally reads
# as cold, even if the LLM personalization above introduced one.
# ---------------------------------------------------------------
HARSH_WORDS = [
    "unfortunately", "failed", "reject", "rejected", "insufficient",
    "inadequate", "not good enough", "disqualified", "denied",
]


def tone_check(email_text: str) -> dict:
    text_lower = email_text.lower()
    found = [w for w in HARSH_WORDS if re.search(r"\b" + re.escape(w) + r"\b", text_lower)]
    return {
        "passed": len(found) == 0,
        "harsh_words_found": found,
        "note": ("Consider softening this email - it contains words that "
                  "may read as harsh." if found else
                  "No harsh/clinical language detected."),
    }


def personalize_shortlist_line(candidate_name, job_title, matched_skills):
    """
    LLM component: writes one warm, specific sentence referencing the
    candidate's actual top matched skill, instead of a fully generic
    template. Falls back to a plain generic line if the LLM isn't
    configured or fails - the email still sends either way.
    """
    if not matched_skills:
        return "We were impressed by your background."

    prompt = (f"Write ONE warm, specific sentence for a job-offer email, "
              f"congratulating a candidate for the {job_title} role on "
              f"their experience with {matched_skills[0]}. Keep it under "
              f"20 words. Do not mention their name or any score.")
    result = call_llm(prompt)
    return result or f"We were particularly impressed by your experience with {matched_skills[0]}."


def shortlist_email(candidate_name, job_title, interview_date, interview_time,
                     interview_location=None, matched_skills=None):
    location_line = f"Interview location: {interview_location}\n" if interview_location else ""
    personalized_line = personalize_shortlist_line(candidate_name, job_title, matched_skills or [])
    return f"""Subject: You've been shortlisted for {job_title}

Dear {candidate_name},

Thank you for applying for the {job_title} position. We're pleased to let
you know that you've been shortlisted for an interview. {personalized_line}

Interview Date: {interview_date}
Interview Time: {interview_time}
{location_line}
Please confirm your availability by replying to this email. We look
forward to speaking with you.

Best regards,
The Hiring Team.
"""


def early_rejection_email(candidate_name, job_title, data_retention_choice="delete", candidate_id=None):
    retention_note = _retention_note(data_retention_choice, candidate_id)
    return f"""Subject: Update on your application for {job_title}

Dear {candidate_name},

Thank you for applying for the {job_title} position. After careful review,
we've decided to move forward with other candidates at this time.

Your application was evaluated using the same skill-based criteria applied
to every applicant for this role, with no consideration of personal or
demographic characteristics.

We encourage you to apply for future openings that match your skills and
experience.
{retention_note}
Best regards,
The Hiring Team.
"""


def post_interview_rejection_email(candidate_name, job_title, data_retention_choice="delete", candidate_id=None):
    retention_note = _retention_note(data_retention_choice, candidate_id)
    return f"""Subject: Update on your application for {job_title}

Dear {candidate_name},

Thank you for taking the time to interview for the {job_title} position.
While we were impressed by your background, we've decided to move forward
with another candidate for this role.

Your application was evaluated using the same skill-based criteria applied
to every applicant for this role, with no consideration of personal or
demographic characteristics.

We'd like to keep your profile on file and encourage you to apply for
future opportunities with us.
{retention_note}
Best regards,
The Hiring Team.
"""


def _retention_note(choice, candidate_id=None):
    """
    Option 3 - candidate-controlled data lifecycle. IMPORTANT: nothing in
    this system actually asks the candidate for a retention preference
    before this point - "choice" here is only ever the "delete" default
    from save_pii(), never something the candidate actually chose. The
    old copy ("As you requested...") was therefore misleading, since no
    request was ever made.

    This version instead tells the candidate the CURRENT default and
    gives them real clickable links to change it - clicking a link hits
    the /candidate-preference endpoint below, which calls
    update_data_retention_choice() and actually updates the vault. If no
    candidate_id is available (e.g. a manual /notify test call), it falls
    back to a generic note with no links.
    """
    current_plan = {
        "delete": "delete your application data once this role is closed",
        "similar_roles": "keep your profile on file to consider you for similar roles over the next 6 months",
        "indefinite": "keep your profile in our talent pool indefinitely",
    }.get(choice, "delete your application data once this role is closed")

    if not candidate_id:
        return (f"\nBy default, we will {current_plan}. Reply to this email if "
                f"you'd like to choose a different option.\n")

    delete_link = f"{PUBLIC_URL}/candidate-preference?candidate_id={candidate_id}&choice=delete"
    similar_link = f"{PUBLIC_URL}/candidate-preference?candidate_id={candidate_id}&choice=similar_roles"
    indefinite_link = f"{PUBLIC_URL}/candidate-preference?candidate_id={candidate_id}&choice=indefinite"

    return f"""
By default, we will {current_plan}. You're in control of this - click one
of the links below any time and we'll update it immediately:

  Delete my data now: {delete_link}
  Keep me in mind for similar roles for 6 months: {similar_link}
  Keep my profile in your talent pool indefinitely: {indefinite_link}
"""


def generate_notification(candidate_name, job_title, status,
                           interview_date=None, interview_time=None, interview_location=None,
                           data_retention_choice="delete", matched_skills=None, candidate_id=None):
    """
    status must be one of: 'shortlisted', 'early_rejected', 'interview_rejected'
    data_retention_choice must be one of: 'delete', 'similar_roles', 'indefinite'
    (this is only ever the stored default unless the candidate has
    clicked a preference link before - see _retention_note above)

    Returns a dict with the email text AND the NLP tone-check result, so
    the Dashboard can flag (not block) any email that reads too harshly.
    """
    if status == "shortlisted":
        email_text = shortlist_email(candidate_name, job_title, interview_date,
                                      interview_time, interview_location, matched_skills)
    elif status == "early_rejected":
        email_text = early_rejection_email(candidate_name, job_title, data_retention_choice, candidate_id)
    elif status == "interview_rejected":
        email_text = post_interview_rejection_email(candidate_name, job_title, data_retention_choice, candidate_id)
    else:
        raise ValueError("Unknown status: " + status)

    return {"email_text": email_text, "tone_check": tone_check(email_text)}


# ---------------------------------------------------------------
# FastAPI wrapper
# ---------------------------------------------------------------
app = FastAPI(title="FairHire Notification Agent")
init_db()


class NotifyRequest(BaseModel):
    candidate_id: str
    job_id: int
    status: str  # 'shortlisted' | 'early_rejected' | 'interview_rejected'
    interview_date: Optional[str] = None
    interview_time: Optional[str] = None
    send: bool = False  # False = preview only (generate text, don't email)


class NotifyJobRequest(BaseModel):
    job_id: int
    send: bool = False  # False = preview only (old behavior), True = actually email


@app.get("/health")
def health():
    return {"status": "ok", "agent": "notification", "smtp_configured": bool(SMTP_USER and SMTP_PASSWORD)}


@app.post("/notify", dependencies=[Depends(verify_api_key)])
def notify_endpoint(request: NotifyRequest):
    """
    Manual version - you specify the status directly. Useful for testing,
    or a one-off email. For real use, prefer /notify-all-for-job below,
    which figures out each candidate's status automatically from the
    recruiter's real accept/reject decisions.
    """
    pii = get_pii(request.candidate_id, accessed_by="notification_agent",
                   purpose=f"generate {request.status} email")
    job = get_job(request.job_id)

    candidate_name = pii["name"] if pii and pii.get("name") else request.candidate_id
    candidate_email = pii["email"] if pii else None
    job_title = job["title"] if job else "the role"
    retention_choice = pii["data_retention_choice"] if pii else "delete"

    result = generate_notification(
        candidate_name, job_title, request.status,
        interview_date=request.interview_date, interview_time=request.interview_time,
        data_retention_choice=retention_choice, candidate_id=request.candidate_id,
    )

    send_result = None
    if request.send:
        send_result = send_email(candidate_email, result["email_text"])

    return {"candidate_id": request.candidate_id, **result, "send_result": send_result}


@app.post("/notify-all-for-job", dependencies=[Depends(verify_api_key)])
def notify_all_for_job_endpoint(request: NotifyJobRequest):
    """
    The batch-send mechanism: only processes decisions that haven't been
    notified yet (notified = 0), instead of re-deriving "latest status" for
    every candidate every time - that distinction matters, because a
    candidate's latest decision doesn't change after their email is sent,
    so checking "latest decision" alone would send the same email again on
    every re-run. Checking the notified flag specifically is what prevents
    duplicate emails going out on a second batch run.

    If request.send is True, each generated email is actually emailed to
    the candidate's real address (from the PII vault, sourced from their
    CV). If it's False, text is generated and returned for preview but
    nothing is emailed.
    """
    job = get_job(request.job_id)
    job_title = job["title"] if job else "the role"
    pending = get_pending_notifications(request.job_id)

    results = []
    for decision_row in pending:
        candidate_id = decision_row["candidate_id"]

        if decision_row["stage"] == "final":
            status = "interview_rejected" if decision_row["decision"] == "rejected" else None
        else:
            status = "early_rejected" if decision_row["decision"] == "rejected" else "shortlisted"

        if status is None:
            # e.g. hired at final stage - no rejection email needed, but
            # still mark it notified so it doesn't get re-checked forever.
            mark_notified(decision_row["decision_id"])
            continue

        pii = get_pii(candidate_id, accessed_by="notification_agent",
                       purpose=f"generate {status} email")
        candidate_name = pii["name"] if pii and pii.get("name") else candidate_id
        candidate_email = pii["email"] if pii else None
        retention_choice = pii["data_retention_choice"] if pii else "delete"

        # Pull this candidate's matched skills from their ranking record,
        # so the LLM personalization line can reference something real.
        job_rankings = get_rankings_for_job(request.job_id)
        matched = next((r["matched_skills"] for r in job_rankings
                        if r["candidate_id"] == candidate_id), [])

        result = generate_notification(
            candidate_name, job_title, status,
            interview_date=decision_row.get("interview_date"),
            interview_time=decision_row.get("interview_time"),
            interview_location=decision_row.get("interview_location"),
            data_retention_choice=retention_choice,
            matched_skills=matched,
            candidate_id=candidate_id,
        )

        send_result = None
        if request.send:
            send_result = send_email(candidate_email, result["email_text"])

        results.append({"candidate_id": candidate_id, "status": status,
                         **result, "send_result": send_result})

        # Only mark notified once the send genuinely succeeded (or when
        # send wasn't requested at all, matching the original preview
        # behavior) - otherwise a bad address / SMTP failure would
        # silently skip that candidate on every future batch run.
        if not request.send or (send_result and send_result["sent"]):
            mark_notified(decision_row["decision_id"])

    return {"job_id": request.job_id, "notifications": results}


# ---------------------------------------------------------------
# Candidate-facing preference link - what the candidate actually clicks
# in their email to record a REAL choice, instead of the choice always
# silently defaulting to "delete" with no one ever having been asked.
# ---------------------------------------------------------------
_CHOICE_LABELS = {
    "delete": "delete your application data once this role is closed",
    "similar_roles": "keep your profile on file for similar roles over the next 6 months",
    "indefinite": "keep your profile in our talent pool indefinitely",
}


@app.get("/candidate-preference", response_class=HTMLResponse)
def candidate_preference_endpoint(candidate_id: str, choice: str):
    """
    Public, no-login link a candidate clicks directly from their email.
    Deliberately a simple GET (not a POST/form) so it works as a plain
    clickable link in an email client with no extra steps required.

    SECURITY NOTE: this endpoint is intentionally NOT protected by
    verify_api_key() - a real candidate clicking a link in their email has
    no way to attach a custom header, so requiring the shared API key here
    would make the feature unusable. The trade-off is that this endpoint
    can only ever perform one narrow, low-risk action (setting one of
    three fixed retention choices for one specific candidate_id it doesn't
    need to guess), never anything more sensitive - so it's an acceptable
    scope-limited exception, not a general bypass of authentication.

    NOTE for deployment: NOTIFICATION_PUBLIC_URL must point to a real,
    publicly reachable HTTPS URL for this to work outside local testing -
    localhost links are only clickable on the same machine.
    """
    if choice not in VALID_RETENTION_CHOICES:
        return HTMLResponse(
            f"<h2>Invalid option</h2><p>'{choice}' isn't a recognized preference.</p>",
            status_code=400,
        )

    updated = update_data_retention_choice(candidate_id, choice)
    if not updated:
        return HTMLResponse(
            "<h2>We couldn't find your application</h2>"
            "<p>This link may have expired or the candidate ID wasn't recognized.</p>",
            status_code=404,
        )

    label = _CHOICE_LABELS[choice]
    return HTMLResponse(f"""
        <h2>Preference updated</h2>
        <p>Thanks for letting us know - we will now <strong>{label}</strong>.</p>
        <p>You can change this at any time by clicking a different link in your email.</p>
    """)


if __name__ == "__main__":
    # Example usage - swap these with real candidate/dashboard data later
    result1 = generate_notification(
        candidate_name="Candidate A",
        job_title="Data Analyst",
        status="shortlisted",
        interview_date="2026-08-25",
        interview_time="10:00 AM",
        matched_skills=["Python", "SQL"],
    )
    print(result1["email_text"])
    print("Tone check:", result1["tone_check"])

    print("\n---\n")

    result2 = generate_notification(
        candidate_name="Candidate B",
        job_title="Data Analyst",
        status="early_rejected",
        candidate_id="CAND_002",
    )
    print(result2["email_text"])
    print("Tone check:", result2["tone_check"])

  