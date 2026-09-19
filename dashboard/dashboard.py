"""
FairHire - Recruiter Dashboard
--------------------------------
Recruiter-facing Streamlit application.

REAL integrations:
    - Shared database
    - JD Fairness Agent
    - Reader Agent
    - Matcher Agent
    - Explainer Agent
    - Notification Agent

Run:
    streamlit run dashboard.py

Required agents:

    uvicorn jd_fairness_checker:app --port 8004 --reload
    uvicorn matcher_agent:app --port 8002 --reload
    uvicorn explainer_agent:app --port 8003 --reload
    uvicorn notification_agent:app --port 8005 --reload
    uvicorn reader_agent:app --port 8001 --reload
"""

# ============================================================
# IMPORTS
# ============================================================

import sys
import os
import io
import hashlib
import hmac
import textwrap
import requests
import streamlit as st

from html import escape as esc
from dotenv import load_dotenv

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)

# ============================================================
# HTML RENDERING FIX
# ------------------------------------------------------------
# Streamlit runs st.markdown() content through a Markdown
# parser first. Any line indented 4+ spaces after a blank line
# is treated as a code block, which makes raw HTML/SVG show up
# as text in white boxes.
#
# This wraps st.markdown so that every HTML string
# (unsafe_allow_html=True) has its indentation and blank lines
# stripped before Streamlit sees it.
# ============================================================


def _flatten_html(body):
    if not isinstance(body, str):
        return body

    lines = (line.strip() for line in body.splitlines())

    return "\n".join(line for line in lines if line)


_original_markdown = st.markdown


def _safe_markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html"):
        body = _flatten_html(body)

    return _original_markdown(body, *args, **kwargs)


st.markdown = _safe_markdown


def safe(value):
    """
    HTML-escapes any user- or AI-generated text before it is
    inserted into an HTML string.
    """

    if value is None:
        return ""

    return esc(str(value))


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "database"
    )
)

from db import (
    init_db,
    get_rankings_for_job,
    log_decision,
    get_job,
    close_job,
    purge_expired_retention,
    get_decisions_for_job,
    get_pii,
    get_candidate,
)

# ============================================================
# AGENT URLS
# ============================================================

JD_AGENT_URL = "http://127.0.0.1:8004"
MATCHER_URL = "http://127.0.0.1:8002"
EXPLAINER_URL = "http://127.0.0.1:8003"
NOTIFICATION_URL = "http://127.0.0.1:8005"
READER_URL = "http://127.0.0.1:8001"

REQUEST_TIMEOUT = 90

# ============================================================
# API SECURITY
# ============================================================

API_KEY = os.environ.get("FAIRHIRE_API_KEY", "")

AUTH_HEADERS = (
    {"X-API-Key": API_KEY}
    if API_KEY
    else {}
)

# ============================================================
# DASHBOARD LOGIN
# ============================================================

DASHBOARD_USERNAME = os.environ.get(
    "DASHBOARD_USERNAME",
    ""
)

DASHBOARD_PASSWORD_HASH = os.environ.get(
    "DASHBOARD_PASSWORD_HASH",
    ""
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FairHire | Recruiter Workspace",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# ============================================================
# FAIRHIRE LOGO
# ============================================================
#
# Geometric "F" monogram.
# No tick/checkmark.
# ============================================================

FAIRHIRE_LOGO_SVG = """
<svg width="{size}" height="{size}"
     viewBox="0 0 60 60"
     xmlns="http://www.w3.org/2000/svg">

    <defs>
        <linearGradient id="fhGradient"
                        x1="0"
                        y1="0"
                        x2="1"
                        y2="1">
            <stop offset="0%" stop-color="#2DD4BF"/>
            <stop offset="100%" stop-color="#38BDF8"/>
        </linearGradient>
    </defs>

    <rect
        x="1"
        y="1"
        width="58"
        height="58"
        rx="16"
        fill="#0D1828"
        stroke="#24415A"
        stroke-width="2"
    />

    <path
        d="M19 17
           H42
           V24
           H27
           V29
           H39
           V36
           H27
           V43
           H19
           Z"
        fill="url(#fhGradient)"
    />

    <circle
        cx="44"
        cy="43"
        r="3"
        fill="#2DD4BF"
    />

</svg>
"""

# ============================================================
# DARK UI / GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

/* ==========================================================
   GLOBAL
   ========================================================== */

html,
body,
[class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 80% 0%,
            rgba(45, 212, 191, 0.07),
            transparent 30%
        ),
        radial-gradient(
            circle at 10% 20%,
            rgba(56, 189, 248, 0.045),
            transparent 25%
        ),
        #07111F;

    color: #F8FAFC;
}

/* Main content */

.block-container {
    max-width: 1240px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Streamlit header */

header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Remove menu/footer but keep header functionality */

#MainMenu,
footer {
    visibility: hidden;
}

/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #091523 0%,
            #07111F 100%
        );

    border-right: 1px solid #1B3045;
}

section[data-testid="stSidebar"] > div {
    background: transparent;
}

section[data-testid="stSidebar"] * {
    color: #DDE7F2;
}

/* Sidebar brand */

.fh-sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.35rem 0 0.8rem 0;
}

.fh-sidebar-brand span {
    font-size: 1.12rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #F8FAFC !important;
}

.fh-sidebar-subtitle {
    color: #7890A7 !important;
    font-size: 0.74rem;
    margin-top: -0.25rem;
    margin-bottom: 1rem;
}

/* Sidebar buttons */

section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;

    min-height: 42px;

    border-radius: 9px;
    border: 1px solid transparent;

    background: transparent;
    color: #9FB2C7;

    font-weight: 600;

    transition:
        background 0.15s ease,
        border 0.15s ease,
        color 0.15s ease;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: #102337;
    border-color: #1F3A52;
    color: #F8FAFC;
}

/* ==========================================================
   MAIN HERO
   ========================================================== */

.fh-hero {
    position: relative;
    overflow: hidden;

    background:
        linear-gradient(
            135deg,
            #0D1B2B 0%,
            #0A1726 55%,
            #0C1D2C 100%
        );

    border: 1px solid #1D354B;

    padding: 1.7rem 2rem;

    border-radius: 18px;

    margin-bottom: 1.4rem;

    display: flex;
    align-items: center;

    gap: 1rem;

    box-shadow:
        0 18px 50px rgba(0, 0, 0, 0.18);
}

.fh-hero::after {
    content: "";

    position: absolute;

    width: 240px;
    height: 240px;

    right: -80px;
    top: -110px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(45, 212, 191, 0.14),
            transparent 70%
        );

    pointer-events: none;
}

.fh-hero-text {
    position: relative;
    z-index: 1;
}

.fh-hero-text h1 {
    margin: 0;

    font-size: 1.55rem;

    font-weight: 800;

    letter-spacing: -0.03em;

    color: #F8FAFC !important;
}

.fh-hero-text p {
    margin: 0.3rem 0 0;

    color: #8EA3B8 !important;

    font-size: 0.88rem;
}

/* ==========================================================
   PAGE HEADER
   ========================================================== */

.fh-page-title {
    margin-top: 0.5rem;
    margin-bottom: 0.2rem;

    font-size: 1.75rem;

    font-weight: 800;

    letter-spacing: -0.04em;

    color: #F8FAFC !important;
}

.fh-page-subtitle {
    margin-top: 0;

    color: #8EA3B8 !important;

    font-size: 0.9rem;
}

/* ==========================================================
   SECTION HEADER
   ========================================================== */

.fh-section-header {
    display: flex;

    align-items: center;

    gap: 0.65rem;

    margin: 1.6rem 0 0.85rem 0;
}

.fh-step-num {
    display: inline-flex;

    align-items: center;
    justify-content: center;

    width: 27px;
    height: 27px;

    border-radius: 8px;

    background:
        linear-gradient(
            135deg,
            rgba(45, 212, 191, 0.17),
            rgba(56, 189, 248, 0.12)
        );

    border: 1px solid #245468;

    color: #5EEAD4 !important;

    font-weight: 800;

    font-size: 0.78rem;

    flex-shrink: 0;
}

.fh-title {
    font-size: 1.12rem;

    font-weight: 700;

    color: #EAF2F8 !important;
}

/* ==========================================================
   CARDS
   ========================================================== */

.fh-card {
    background:
        linear-gradient(
            145deg,
            #0E1B2B,
            #0B1725
        );

    border: 1px solid #1D344A;

    border-radius: 15px;

    padding: 1.15rem;

    margin-bottom: 0.8rem;

    box-shadow:
        0 12px 30px rgba(0, 0, 0, 0.12);
}

.fh-card-title {
    color: #F8FAFC !important;

    font-weight: 700;

    font-size: 0.98rem;
}

.fh-card-description {
    color: #8499AE !important;

    font-size: 0.82rem;

    margin-top: 0.25rem;
}

/* ==========================================================
   KPI CARDS
   ========================================================== */

.fh-kpi {
    background:
        linear-gradient(
            145deg,
            #0E1B2B,
            #0A1624
        );

    border: 1px solid #1D344A;

    border-radius: 15px;

    padding: 1.05rem 1.1rem;

    min-height: 108px;
}

.fh-kpi-label {
    color: #8197AC !important;

    font-size: 0.74rem;

    font-weight: 600;

    text-transform: uppercase;

    letter-spacing: 0.08em;
}

.fh-kpi-value {
    color: #F8FAFC !important;

    font-size: 1.55rem;

    font-weight: 800;

    margin-top: 0.25rem;
}

.fh-kpi-note {
    color: #627B91 !important;

    font-size: 0.72rem;

    margin-top: 0.1rem;
}

/* ==========================================================
   BADGES
   ========================================================== */

.fh-badge {
    display: inline-block;

    padding: 0.22rem 0.65rem;

    border-radius: 999px;

    font-size: 0.72rem;

    font-weight: 700;

    border: 1px solid transparent;
}

.fh-badge-strong {
    color: #86EFAC !important;

    background: rgba(34, 197, 94, 0.12);

    border-color: rgba(34, 197, 94, 0.24);
}

.fh-badge-moderate {
    color: #FCD34D !important;

    background: rgba(245, 158, 11, 0.12);

    border-color: rgba(245, 158, 11, 0.24);
}

.fh-badge-weak {
    color: #FDA4AF !important;

    background: rgba(248, 113, 113, 0.11);

    border-color: rgba(248, 113, 113, 0.22);
}

/* ==========================================================
   BUTTONS
   ========================================================== */

.stButton > button {
    border-radius: 9px;

    min-height: 40px;

    font-weight: 650;

    border: 1px solid #29445A;

    background: #102033;

    color: #DCE8F2;

    transition:
        transform 0.12s ease,
        border 0.12s ease,
        background 0.12s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);

    border-color: #3A647C;

    background: #142A3E;

    color: #FFFFFF;
}

.stButton > button[kind="primary"] {
    background:
        linear-gradient(
            135deg,
            #14B8A6,
            #0EA5A0
        ) !important;

    border-color: #1CC8B5 !important;

    color: #041615 !important;

    font-weight: 800;
}

.stButton > button[kind="primary"]:hover {
    background:
        linear-gradient(
            135deg,
            #2DD4BF,
            #14B8A6
        ) !important;
}

/* ==========================================================
   INPUTS
   ========================================================== */

.stTextInput input,
.stTextArea textarea,
.stDateInput input,
.stTimeInput input {
    background: #0A1725 !important;

    color: #F8FAFC !important;

    border: 1px solid #223B51 !important;

    border-radius: 9px !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus,
.stDateInput input:focus,
.stTimeInput input:focus {
    border-color: #2A9D96 !important;

    box-shadow:
        0 0 0 1px rgba(45, 212, 191, 0.18) !important;
}

div[data-baseweb="select"] > div {
    background: #0A1725 !important;

    color: #F8FAFC !important;

    border-color: #223B51 !important;

    border-radius: 9px !important;
}

div[data-baseweb="popover"] {
    background: #0D1B2B !important;
}

div[data-baseweb="menu"] {
    background: #0D1B2B !important;
}

div[data-baseweb="option"] {
    color: #E6EEF5 !important;
}

div[data-baseweb="option"]:hover {
    background: #163047 !important;
}

/* Labels */

.stTextInput label,
.stTextArea label,
.stDateInput label,
.stTimeInput label,
.stSelectbox label,
.stMultiSelect label,
.stFileUploader label {
    color: #B8C8D8 !important;

    font-weight: 600 !important;
}

/* ==========================================================
   EXPANDERS
   ========================================================== */

div[data-testid="stExpander"] {
    background: #0D1B2B !important;

    border: 1px solid #1D344A !important;

    border-radius: 13px !important;
}

div[data-testid="stExpander"] summary {
    color: #EAF2F8 !important;
}

/* ==========================================================
   CONTAINERS
   ========================================================== */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background:
        linear-gradient(
            145deg,
            rgba(14, 27, 43, 0.94),
            rgba(10, 22, 36, 0.94)
        );

    border: 1px solid #1D344A !important;

    border-radius: 14px !important;
}

/* ==========================================================
   METRICS
   ========================================================== */

div[data-testid="stMetric"] {
    background: transparent;
}

div[data-testid="stMetricLabel"] {
    color: #8197AC !important;
}

div[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
}

/* ==========================================================
   PROGRESS
   ========================================================== */

div[data-testid="stProgress"] > div {
    background: #17283A !important;
}

div[data-testid="stProgress"] > div > div {
    background:
        linear-gradient(
            90deg,
            #14B8A6,
            #38BDF8
        ) !important;
}

/* ==========================================================
   ALERTS
   ========================================================== */

div[data-testid="stAlert"] {
    border-radius: 10px !important;
}

/* ==========================================================
   FILE UPLOADER
   ========================================================== */

section[data-testid="stFileUploaderDropzone"] {
    background: #0A1725 !important;

    border: 1px dashed #31506A !important;

    border-radius: 11px !important;
}

/* ==========================================================
   LOGIN
   ========================================================== */

.fh-login-wrapper {
    max-width: 510px;

    margin: 4rem auto 0;

    text-align: center;
}

.fh-login-logo {
    margin-bottom: 0.8rem;
}

.fh-login-title {
    font-size: 2rem;

    font-weight: 800;

    letter-spacing: -0.04em;

    color: #F8FAFC !important;

    margin: 0;
}

.fh-login-subtitle {
    color: #8499AE !important;

    font-size: 0.9rem;

    margin-top: 0.35rem;
}

.fh-login-feature-row {
    display: flex;

    justify-content: center;

    flex-wrap: wrap;

    gap: 0.45rem;

    margin-top: 1.1rem;
}

.fh-login-feature {
    border: 1px solid #1D4050;

    background: rgba(45, 212, 191, 0.06);

    color: #9FE8DE !important;

    border-radius: 999px;

    padding: 0.28rem 0.65rem;

    font-size: 0.68rem;

    font-weight: 650;
}

.fh-login-note {
    margin-top: 1.1rem;

    color: #617A90 !important;

    font-size: 0.72rem;

    line-height: 1.5;
}

/* ==========================================================
   WORKFLOW
   ========================================================== */

.fh-workflow {
    display: flex;

    align-items: center;

    gap: 0.45rem;

    margin: 0.7rem 0 1.4rem;

    overflow-x: auto;
}

.fh-workflow-item {
    display: flex;

    align-items: center;

    gap: 0.35rem;

    white-space: nowrap;

    color: #60788E !important;

    font-size: 0.7rem;

    font-weight: 650;
}

.fh-workflow-item.active {
    color: #5EEAD4 !important;
}

.fh-workflow-dot {
    width: 8px;

    height: 8px;

    border-radius: 50%;

    background: #284156;
}

.fh-workflow-item.active .fh-workflow-dot {
    background: #2DD4BF;

    box-shadow:
        0 0 10px rgba(45, 212, 191, 0.45);
}

.fh-workflow-line {
    width: 24px;

    height: 1px;

    background: #20364A;
}

/* ==========================================================
   SMALL TEXT
   ========================================================== */

.fh-muted {
    color: #7890A6 !important;

    font-size: 0.82rem;
}

.fh-success {
    color: #86EFAC !important;
}

.fh-warning {
    color: #FCD34D !important;
}

.fh-danger {
    color: #FDA4AF !important;
}

/* ==========================================================
   READABILITY FIXES (widgets that kept Streamlit's default
   light-theme colors on the dark background)
   ========================================================== */

/* Checkbox / toggle / radio labels and any widget label */

div[data-testid="stCheckbox"] label,
div[data-testid="stCheckbox"] label p,
div[data-testid="stCheckbox"] label span,
div[data-testid="stToggle"] label p,
div[data-testid="stRadio"] label p,
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] label {
    color: #DCE8F2 !important;
}

/* Tooltip (?) icon */

div[data-testid="stTooltipIcon"] svg {
    color: #8EA3B8 !important;
    fill: #8EA3B8 !important;
}

/* File uploader: text, icons, file list */

section[data-testid="stFileUploaderDropzone"] * {
    color: #B8C8D8 !important;
}

section[data-testid="stFileUploaderDropzone"] svg {
    fill: #B8C8D8 !important;
    color: #B8C8D8 !important;
}

section[data-testid="stFileUploaderDropzone"] small {
    color: #7F95AA !important;
}

/* File uploader "Upload" / "Browse files" button */

section[data-testid="stFileUploaderDropzone"] button {
    background: #102033 !important;
    border: 1px solid #29445A !important;
    border-radius: 9px !important;
}

section[data-testid="stFileUploaderDropzone"] button:hover {
    background: #142A3E !important;
    border-color: #3A647C !important;
}

section[data-testid="stFileUploaderDropzone"] button,
section[data-testid="stFileUploaderDropzone"] button * {
    color: #DCE8F2 !important;
}

/* Uploaded file names / sizes */

div[data-testid="stFileUploaderFile"],
div[data-testid="stFileUploaderFile"] * {
    color: #DCE8F2 !important;
}

/* Captions */

div[data-testid="stCaptionContainer"],
div[data-testid="stCaptionContainer"] * {
    color: #8EA3B8 !important;
}

/* Alerts: keep the tinted background but make the text readable */

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] div {
    color: #E6EEF5 !important;
}

/* Download button */

.stDownloadButton > button {
    border-radius: 9px;
    min-height: 40px;
    font-weight: 650;
    border: 1px solid #29445A;
    background: #102033;
    color: #DCE8F2 !important;
}

.stDownloadButton > button:hover {
    border-color: #3A647C;
    background: #142A3E;
    color: #FFFFFF !important;
}

/* ==========================================================
   EXPANDER HEADER + PLAIN TEXT (email previews)
   ========================================================== */

div[data-testid="stExpander"] details,
div[data-testid="stExpander"] summary {
    background: #0D1B2B !important;
    color: #EAF2F8 !important;
    border-radius: 13px !important;
}

div[data-testid="stExpander"] summary:hover {
    background: #102337 !important;
}

div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary *,
div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] summary span {
    color: #EAF2F8 !important;
}

div[data-testid="stExpander"] summary svg {
    fill: #EAF2F8 !important;
    color: #EAF2F8 !important;
}

div[data-testid="stExpanderDetails"] {
    background: transparent !important;
}

/* st.text() output */

div[data-testid="stText"],
div[data-testid="stText"] *,
div[data-testid="stExpanderDetails"] pre,
div[data-testid="stExpanderDetails"] pre * {
    color: #DCE7F0 !important;
    background: transparent !important;
}

/* st.markdown() paragraphs inside expanders */

div[data-testid="stExpanderDetails"] div[data-testid="stMarkdownContainer"] p {
    color: #DCE7F0 !important;
}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# PDF GENERATOR
# ============================================================

def build_interview_questions_pdf(
    candidate_id,
    job_title,
    interview_date,
    interview_time,
    questions,
    cv_based,
):
    """
    Builds the interviewer PDF.
    Candidate PII is intentionally not included.

    ReportLab Paragraphs parse XML-like markup, so all dynamic
    text is escaped first.
    """

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=16,
        spaceAfter=4,
    )

    meta_style = ParagraphStyle(
        "MetaStyle",
        parent=styles["Normal"],
        fontSize=10,
        textColor="#444444",
        spaceAfter=2,
    )

    question_style = ParagraphStyle(
        "QuestionStyle",
        parent=styles["Normal"],
        fontSize=11,
        spaceAfter=14,
        leading=15,
    )

    note_style = ParagraphStyle(
        "NoteStyle",
        parent=styles["Normal"],
        fontSize=8,
        textColor="#888888",
        spaceBefore=20,
    )

    story = [
        Paragraph(
            f"Interview Questions - {safe(job_title)}",
            title_style,
        ),
        Paragraph(
            f"Candidate reference: {safe(candidate_id)}",
            meta_style,
        ),
    ]

    if interview_date:
        story.append(
            Paragraph(
                f"Interview date: {safe(interview_date)}",
                meta_style,
            )
        )

    if interview_time:
        story.append(
            Paragraph(
                f"Interview time: {safe(interview_time)}",
                meta_style,
            )
        )

    story.append(
        Spacer(1, 0.25 * inch)
    )

    for i, question in enumerate(
        questions,
        start=1,
    ):
        story.append(
            Paragraph(
                f"{i}. {safe(question)}",
                question_style,
            )
        )

    if cv_based:
        basis_note = (
            "Questions generated from the candidate's submitted CV."
        )
    else:
        basis_note = (
            "Note: no stored CV text was found for this candidate - "
            "questions were generated from their skill list only."
        )

    story.append(
        Paragraph(
            basis_note,
            note_style,
        )
    )

    story.append(
        Paragraph(
            "For interviewer use only.",
            note_style,
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.read()


# ============================================================
# AUTHENTICATION
# ============================================================

def check_login(username, password):
    """
    Authenticates using credentials from environment variables.
    """

    if (
        not DASHBOARD_USERNAME
        or not DASHBOARD_PASSWORD_HASH
    ):
        return False

    entered_hash = hashlib.sha256(
        password.encode()
    ).hexdigest()

    username_ok = hmac.compare_digest(
        username,
        DASHBOARD_USERNAME,
    )

    password_ok = hmac.compare_digest(
        entered_hash,
        DASHBOARD_PASSWORD_HASH,
    )

    return username_ok and password_ok


# ============================================================
# NAVIGATION
# ============================================================

NAV_ITEMS = [
    "Dashboard",
    "Jobs",
    "Candidates",
    "Interviews",
    "Notifications",
    "Analytics",
]


def navigate(page):
    """
    Central navigation helper.

    Every workflow action can use this so the application
    moves to the correct screen after an action.
    """

    st.session_state.page = page
    st.rerun()


# ============================================================
# UI HELPERS
# ============================================================

def section_header(number, title):
    st.markdown(
        f"""
        <div class="fh-section-header">
            <span class="fh-step-num">{number}</span>
            <span class="fh-title">{safe(title)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_badge(score):
    if score >= 0.6:
        css_class = "fh-badge-strong"
        label = "Strong match"

    elif score >= 0.35:
        css_class = "fh-badge-moderate"
        label = "Moderate match"

    else:
        css_class = "fh-badge-weak"
        label = "Weak match"

    return (
        f'<span class="fh-badge {css_class}">'
        f'{label} · {score:.2f}'
        f'</span>'
    )


def render_hero():
    hero_html = f"""
    <div class="fh-hero">

        {FAIRHIRE_LOGO_SVG.format(size=44)}

        <div class="fh-hero-text">
            <h1>FairHire Recruiter Workspace</h1>
            <p>
                Anonymized screening · Bias-audited ranking ·
                Explainable decisions
            </p>
        </div>

    </div>
    """

    st.markdown(
        textwrap.dedent(hero_html),
        unsafe_allow_html=True,
    )


def render_page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="fh-page-title">{safe(title)}</div>
        <div class="fh-page-subtitle">{safe(subtitle)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow(current_page):
    items = [
        "Jobs",
        "Candidates",
        "Interviews",
        "Notifications",
    ]

    workflow_html = '<div class="fh-workflow">'

    for index, item in enumerate(items):

        active = (
            item == current_page
            or (
                current_page == "Dashboard"
                and item == "Jobs"
            )
        )

        active_class = "active" if active else ""

        workflow_html += f"""
        <div class="fh-workflow-item {active_class}">
            <span class="fh-workflow-dot"></span>
            {item}
        </div>
        """

        if index < len(items) - 1:
            workflow_html += '<div class="fh-workflow-line"></div>'

    workflow_html += "</div>"

    st.markdown(
        workflow_html,
        unsafe_allow_html=True,
    )


def render_kpi(label, value, note=""):
    st.markdown(
        f"""
        <div class="fh-kpi">
            <div class="fh-kpi-label">{safe(label)}</div>
            <div class="fh-kpi-value">{safe(value)}</div>
            <div class="fh-kpi-note">{safe(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_active_job():
    job_id = st.session_state.get("job_id")

    if not job_id:
        return None

    return get_job(job_id)


def get_latest_decisions(job_id):
    """
    Creates a candidate -> latest decision mapping.
    """

    decisions = get_decisions_for_job(job_id)

    latest = {}

    for decision in decisions:
        candidate_id = decision.get(
            "candidate_id"
        )

        if candidate_id:
            latest[candidate_id] = decision

    return latest


# ============================================================
# LOGIN SCREEN
# ============================================================

def render_login():
    st.markdown(
        f"""
        <div class="fh-login-wrapper">

            <div class="fh-login-logo">
                {FAIRHIRE_LOGO_SVG.format(size=64)}
            </div>

            <h1 class="fh-login-title">
                FairHire
            </h1>

            <div class="fh-login-subtitle">
                Recruiter Workspace
            </div>

            <div class="fh-login-feature-row">

                <span class="fh-login-feature">
                    Fairness checks
                </span>

                <span class="fh-login-feature">
                    Explainable matching
                </span>

                <span class="fh-login-feature">
                    Privacy-aware screening
                </span>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns(
        [1, 1.15, 1]
    )

    with center:

        st.write("")

        with st.container(border=True):

            st.markdown(
                """
                <div style="
                    text-align:center;
                    margin-bottom:1rem;
                ">
                    <div style="
                        color:#F8FAFC;
                        font-size:1.1rem;
                        font-weight:750;
                    ">
                        Sign in
                    </div>

                    <div style="
                        color:#71879B;
                        font-size:0.76rem;
                        margin-top:0.25rem;
                    ">
                        Access the recruiter workspace
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            username = st.text_input(
                "Username",
                placeholder="Enter your recruiter username",
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
            )

            if st.button(
                "Sign in to FairHire",
                type="primary",
                use_container_width=True,
            ):

                if check_login(
                    username,
                    password,
                ):
                    st.session_state.logged_in = True
                    st.session_state.page = "Dashboard"

                    st.rerun()

                else:
                    st.error(
                        "Invalid username or password."
                    )

        st.markdown(
            """
            <div class="fh-login-note">
                FairHire keeps candidate identities separate from
                ranking decisions wherever possible and uses
                authenticated agent-to-agent requests.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    current_page = st.session_state.page

    with st.sidebar:

        brand_html = f"""
        <div class="fh-sidebar-brand">

            {FAIRHIRE_LOGO_SVG.format(size=32)}

            <span>FairHire</span>

        </div>

        <div class="fh-sidebar-subtitle">
            Recruiter Workspace
        </div>
        """

        st.markdown(
            textwrap.dedent(brand_html),
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="
                color:#5EEAD4;
                font-size:0.66rem;
                font-weight:700;
                letter-spacing:0.08em;
                text-transform:uppercase;
                margin-bottom:0.55rem;
            ">
                Workspace
            </div>
            """,
            unsafe_allow_html=True,
        )

        for item in NAV_ITEMS:

            if st.button(
                item,
                key=f"nav_{item}",
                use_container_width=True,
            ):

                if current_page != item:
                    navigate(item)

        st.divider()

        active_job_id = st.session_state.get(
            "job_id"
        )

        if active_job_id:

            active_job = get_job(
                active_job_id
            )

            if active_job:

                st.markdown(
                    """
                    <div style="
                        color:#71879B;
                        font-size:0.67rem;
                        font-weight:700;
                        text-transform:uppercase;
                        letter-spacing:0.07em;
                    ">
                        Active job
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div style="
                        color:#EAF2F8;
                        font-size:0.84rem;
                        font-weight:700;
                        margin-top:0.25rem;
                    ">
                        #{safe(active_job_id)}
                    </div>

                    <div style="
                        color:#71879B;
                        font-size:0.72rem;
                        margin-top:0.12rem;
                        line-height:1.4;
                    ">
                        {safe(active_job.get("title", "Untitled role"))}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                status = active_job.get(
                    "status",
                    "open",
                )

                status_color = (
                    "#86EFAC"
                    if status == "open"
                    else "#94A3B8"
                )

                st.markdown(
                    f"""
                    <div style="
                        margin-top:0.55rem;
                        color:{status_color};
                        font-size:0.7rem;
                        font-weight:700;
                    ">
                        ● {safe(status.upper())}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.caption(
                "No active job selected."
            )

        st.divider()

        if st.button(
            "Log out",
            use_container_width=True,
        ):

            st.session_state.logged_in = False

            st.rerun()


# ============================================================
# DASHBOARD PAGE
# ============================================================

def render_dashboard():

    render_page_header(
        "Dashboard",
        "A high-level view of your current FairHire recruitment workflow.",
    )

    render_workflow("Dashboard")

    job_id = st.session_state.get(
        "job_id"
    )

    current_job = get_active_job()

    if not job_id:

        col1, col2, col3 = st.columns(3)

        with col1:
            render_kpi(
                "Active role",
                "—",
                "No job posted",
            )

        with col2:
            render_kpi(
                "Candidates",
                "0",
                "Waiting for screening",
            )

        with col3:
            render_kpi(
                "Decisions",
                "0",
                "No recruiter decisions yet",
            )

        st.write("")

        with st.container(border=True):

            st.markdown(
                """
                <div class="fh-card-title">
                    Start a recruitment workflow
                </div>

                <div class="fh-card-description">
                    Create a job, run the fairness check, process
                    candidate CVs, review explainable rankings,
                    and manage interview decisions.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            if st.button(
                "Create a job",
                type="primary",
            ):
                navigate("Jobs")

        return

    rankings = get_rankings_for_job(
        job_id
    )

    decisions = get_decisions_for_job(
        job_id
    )

    accepted = len([
        d for d in decisions
        if d.get("decision") == "accepted"
    ])

    rejected = len([
        d for d in decisions
        if d.get("decision") == "rejected"
    ])

    status = (
        current_job.get("status", "open")
        if current_job
        else "open"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_kpi(
            "Active role",
            f"#{job_id}",
            current_job.get(
                "title",
                "Untitled role",
            )
            if current_job
            else "",
        )

    with col2:
        render_kpi(
            "Candidates",
            len(rankings),
            "Ranked candidates",
        )

    with col3:
        render_kpi(
            "Accepted",
            accepted,
            "Interview pipeline",
        )

    with col4:
        render_kpi(
            "Rejected",
            rejected,
            f"Job status: {status}",
        )

    st.write("")

    section_header(
        1,
        "Current recruitment workflow",
    )

    c1, c2 = st.columns(2)

    with c1:

        with st.container(border=True):

            st.markdown(
                """
                <div class="fh-card-title">
                    Candidate screening
                </div>

                <div class="fh-card-description">
                    Process CVs through the Reader Agent and
                    generate explainable candidate rankings.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            if st.button(
                "Open Candidates",
                type="primary",
                use_container_width=True,
            ):
                navigate("Candidates")

    with c2:

        with st.container(border=True):

            st.markdown(
                """
                <div class="fh-card-title">
                    Interview pipeline
                </div>

                <div class="fh-card-description">
                    Review accepted candidates and generate
                    structured, CV-grounded interview questions.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            if st.button(
                "Open Interviews",
                use_container_width=True,
            ):
                navigate("Interviews")

    st.write("")

    section_header(
        2,
        "Active role",
    )

    if current_job:

        with st.container(border=True):

            st.markdown(
                f"""
                <div class="fh-card-title">
                    {safe(current_job.get("title", "Untitled role"))}
                </div>

                <div class="fh-card-description">
                    Job ID: #{safe(job_id)}
                    · Status: {safe(status)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            description = (
                current_job.get(
                    "job_description"
                )
                or st.session_state.get(
                    "job_description",
                    "",
                )
            )

            if description:

                st.markdown(
                    f"""
                    <div style="
                        color:#9DB0C2;
                        font-size:0.84rem;
                        line-height:1.65;
                    ">
                        {safe(description)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.write("")

    section_header(
        3,
        "Next actions",
    )

    actions = st.columns(4)

    with actions[0]:
        if st.button(
            "Manage job",
            use_container_width=True,
        ):
            navigate("Jobs")

    with actions[1]:
        if st.button(
            "Review candidates",
            use_container_width=True,
        ):
            navigate("Candidates")

    with actions[2]:
        if st.button(
            "Interviews",
            use_container_width=True,
        ):
            navigate("Interviews")

    with actions[3]:
        if st.button(
            "Notifications",
            use_container_width=True,
        ):
            navigate("Notifications")


# ============================================================
# JOBS PAGE
# ============================================================

def render_jobs():

    render_page_header(
        "Jobs",
        "Create, review and manage recruiter job descriptions.",
    )

    render_workflow("Jobs")

    section_header(
        1,
        "Post a new job",
    )

    with st.container(border=True):

        title = st.text_input(
            "Job title",
            value="Data Analyst",
            key="job_title_input",
        )

        job_desc = st.text_area(
            "Job description",
            value=(
                "Looking for a Data Analyst with Python, SQL, "
                "and Power BI experience"
            ),
            height=160,
            key="job_description_input",
        )

        st.caption(
            "Run the fairness check before posting the role."
        )

        if st.button(
            "Check job description",
            type="primary",
            use_container_width=True,
        ):

            try:

                response = requests.post(
                    f"{JD_AGENT_URL}/check",
                    json={
                        "job_description": job_desc,
                        "title": title,
                    },
                    headers=AUTH_HEADERS,
                    timeout=REQUEST_TIMEOUT,
                )

                if response.status_code == 200:

                    st.session_state.jd_check_result = (
                        response.json()
                    )

                    st.success(
                        "Fairness check completed."
                    )

                else:

                    st.error(
                        "JD Fairness Agent returned an error "
                        f"(status {response.status_code}): "
                        f"{response.text[:400]}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not reach the JD Fairness Agent. "
                    "Make sure it is running on port 8004."
                )

            except requests.exceptions.RequestException as exc:

                st.error(
                    f"JD Fairness Agent request failed: {exc}"
                )

            except ValueError:

                st.error(
                    "The JD Fairness Agent returned an invalid response."
                )

    # --------------------------------------------------------
    # FAIRNESS RESULT
    # --------------------------------------------------------

    if "jd_check_result" in st.session_state:

        result = st.session_state.jd_check_result

        flags = result.get(
            "keyword_flags",
            [],
        )

        section_header(
            2,
            "Fairness check result",
        )

        with st.container(border=True):

            if flags:

                st.warning(
                    f"{len(flags)} potential fairness "
                    "concern(s) found."
                )

                for flag in flags:

                    st.markdown(
                        f"""
                        **{flag.get("term", "Unknown term")}**

                        Category:
                        `{flag.get("category", "Unknown")}`

                        — {flag.get("note", "")}
                        """
                    )

            else:

                st.success(
                    "No obvious fairness concerns detected."
                )

            st.write("")

            if st.button(
                "Post this job",
                type="primary",
                use_container_width=True,
            ):

                try:

                    response = requests.post(
                        f"{JD_AGENT_URL}/confirm-and-post",
                        json={
                            "title": title,
                            "job_description": job_desc,
                            "proceed_despite_flags": bool(flags),
                        },
                        headers=AUTH_HEADERS,
                        timeout=REQUEST_TIMEOUT,
                    )

                    if response.status_code == 200:

                        data = response.json()

                        st.session_state.job_id = data[
                            "job_id"
                        ]

                        st.session_state.job_description = (
                            job_desc
                        )

                        st.session_state.job_title = (
                            title
                        )

                        st.success(
                            f"Job posted successfully "
                            f"(job_id={st.session_state.job_id})."
                        )

                        # Important workflow navigation:
                        # POST JOB -> CANDIDATES
                        navigate("Candidates")

                    else:

                        st.error(
                            "Could not post the job: "
                            f"{response.text[:400]}"
                        )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Could not reach the JD Fairness Agent."
                    )

                except requests.exceptions.RequestException as exc:

                    st.error(
                        f"Job posting request failed: {exc}"
                    )

                except (ValueError, KeyError):

                    st.error(
                        "The JD Fairness Agent returned "
                        "an invalid posting response."
                    )

    # --------------------------------------------------------
    # ACTIVE JOB
    # --------------------------------------------------------

    current_job = get_active_job()

    if not current_job:
        return

    st.write("")

    section_header(
        3,
        "Active job",
    )

    job_id = st.session_state.get(
        "job_id"
    )

    status = current_job.get(
        "status",
        "open",
    )

    with st.container(border=True):

        col1, col2 = st.columns(
            [3, 1]
        )

        with col1:

            active_title = current_job.get(
                "title",
                st.session_state.get(
                    "job_title",
                    "Untitled role",
                ),
            )

            st.markdown(
                f"""
                <div class="fh-card-title">
                    {safe(active_title)}
                </div>

                <div class="fh-card-description">
                    Job ID: #{safe(job_id)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:

            if status == "open":

                st.markdown(
                    """
                    <div style="
                        text-align:right;
                        color:#86EFAC;
                        font-weight:700;
                        font-size:0.76rem;
                    ">
                        ● OPEN
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    """
                    <div style="
                        text-align:right;
                        color:#94A3B8;
                        font-weight:700;
                        font-size:0.76rem;
                    ">
                        ● CLOSED
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.write("")

        description = (
            current_job.get(
                "job_description"
            )
            or st.session_state.get(
                "job_description",
                "",
            )
        )

        if description:
            st.write(description)

    # --------------------------------------------------------
    # CLOSE JOB
    # --------------------------------------------------------

    if status != "closed":

        st.write("")

        with st.expander(
            "Close this job"
        ):

            st.caption(
                "Closing a job deletes PII for candidates "
                "whose retention preference is 'delete'. "
                "Other retention preferences are handled "
                "by the retention cleanup process."
            )

            if st.button(
                "Close job and delete eligible PII",
                key="close_job_btn",
            ):

                result = close_job(
                    job_id
                )

                deleted = result.get(
                    "pii_deleted_for",
                    [],
                )

                if deleted:

                    st.success(
                        "Job closed. Deleted PII for: "
                        + ", ".join(deleted)
                    )

                else:

                    st.success(
                        "Job closed. No candidates currently "
                        "had a 'delete' preference on file."
                    )

                navigate("Jobs")


# ============================================================
# CANDIDATES PAGE
# ============================================================

def render_candidates():

    render_page_header(
        "Candidates",
        "Process CVs, review explainable rankings and record decisions.",
    )

    render_workflow("Candidates")

    job_id = st.session_state.get(
        "job_id"
    )

    if not job_id:

        with st.container(border=True):

            st.info(
                "No active job. Create a job before "
                "processing candidates."
            )

            if st.button(
                "Go to Jobs",
                type="primary",
            ):
                navigate("Jobs")

        return

    current_job = get_job(
        job_id
    )

    job_status = (
        current_job.get("status", "open")
        if current_job
        else "open"
    )

    section_header(
        1,
        f"Candidate screening · Job #{job_id}",
    )

    # --------------------------------------------------------
    # CLOSED JOB NOTICE
    # --------------------------------------------------------

    if job_status == "closed":

        st.info(
            "This job is closed. Existing candidate data can "
            "still be reviewed, but the recruitment workflow "
            "should not be used to add new candidates."
        )

    # --------------------------------------------------------
    # REAL CV UPLOAD
    # --------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            """
            <div class="fh-card-title">
                Upload real CVs
            </div>

            <div class="fh-card-description">
                PDF or Word CVs are processed by the Reader Agent,
                then passed through the Matcher and Explainer agents.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        uploaded_files = st.file_uploader(
            "Candidate CVs",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="candidate_cv_upload",
        )

        if uploaded_files and st.button(
            "Process uploaded CVs",
            type="primary",
            use_container_width=True,
        ):

            processed_ids = []

            for i, uploaded_file in enumerate(
                uploaded_files
            ):

                candidate_id = (
                    f"CAND_{i + 1:03d}"
                )

                try:

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                        )
                    }

                    response = requests.post(
                        f"{READER_URL}/read-cv",
                        params={
                            "candidate_id": candidate_id,
                            "job_id": job_id,
                        },
                        files=files,
                        headers=AUTH_HEADERS,
                        timeout=REQUEST_TIMEOUT,
                    )

                    if response.status_code == 200:

                        processed_ids.append(
                            candidate_id
                        )

                    else:

                        st.error(
                            f"Reader Agent error for "
                            f"{uploaded_file.name}: "
                            f"{response.text[:400]}"
                        )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Could not reach the Reader Agent. "
                        "Make sure it is running on port 8001."
                    )

                    break

                except requests.exceptions.RequestException as exc:

                    st.error(
                        f"Reader Agent request failed: {exc}"
                    )

                    break

            # ------------------------------------------------
            # MATCH + EXPLAIN
            # ------------------------------------------------

            if processed_ids:

                st.success(
                    f"Processed {len(processed_ids)} CV(s): "
                    f"{', '.join(processed_ids)}"
                )

                from db import get_candidates_for_job

                real_candidates = (
                    get_candidates_for_job(
                        job_id
                    )
                )

                skills_by_id = {
                    c["candidate_id"]: c.get(
                        "skills",
                        []
                    )
                    for c in real_candidates
                }

                try:

                    job_description = (
                        st.session_state.get(
                            "job_description",
                            "",
                        )
                    )

                    matcher_response = requests.post(
                        f"{MATCHER_URL}/match",
                        json={
                            "job_description": job_description,
                            "candidates": [
                                {
                                    "candidate_id": c[
                                        "candidate_id"
                                    ],
                                    "skills": c.get(
                                        "skills",
                                        []
                                    ),
                                    "organizations": c.get(
                                        "organizations",
                                        []
                                    ),
                                }
                                for c in real_candidates
                            ],
                            "job_id": job_id,
                        },
                        headers=AUTH_HEADERS,
                        timeout=REQUEST_TIMEOUT,
                    )

                    if matcher_response.status_code != 200:

                        st.error(
                            "Matcher Agent returned an error: "
                            f"{matcher_response.status_code} - "
                            f"{matcher_response.text[:400]}"
                        )

                    else:

                        ranked = matcher_response.json()[
                            "ranked_candidates"
                        ]

                        explain_payload = {
                            "job_description": job_description,
                            "candidates": [
                                {
                                    **candidate,
                                    "skills": skills_by_id.get(
                                        candidate[
                                            "candidate_id"
                                        ],
                                        [],
                                    ),
                                }
                                for candidate in ranked
                            ],
                            "job_id": job_id,
                        }

                        explain_response = requests.post(
                            f"{EXPLAINER_URL}/explain",
                            json=explain_payload,
                            headers=AUTH_HEADERS,
                            timeout=REQUEST_TIMEOUT,
                        )

                        if (
                            explain_response.status_code
                            == 200
                        ):

                            st.success(
                                "Matching and explanation "
                                "completed."
                            )

                        else:

                            st.error(
                                "Explainer Agent returned "
                                f"an error: "
                                f"{explain_response.status_code} - "
                                f"{explain_response.text[:400]}"
                            )

                except requests.exceptions.ConnectionError as exc:

                    st.error(
                        f"Could not reach the Matcher/Explainer "
                        f"agent: {exc}"
                    )

                except requests.exceptions.RequestException as exc:

                    st.error(
                        f"Matching request failed: {exc}"
                    )

                except (ValueError, KeyError) as exc:

                    st.error(
                        f"Invalid response from matching pipeline: "
                        f"{exc}"
                    )

    # --------------------------------------------------------
    # SAMPLE CANDIDATES
    # --------------------------------------------------------

    with st.expander(
        "Use sample candidates for quick testing"
    ):

        st.caption(
            "This bypasses the Reader Agent and is useful "
            "for testing the Matcher and Explainer workflow."
        )

        if st.button(
            "Load 3 sample candidates",
            key="load_samples",
        ):

            sample_candidates = [
                {
                    "candidate_id": "CAND_001",
                    "skills": [
                        "python",
                        "sql",
                        "power bi",
                    ],
                },
                {
                    "candidate_id": "CAND_002",
                    "skills": [
                        "sql",
                        "excel",
                    ],
                },
                {
                    "candidate_id": "CAND_003",
                    "skills": [
                        "java",
                        "project management",
                    ],
                },
            ]

            skills_by_id = {
                c["candidate_id"]: c["skills"]
                for c in sample_candidates
            }

            try:

                matcher_response = requests.post(
                    f"{MATCHER_URL}/match",
                    json={
                        "job_description": st.session_state.get(
                            "job_description",
                            "",
                        ),
                        "candidates": sample_candidates,
                        "job_id": job_id,
                    },
                    headers=AUTH_HEADERS,
                    timeout=REQUEST_TIMEOUT,
                )

                if matcher_response.status_code != 200:

                    st.error(
                        "Matcher Agent returned an error: "
                        f"{matcher_response.status_code}"
                    )

                else:

                    ranked = matcher_response.json()[
                        "ranked_candidates"
                    ]

                    explain_payload = {
                        "job_description": st.session_state.get(
                            "job_description",
                            "",
                        ),
                        "candidates": [
                            {
                                **candidate,
                                "skills": skills_by_id.get(
                                    candidate[
                                        "candidate_id"
                                    ],
                                    [],
                                ),
                            }
                            for candidate in ranked
                        ],
                        "job_id": job_id,
                    }

                    explain_response = requests.post(
                        f"{EXPLAINER_URL}/explain",
                        json=explain_payload,
                        headers=AUTH_HEADERS,
                        timeout=REQUEST_TIMEOUT,
                    )

                    if explain_response.status_code == 200:

                        st.success(
                            "Sample candidates matched "
                            "and explained successfully."
                        )

                    else:

                        st.error(
                            "Explainer Agent returned an error: "
                            f"{explain_response.status_code}"
                        )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not reach one of the agents."
                )

            except requests.exceptions.RequestException as exc:

                st.error(
                    f"Agent request failed: {exc}"
                )

    # --------------------------------------------------------
    # RANKINGS
    # --------------------------------------------------------

    rankings = get_rankings_for_job(
        job_id
    )

    section_header(
        2,
        "Candidate rankings",
    )

    if not rankings:

        with st.container(border=True):

            st.info(
                "No candidates ranked yet. Upload CVs or "
                "load the sample candidates above."
            )

        return

    latest_decisions = get_latest_decisions(
        job_id
    )

    for rank in rankings:

        cand_id = rank[
            "candidate_id"
        ]

        current_decision = latest_decisions.get(
            cand_id
        )

        with st.container(border=True):

            col1, col2, col3 = st.columns(
                [2, 4, 2]
            )

            # ----------------------------------------------
            # SCORE
            # ----------------------------------------------

            with col1:

                st.markdown(
                    f"""
                    <div style="
                        color:#F8FAFC;
                        font-size:1rem;
                        font-weight:800;
                    ">
                        {safe(cand_id)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("")

                st.markdown(
                    score_badge(
                        rank["final_score"]
                    ),
                    unsafe_allow_html=True,
                )

                st.progress(
                    min(
                        max(
                            rank["final_score"],
                            0.0,
                        ),
                        1.0,
                    )
                )

            # ----------------------------------------------
            # EXPLANATION
            # ----------------------------------------------

            with col2:

                candidate_record = get_candidate(
                    cand_id
                )

                cv_summary = (
                    candidate_record.get(
                        "cv_summary"
                    )
                    if candidate_record
                    else None
                )

                if cv_summary:

                    st.markdown(
                        f"""
                        <div style="
                            color:#9CB0C2;
                            font-size:0.78rem;
                            line-height:1.5;
                            margin-bottom:0.6rem;
                        ">
                            <strong>CV summary:</strong>
                            {safe(cv_summary)}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                explanation = (
                    rank.get(
                        "explanation"
                    )
                    or "_No explanation generated yet._"
                )

                st.write(
                    explanation
                )

                matched = ", ".join(
                    rank.get(
                        "matched_skills",
                        [],
                    )
                ) or "-"

                missing = ", ".join(
                    rank.get(
                        "missing_skills",
                        [],
                    )
                ) or "-"

                st.markdown(
                    f"""
                    <div style="
                        margin-top:0.55rem;
                        color:#748BA0;
                        font-size:0.72rem;
                        line-height:1.55;
                    ">
                        <strong style="color:#8EA6BA;">
                            Matched:
                        </strong>
                        {safe(matched)}

                        <br>

                        <strong style="color:#8EA6BA;">
                            Missing:
                        </strong>
                        {safe(missing)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ----------------------------------------------
            # DECISION
            # ----------------------------------------------

            with col3:

                if current_decision:

                    decision_value = current_decision.get(
                        "decision"
                    )

                    if decision_value == "accepted":

                        st.success(
                            "Accepted"
                        )

                    elif decision_value == "rejected":

                        st.error(
                            "Rejected"
                        )

                    else:

                        st.info(
                            decision_value
                        )

                    st.caption(
                        "Decision already recorded."
                    )

                else:

                    if st.button(
                        "Accept",
                        key=f"accept_{cand_id}",
                        type="primary",
                        use_container_width=True,
                    ):

                        st.session_state[
                            f"decision_state_{cand_id}"
                        ] = (
                            "awaiting_interview_details"
                        )

                        st.rerun()

                    if st.button(
                        "Reject",
                        key=f"reject_{cand_id}",
                        use_container_width=True,
                    ):

                        log_decision(
                            cand_id,
                            job_id,
                            "rejected",
                            decided_by="recruiter",
                        )

                        st.session_state[
                            f"rejection_recorded_{cand_id}"
                        ] = True

                        # Reject -> Notifications
                        navigate(
                            "Notifications"
                        )

            # ----------------------------------------------
            # ACCEPT FLOW
            # ----------------------------------------------

            decision_state = st.session_state.get(
                f"decision_state_{cand_id}"
            )

            if (
                decision_state
                == "awaiting_interview_details"
            ):

                st.divider()

                st.markdown(
                    """
                    <div class="fh-card-title">
                        Interview details
                    </div>

                    <div class="fh-card-description">
                        These details are stored with the
                        acceptance decision and can be used
                        by the notification workflow.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                i_date = st.date_input(
                    "Interview date",
                    key=f"date_{cand_id}",
                )

                i_time = st.time_input(
                    "Interview time",
                    key=f"time_{cand_id}",
                )

                i_location = st.text_input(
                    "Interview location",
                    placeholder=(
                        "Office - 3rd floor or video call link"
                    ),
                    key=f"loc_{cand_id}",
                )

                if st.button(
                    "Confirm acceptance",
                    key=f"confirm_{cand_id}",
                    type="primary",
                ):

                    log_decision(
                        cand_id,
                        job_id,
                        "accepted",
                        interview_date=str(i_date),
                        interview_time=str(i_time),
                        interview_location=i_location,
                        decided_by="recruiter",
                    )

                    st.session_state[
                        f"decision_state_{cand_id}"
                    ] = "done"

                    # Accept -> Interviews
                    navigate(
                        "Interviews"
                    )


# ============================================================
# INTERVIEWS PAGE
# ============================================================

def render_interviews():

    render_page_header(
        "Interviews",
        "Manage accepted candidates and generate structured interview questions.",
    )

    render_workflow("Interviews")

    job_id = st.session_state.get(
        "job_id"
    )

    if not job_id:

        with st.container(border=True):

            st.info(
                "No active job is selected."
            )

            if st.button(
                "Go to Jobs",
                type="primary",
            ):
                navigate("Jobs")

        return

    current_job = get_job(
        job_id
    )

    section_header(
        1,
        "Accepted candidates",
    )

    decisions = get_decisions_for_job(
        job_id
    )

    accepted_decisions = [
        decision
        for decision in decisions
        if decision.get("stage") == "shortlist"
        and decision.get("decision") == "accepted"
    ]

    if not accepted_decisions:

        with st.container(border=True):

            st.info(
                "No candidates have been accepted for "
                "interview yet."
            )

            if st.button(
                "Review candidates",
                type="primary",
            ):
                navigate("Candidates")

        return

    st.caption(
        "Interview questions are grounded in the candidate's "
        "submitted CV whenever CV text is available."
    )

    for decision in accepted_decisions:

        cand_id = decision[
            "candidate_id"
        ]

        questions_key = (
            f"interview_questions_"
            f"{job_id}_"
            f"{cand_id}"
        )

        with st.container(border=True):

            interview_date = (
                decision.get(
                    "interview_date"
                )
                or "TBD"
            )

            interview_time = (
                decision.get(
                    "interview_time"
                )
                or "TBD"
            )

            interview_location = (
                decision.get(
                    "interview_location"
                )
                or "Not specified"
            )

            st.markdown(
                f"""
                <div class="fh-card-title">
                    {safe(cand_id)}
                </div>

                <div class="fh-card-description">
                    Interview:
                    {safe(interview_date)}
                    ·
                    {safe(interview_time)}
                    ·
                    {safe(interview_location)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            if st.button(
                "Generate interview questions",
                key=f"gen_q_{cand_id}",
                type="primary",
            ):

                try:

                    response = requests.post(
                        f"{EXPLAINER_URL}/interview-questions",
                        json={
                            "candidate_id": cand_id,
                            "job_id": job_id,
                        },
                        headers=AUTH_HEADERS,
                        timeout=REQUEST_TIMEOUT,
                    )

                    if response.status_code == 200:

                        st.session_state[
                            questions_key
                        ] = response.json()

                    else:

                        st.error(
                            "Explainer Agent returned an error: "
                            f"{response.status_code} - "
                            f"{response.text[:400]}"
                        )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Could not reach the Explainer Agent. "
                        "Make sure it is running on port 8003."
                    )

                except requests.exceptions.RequestException as exc:

                    st.error(
                        f"Interview question request failed: {exc}"
                    )

            if questions_key in st.session_state:

                data = st.session_state[
                    questions_key
                ]

                if not data.get(
                    "cv_based"
                ):

                    st.warning(
                        "No stored CV text was found. "
                        "Questions are based on the candidate's "
                        "skill list only."
                    )

                elif not data.get(
                    "projects_found"
                ):

                    st.info(
                        "No distinct Projects section was detected "
                        "in this CV. Questions are based on the "
                        "full CV text instead."
                    )

                questions = data.get(
                    "questions",
                    [],
                )

                st.write("")

                for i, question in enumerate(
                    questions,
                    start=1,
                ):

                    st.markdown(
                        f"""
                        <div style="
                            margin-bottom:0.65rem;
                            padding:0.75rem 0.85rem;
                            background:#0A1725;
                            border:1px solid #1D344A;
                            border-radius:9px;
                            color:#DCE7F0;
                            font-size:0.84rem;
                            line-height:1.5;
                        ">
                            <strong style="color:#5EEAD4;">
                                {i}.
                            </strong>
                            {safe(question)}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                job_title = (
                    data.get(
                        "job_title"
                    )
                    or (
                        current_job.get(
                            "title",
                            "the role",
                        )
                        if current_job
                        else "the role"
                    )
                )

                pdf_bytes = (
                    build_interview_questions_pdf(
                        candidate_id=cand_id,
                        job_title=job_title,
                        interview_date=decision.get(
                            "interview_date"
                        ),
                        interview_time=decision.get(
                            "interview_time"
                        ),
                        questions=questions,
                        cv_based=data.get(
                            "cv_based",
                            False,
                        ),
                    )
                )

                st.download_button(
                    "Download interview questions as PDF",
                    data=pdf_bytes,
                    file_name=(
                        f"interview_questions_"
                        f"{cand_id}.pdf"
                    ),
                    mime="application/pdf",
                    key=f"download_q_{cand_id}",
                    use_container_width=True,
                )


# ============================================================
# NOTIFICATIONS PAGE
# ============================================================

def render_notifications():

    render_page_header(
        "Notifications",
        "Preview and send pending candidate communications.",
    )

    render_workflow("Notifications")

    job_id = st.session_state.get(
        "job_id"
    )

    if not job_id:

        with st.container(border=True):

            st.info(
                "No active job is selected."
            )

            if st.button(
                "Go to Jobs",
                type="primary",
            ):
                navigate("Jobs")

        return

    section_header(
        1,
        "Candidate notifications",
    )

    with st.container(border=True):

        st.markdown(
            """
            <div class="fh-card-title">
                Batch notification workflow
            </div>

            <div class="fh-card-description">
                One notification is generated for each candidate
                with a logged decision that has not yet been notified.
                Accepted candidates receive interview information;
                rejected candidates receive a rejection message.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        actually_send = st.checkbox(
            "Actually send emails via SMTP",
            value=False,
            help=(
                "Leave unchecked to preview generated "
                "messages without sending them."
            ),
        )

        if actually_send:

            st.warning(
                "Email sending is enabled. The Notification "
                "Agent will attempt to send messages."
            )

        else:

            st.info(
                "Preview mode is enabled. No email will be sent."
            )

        if st.button(
            "Process pending notifications",
            type="primary",
            use_container_width=True,
        ):

            try:

                response = requests.post(
                    f"{NOTIFICATION_URL}/notify-all-for-job",
                    json={
                        "job_id": job_id,
                        "send": actually_send,
                    },
                    headers=AUTH_HEADERS,
                    timeout=REQUEST_TIMEOUT,
                )

                if response.status_code != 200:

                    st.error(
                        "Notification Agent returned an error: "
                        f"{response.status_code} - "
                        f"{response.text[:400]}"
                    )

                else:

                    data = response.json()

                    notifications = data.get(
                        "notifications",
                        [],
                    )

                    if not notifications:

                        st.info(
                            "No pending notifications. "
                            "Every logged decision has already "
                            "been notified."
                        )

                    else:

                        st.success(
                            f"Processed "
                            f"{len(notifications)} "
                            f"notification(s)."
                        )

                        for item in notifications:

                            candidate_id = item.get(
                                "candidate_id",
                                "Unknown",
                            )

                            status = item.get(
                                "status",
                                "Unknown",
                            )

                            with st.expander(
                                f"{candidate_id} · {status}"
                            ):

                                st.text(
                                    item.get(
                                        "email_text",
                                        "",
                                    )
                                )

                                send_result = item.get(
                                    "send_result"
                                )

                                if send_result is None:

                                    st.caption(
                                        "Preview only - not sent."
                                    )

                                elif send_result.get(
                                    "sent"
                                ):

                                    st.caption(
                                        "Email sent successfully."
                                    )

                                else:

                                    st.caption(
                                        "Send failed: "
                                        + str(
                                            send_result.get(
                                                "error",
                                                "Unknown error",
                                            )
                                        )
                                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not reach the Notification Agent. "
                    "Make sure it is running on port 8005."
                )

            except requests.exceptions.RequestException as exc:

                st.error(
                    f"Notification request failed: {exc}"
                )

    st.write("")

    section_header(
        2,
        "Retention cleanup",
    )

    with st.container(border=True):

        st.markdown(
            """
            <div class="fh-card-title">
                Privacy retention cleanup
            </div>

            <div class="fh-card-description">
                Removes PII when retention windows have expired.
                In production this should normally run automatically
                through a scheduled job.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "Run retention cleanup now",
        ):

            result = purge_expired_retention()

            deleted = result.get(
                "pii_deleted_for",
                [],
            )

            if deleted:

                st.success(
                    "Deleted PII for: "
                    + ", ".join(deleted)
                )

            else:

                st.info(
                    "Nothing to clean up right now."
                )


# ============================================================
# ANALYTICS PAGE
# ============================================================

def render_analytics():

    render_page_header(
        "Analytics",
        "Operational metrics for the current recruitment workflow.",
    )

    job_id = st.session_state.get(
        "job_id"
    )

    if not job_id:

        with st.container(border=True):

            st.info(
                "No active job is available for analytics."
            )

            if st.button(
                "Go to Jobs",
                type="primary",
            ):
                navigate("Jobs")

        return

    rankings = get_rankings_for_job(
        job_id
    )

    decisions = get_decisions_for_job(
        job_id
    )

    accepted = [
        d for d in decisions
        if d.get("decision") == "accepted"
    ]

    rejected = [
        d for d in decisions
        if d.get("decision") == "rejected"
    ]

    section_header(
        1,
        "Screening overview",
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_kpi(
            "Ranked candidates",
            len(rankings),
            "Matcher output",
        )

    with c2:
        render_kpi(
            "Accepted",
            len(accepted),
            "Interview decisions",
        )

    with c3:
        render_kpi(
            "Rejected",
            len(rejected),
            "Recruiter decisions",
        )

    with c4:
        render_kpi(
            "Decisions",
            len(decisions),
            "Recorded decisions",
        )

    st.write("")

    section_header(
        2,
        "Decision distribution",
    )

    if not decisions:

        with st.container(border=True):

            st.info(
                "No recruiter decisions have been recorded yet."
            )

    else:

        total = len(decisions)

        accepted_pct = (
            len(accepted) / total * 100
            if total
            else 0
        )

        rejected_pct = (
            len(rejected) / total * 100
            if total
            else 0
        )

        col1, col2 = st.columns(2)

        with col1:

            with st.container(border=True):

                st.markdown(
                    """
                    <div class="fh-card-title">
                        Accepted decisions
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.metric(
                    "Count",
                    len(accepted),
                )

                st.progress(
                    accepted_pct / 100
                    if total
                    else 0
                )

                st.caption(
                    f"{accepted_pct:.1f}% of recorded decisions"
                )

        with col2:

            with st.container(border=True):

                st.markdown(
                    """
                    <div class="fh-card-title">
                        Rejected decisions
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.metric(
                    "Count",
                    len(rejected),
                )

                st.progress(
                    rejected_pct / 100
                    if total
                    else 0
                )

                st.caption(
                    f"{rejected_pct:.1f}% of recorded decisions"
                )

    st.write("")

    section_header(
        3,
        "Candidate score distribution",
    )

    if not rankings:

        st.info(
            "No ranking data is available yet."
        )

    else:

        for rank in rankings:

            candidate_id = rank.get(
                "candidate_id",
                "Unknown",
            )

            score = float(
                rank.get(
                    "final_score",
                    0,
                )
            )

            col1, col2 = st.columns(
                [2, 5]
            )

            with col1:

                st.markdown(
                    f"""
                    <div style="
                        color:#EAF2F8;
                        font-weight:700;
                        font-size:0.82rem;
                    ">
                        {safe(candidate_id)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:

                st.progress(
                    min(
                        max(
                            score,
                            0.0,
                        ),
                        1.0,
                    )
                )

                st.caption(
                    f"Final score: {score:.2f}"
                )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # --------------------------------------------------------
    # LOGIN
    # --------------------------------------------------------

    if not st.session_state.logged_in:

        render_login()

        return

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    render_sidebar()

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    render_hero()

    # --------------------------------------------------------
    # CURRENT PAGE
    # --------------------------------------------------------

    page = st.session_state.get(
        "page",
        "Dashboard",
    )

    if page == "Dashboard":

        render_dashboard()

    elif page == "Jobs":

        render_jobs()

    elif page == "Candidates":

        render_candidates()

    elif page == "Interviews":

        render_interviews()

    elif page == "Notifications":

        render_notifications()

    elif page == "Analytics":

        render_analytics()

    else:

        st.session_state.page = "Dashboard"

        st.rerun()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()