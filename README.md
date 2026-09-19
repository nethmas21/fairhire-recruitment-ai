# fairhire-recruitment-ai

An explainable, bias-audited multi-agent recruitment assistant, built for
IT3041 — Information Retrieval and Web Analytics (SLIIT).

## What it does

FairHire reads CVs, ranks candidates against a job description using hybrid
information retrieval (semantic embeddings + BM25 keyword matching), explains
its rankings in plain English, audits for bias, drafts interview questions,
and notifies candidates by email — while a human recruiter always makes the
final decision.

## System architecture

Five agents:
1. **Reader Agent** — parses CVs (NER), extracts skills/experience, removes
   and vaults personal identifying information
2. **Matcher Agent** — hybrid retrieval (embeddings + BM25) ranks candidates
   against the job description
3. **Explainer Agent** — generates plain-English ranking rationale, audits
   for bias, generates tailored interview questions
4. **Dashboard Agent** — recruiter login, review shortlist, accept/reject,
   schedule interviews
5. **Notification Agent** — sends shortlist/rejection emails to every
   candidate who applies

See `/docs/master_spec.md` for the full technical specification.

## Tech stack

Python · FastAPI · spaCy · sentence-transformers · rank_bm25 · ChromaDB ·
SQLite · Streamlit · Claude/OpenAI API

## Project structure

```
fairhire-recruitment-ai/
├── README.md
├── requirements.txt
├── reader-agent/
│   └── reader_agent.py
├── matcher-agent/
│   └── matcher_agent.py
├── explainer-agent/
│   └── explainer_agent.py
├── dashboard/
│   └── dashboard.py
├── notification-agent/
│   └── notification_agent.py
├── database/
│   └── schema.sql
└── docs/
    ├── master_spec.md
    ├── architecture.png
    └── mid_eval_slides.pptx
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running an agent

Each agent can be run standalone for testing:
```bash
python reader-agent/reader_agent.py path/to/cv.pdf
python matcher-agent/matcher_agent.py
python notification-agent/notification_agent.py
```

## Contributors

| Name | Role |
|---|---|
| Nethma Weerasinghe | Matcher Agent, Explainer Agent, system integration |
| [Name] | Reader Agent |
| [Name] | Dashboard Agent |
| [Name] | Notification Agent |

## Course

IT3041 — Information Retrieval and Web Analytics
SLIIT — 2026 July intake