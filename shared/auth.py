"""
FairHire - Shared Authentication Module
------------------------------------------
Security feature: every agent-to-agent call (Dashboard -> Reader,
Dashboard -> Matcher, Dashboard -> Explainer, etc.) must present a shared
API key in the 'X-API-Key' header. Without this, anyone who can reach an
agent's port directly (bypassing the Dashboard entirely) could call
/read-cv, /notify, /match, etc. with no restriction at all - a real gap
for a system that handles candidate PII.

Setup (.env, same value in EVERY agent's .env AND the Dashboard's .env):
    FAIRHIRE_API_KEY=<a long random string>

Generate one with:
    python -c "import secrets; print(secrets.token_urlsafe(32))"

If FAIRHIRE_API_KEY is not set anywhere, auth is skipped entirely (so the
system still runs for local development/testing) - but a warning is
printed at import time so this is never silently insecure without you
knowing about it.
"""

import os
import warnings
from fastapi import Header, HTTPException
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("FAIRHIRE_API_KEY", "")

if not API_KEY:
    warnings.warn(
        "FAIRHIRE_API_KEY is not set - agent-to-agent authentication is "
        "DISABLED. Any endpoint protected by verify_api_key() will accept "
        "requests with no credentials at all. Set FAIRHIRE_API_KEY in your "
        ".env file (the SAME value in every agent's .env and the "
        "Dashboard's .env) before treating this as production-ready.",
        stacklevel=2,
    )


def verify_api_key(x_api_key: str = Header(default=None)):
    """
    FastAPI dependency - add `Depends(verify_api_key)` to any endpoint that
    should require the shared API key. Health-check endpoints are
    deliberately left unprotected so monitoring/uptime checks don't need
    credentials. Candidate-facing public links (e.g. the notification
    agent's /candidate-preference, which a real candidate clicks from
    their email with no way to attach a header) are also deliberately
    NOT protected by this - see the comment at that endpoint for why.
    """
    if not API_KEY:
        return  # auth disabled - see warning above
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")