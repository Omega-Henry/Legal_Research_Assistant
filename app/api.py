# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Tuple
import os, psycopg2, requests, textwrap, re
from pathlib import Path
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# ---------- env ----------
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

AOAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
AOAI_API_KEY  = os.environ["AZURE_OPENAI_API_KEY"]
EMB_DEPLOY    = os.environ["AZURE_EMBED_DEPLOYMENT"]
CHAT_DEPLOY   = os.environ["AZURE_CHAT_DEPLOYMENT"]
API_VER       = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")

DB = dict(
    host=os.getenv("PGHOST"),
    port=os.getenv("PGPORT", "5432"),
    dbname=os.getenv("PGDATABASE"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
    sslmode=os.getenv("PGSSLMODE", "require"),
)

# ---------- app ----------
app = FastAPI(title="Legal RAG (DE)")

# CORS for GitHub Pages
GHPAGES_ORIGIN = "https://omega-henry.github.io"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[GHPAGES_ORIGIN],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"ok": True}

# ---------- models ----------
class AskReq(BaseModel):
    question: str
    k: int = 8
    law: str = "auto"   # <-- default: auto-detect & search across all if needed

class Cite(BaseModel):
    law_abbr: str
    section_number: str
    section_title: str
    similarity: float
    url: Optional[str] = None

class AskResp(BaseModel):
    answer: str
    citations: List[Cite]

# ---------- helpers ----------
def embed(text: str):
    url = f"{AOAI_ENDPOINT}/openai/deployments/{EMB_DEPLOY}/embeddings?api-version={API_VER}"
    r = requests.post(
        url,
        headers={"api-key": AOAI_API_KEY, "Content-Type": "application/json"},
        json={"input": [text]},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]

def vec_str(v): 
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"

LAW_HINTS = [
    "StGB", "HGB", "BGB", "StPO", "ZPO", "GG", "AO", "UWG", "StVG", "IfSG"
]
LAW_HINT_PATTERN = re.compile(r"\b(" + "|".join(LAW_HINTS) + r")\b", re.IGNORECASE)

def detect_law_hints(question: str) -> List[str]:
    """Return a prioritized list of law abbreviations found in the question."""
    hits = LAW_HINT_PATTERN.findall(question or "")
    # Normalize case: keep canonical uppercase
    hits = [h.upper() for h in hits]
    # Preserve order of first appearance, de-dup
    seen, ordered = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h); ordered.append(h)
    return ordered

def cite_url(law_abbr: str, section_number: str) -> str:
    # StGB -> stgb, "242a" etc. Build Gesetze-im-Internet § link
    num = "".join(ch for ch in (section_number or "") if ch.isalnum())
    return f"https://www.gesetze-im-internet.de/{law_abbr.lower()}/__{num}.html"

def rows_to_dicts(rows) -> List[dict]:
    # rows: (law_abbr, section_number, section_title, full_text, sim)
    out = []
    for r in rows:
        out.append({
            "law_abbr":       r[0],
            "section_number": r[1],
            "section_title":  r[2] or "",
            "text":           r[3],
            "similarity":     float(r[4]),
        })
    return out

# ---------- retrieval ----------
def retrieve_any(question: str, k: int) -> List[dict]:
    """Search across ALL laws."""
    qvec = vec_str(embed(question))
    conn = psycopg2.connect(**DB)
    with conn, conn.cursor() as cur:
        cur.execute("""
            SELECT d.law_abbr, c.section_number, c.section_title, c.full_text,
                   1 - (c.embedding <=> %s::vector) AS sim
            FROM legal.chunks c
            JOIN legal.documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s;
        """, (qvec, qvec, k))
        rows = cur.fetchall()
    conn.close()
    return rows_to_dicts(rows)

def retrieve_in_law(question: str, k: int, law: str) -> List[dict]:
    """Search within a specific law (e.g., 'StGB')."""
    qvec = vec_str(embed(question))
    conn = psycopg2.connect(**DB)
    with conn, conn.cursor() as cur:
        cur.execute("""
            SELECT d.law_abbr, c.section_number, c.section_title, c.full_text,
                   1 - (c.embedding <=> %s::vector) AS sim
            FROM legal.chunks c
            JOIN legal.documents d ON d.id = c.document_id
            WHERE d.law_abbr = %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s;
        """, (qvec, law, qvec, k))
        rows = cur.fetchall()
    conn.close()
    return rows_to_dicts(rows)

def retrieve_auto(question: str, k: int) -> Tuple[List[dict], str]:
    """
    Auto-detect: try hinted laws first; if weak (<0.65 best sim) or empty → global search.
    Returns (docs, strategy_info).
    """
    hints = detect_law_hints(question)
    # Try hinted laws in order of appearance
    for law in hints:
        docs = retrieve_in_law(question, k, law)
        if docs and docs[0]["similarity"] >= 0.65:
            return docs, f"hint:{law}"
    # Fallback: global search
    docs = retrieve_any(question, k)
    return docs, "global"

def build_context(docs, max_chars=8000):
    parts, used = [], 0
    for d in docs:
        header = f"{d['law_abbr']} § {d['section_number']} {d['section_title']}".strip()
        snippet = textwrap.shorten(" ".join(d["text"].split()), width=1200, placeholder=" …")
        chunk = f"{header}\n{snippet}"
        if used + len(chunk) > max_chars:
            break
        parts.append(chunk); used += len(chunk)
    return "\n\n---\n\n".join(parts)

# ---------- LLM ----------
SYSTEM_PROMPT_DE = (
    "Du bist ein präziser juristischer Assistent für deutsches Recht. "
    "ANTWORTE ZUERST kurz und klar auf DEUTSCH (2–6 Sätze). "
    "DANN liste die QUELLEN unter der Überschrift „Quellen:“ – je Zeile im Format "
    "„§ <Nummer> <Titel>“ (die Gesetzesabkürzung nicht wiederholen). "
    "Nutze AUSSCHLIESSLICH den bereitgestellten Kontext. "
    "Wenn der Kontext nicht ausreicht, sage das knapp und schlage vor, wie man die Frage präzisieren kann. "
    "Beende deine Antwort mit: „Keine Rechtsberatung. Bitte im Gesetzestext prüfen.“"
)

def ask_llm(question: str, context: str):
    url = f"{AOAI_ENDPOINT}/openai/deployments/{CHAT_DEPLOY}/chat/completions?api-version={API_VER}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_DE},
        {"role": "user", "content": (
            f"Frage:\n{question}\n\n"
            f"Kontext (nur verwenden):\n{context}\n\n"
            "Formatiere GENAU so:\n"
            "<Antwort in 2–6 Sätzen>\n\n"
            "Quellen:\n"
            "- § <Nummer> <Titel>\n"
            "- § <Nummer> <Titel>"
        )},
    ]
    r = requests.post(
        url,
        headers={"api-key": AOAI_API_KEY, "Content-Type": "application/json"},
        json={"messages": messages, "temperature": 0.2, "max_tokens": 500},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ---------- API ----------
@app.post("/ask", response_model=AskResp)
def ask(body: AskReq):
    # guardrails
    k = max(1, min(body.k, 12))
    law = (body.law or "").strip()

    # retrieval strategy
    if law.lower() == "auto" or law == "":
        docs, mode = retrieve_auto(body.question, k)
    else:
        docs = retrieve_in_law(body.question, k, law)
        mode = f"forced:{law}"

    ctx  = build_context(docs)
    ans  = ask_llm(body.question, ctx)

    # Build citations (with URLs)
    cits: List[Cite] = []
    for d in docs:
        cits.append(Cite(
            law_abbr=d["law_abbr"],
            section_number=d["section_number"],
            section_title=d["section_title"],
            similarity=d["similarity"],
            url=cite_url(d["law_abbr"], d["section_number"])
        ))

    return AskResp(answer=ans, citations=cits)
