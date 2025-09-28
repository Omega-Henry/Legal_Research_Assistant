# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os, psycopg2, requests, textwrap
from pathlib import Path
from dotenv import load_dotenv

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

app = FastAPI(title="Legal RAG (DE)")

class AskReq(BaseModel):
    question: str
    k: int = 8
    law: str = "StGB"

class Cite(BaseModel):
    section_number: str
    section_title: str
    similarity: float

class AskResp(BaseModel):
    answer: str
    citations: List[Cite]

# ---------------------------
# Embeddings + Retrieval
# ---------------------------
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

def retrieve(question: str, k: int, law: str):
    qvec = vec_str(embed(question))
    conn = psycopg2.connect(**DB)
    with conn, conn.cursor() as cur:
        cur.execute("""
            SELECT c.section_number, c.section_title, c.full_text,
                   1 - (c.embedding <=> %s::vector) AS sim
            FROM legal.chunks c
            JOIN legal.documents d ON d.id = c.document_id
            WHERE d.law_abbr = %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s;
        """, (qvec, law, qvec, k))
        rows = cur.fetchall()
    conn.close()
    return [
        {
            "section_number": r[0],
            "section_title": r[1] or "",
            "text": r[2],
            "similarity": float(r[3])
        }
        for r in rows
    ]

def build_context(docs, max_chars=8000):
    """Compact, readable Kontextblöcke für das LLM."""
    parts, used = [], 0
    for d in docs:
        header = f"§ {d['section_number']} {d['section_title']}".strip()
        # Kurzer Auszug, um Token zu sparen – Langtext ist im DB abrufbar
        snippet = textwrap.shorten(" ".join(d["text"].split()), width=1200, placeholder=" …")
        chunk = f"{header}\n{snippet}"
        if used + len(chunk) > max_chars:
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n\n---\n\n".join(parts)

# ---------------------------
# Chat / LLM
# ---------------------------

# Finaler deutscher Prompt: Antwort zuerst, dann Quellenliste
SYSTEM_PROMPT_DE = (
    "Du bist ein präziser juristischer Assistent für das deutsche Strafrecht (StGB). "
    "ANTWORTE ZUERST kurz und klar auf DEUTSCH (2–6 Sätze). "
    "DANN liste die QUELLEN unter der Überschrift „Quellen:“ – je Zeile im Format "
    "„§ <Nummer> <Titel>“. "
    "Nutze AUSSCHLIESSLICH den bereitgestellten Kontext. "
    "Wenn der Kontext nicht ausreicht, sage das knapp und schlage vor, wie man die Frage präzisieren kann. "
    "Beende deine Antwort mit: „Keine Rechtsberatung. Bitte im Gesetzestext prüfen.“"
)

def ask_llm(question: str, context: str):
    url = f"{AOAI_ENDPOINT}/openai/deployments/{CHAT_DEPLOY}/chat/completions?api-version={API_VER}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_DE},
        {
            "role": "user",
            "content": (
                f"Frage:\n{question}\n\n"
                f"Kontext (nur verwenden):\n{context}\n\n"
                "Formatiere GENAU so:\n"
                "<Antwort in 2–6 Sätzen>\n\n"
                "Quellen:\n"
                "- § <Nummer> <Titel>\n"
                "- § <Nummer> <Titel>"
            ),
        },
    ]
    r = requests.post(
        url,
        headers={"api-key": AOAI_API_KEY, "Content-Type": "application/json"},
        json={"messages": messages, "temperature": 0.2, "max_tokens": 500},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ---------------------------
# API
# ---------------------------

@app.post("/ask", response_model=AskResp)
def ask(body: AskReq):
    docs = retrieve(body.question, k=body.k, law=body.law)
    ctx  = build_context(docs)
    ans  = ask_llm(body.question, ctx)

    cits = [
        Cite(
            section_number=d["section_number"],
            section_title=d["section_title"],
            similarity=d["similarity"]
        )
        for d in docs
    ]
    return AskResp(answer=ans, citations=cits)
