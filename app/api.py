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

def embed(text: str):
    url = f"{AOAI_ENDPOINT}/openai/deployments/{EMB_DEPLOY}/embeddings?api-version={API_VER}"
    r = requests.post(url, headers={"api-key": AOAI_API_KEY, "Content-Type":"application/json"},
                      json={"input":[text]}, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]

def vec_str(v): return "[" + ",".join(f"{x:.7f}" for x in v) + "]"

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
    return [{"section_number": r[0], "section_title": r[1] or "", "text": r[2], "similarity": float(r[3])} for r in rows]

def build_context(docs, max_chars=8000):
    parts, used = [], 0
    for d in docs:
        header = f"§ {d['section_number']} {d['section_title']}".strip()
        snippet = textwrap.shorten(" ".join(d["text"].split()), width=1200, placeholder=" …")
        chunk = f"{header}\n{snippet}"
        if used + len(chunk) > max_chars: break
        parts.append(chunk); used += len(chunk)
    return "\n\n---\n\n".join(parts)

def ask_llm(question: str, context: str):
    url = f"{AOAI_ENDPOINT}/openai/deployments/{CHAT_DEPLOY}/chat/completions?api-version={API_VER}"
    sys = ("Du bist ein vorsichtiger juristischer Assistent (DE). "
           "Antworte präzise in Deutsch und zitiere immer die relevanten Paragraphen "
           "aus dem Kontext als (§ Nummer – Titel). Wenn der Kontext nicht reicht, sag das klar.")
    user = (f"Frage:\n{question}\n\nKontextauszüge:\n{context}\n\n"
            "Anweisung: Kurze, sachliche Antwort mit Zitaten in Klammern, z. B. (§ 242 – Diebstahl).")
    r = requests.post(url, headers={"api-key": AOAI_API_KEY, "Content-Type":"application/json"},
                      json={"messages":[{"role":"system","content":sys},{"role":"user","content":user}],
                            "temperature":0.2, "max_tokens":450}, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

@app.post("/ask", response_model=AskResp)
def ask(body: AskReq):
    docs = retrieve(body.question, k=body.k, law=body.law)
    ctx  = build_context(docs)
    ans  = ask_llm(body.question, ctx)
    cits = [Cite(section_number=d["section_number"], section_title=d["section_title"], similarity=d["similarity"]) for d in docs]
    # Optional footer disclaimer
    ans += "\n\n*Hinweis: Keine Rechtsberatung. Angaben ohne Gewähr; prüfen Sie stets den Gesetzestext.*"
    return AskResp(answer=ans, citations=cits)
