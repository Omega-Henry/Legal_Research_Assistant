# rag/answer.py
import os, psycopg2, requests, textwrap
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ----- Azure config -----
AOAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
AOAI_API_KEY  = os.environ["AZURE_OPENAI_API_KEY"]
EMB_DEPLOY    = os.environ["AZURE_EMBED_DEPLOYMENT"]        
CHAT_DEPLOY   = os.environ["AZURE_CHAT_DEPLOYMENT"]         
API_VER       = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")

# ----- DB connection -----
DB = dict(
    host=os.getenv("PGHOST"),
    port=os.getenv("PGPORT", "5432"),
    dbname=os.getenv("PGDATABASE"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
    sslmode=os.getenv("PGSSLMODE", "require"),
)

def embed(text: str):
    url = f"{AOAI_ENDPOINT}/openai/deployments/{EMB_DEPLOY}/embeddings?api-version={API_VER}"
    r = requests.post(url,
                      headers={"api-key": AOAI_API_KEY, "Content-Type": "application/json"},
                      json={"input": [text]}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Embeddings {r.status_code}: {r.text}")
    return r.json()["data"][0]["embedding"]

def vec_str(v):  # pgvector literal
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"

def retrieve(query: str, k: int = 8, law="StGB"):
    qvec = vec_str(embed(query))
    conn = psycopg2.connect(**DB)
    with conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.section_number, c.section_title, c.full_text,
                   1 - (c.embedding <=> %s::vector) AS similarity
            FROM legal.chunks c
            JOIN legal.documents d ON d.id = c.document_id
            WHERE d.law_abbr = %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s;
            """,
            (qvec, law, qvec, k)
        )
        rows = cur.fetchall()
    conn.close()
    # shape into small dicts
    docs = [{
        "sec": r[0],
        "title": r[1] or "",
        "text": r[2],
        "sim": float(r[3]),
    } for r in rows]
    return docs

def build_context(docs, max_chars=8000):
    #  keep § header + a short snippet per section
    parts, used = [], 0
    for d in docs:
        header = f"§ {d['sec']} {d['title']}".strip()
        body   = d["text"]
        snippet = textwrap.shorten(" ".join(body.split()), width=1200, placeholder=" …")
        chunk = f"{header}\n{snippet}"
        if used + len(chunk) > max_chars:
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n\n---\n\n".join(parts)

def ask_llm(question: str, context: str):
    url = f"{AOAI_ENDPOINT}/openai/deployments/{CHAT_DEPLOY}/chat/completions?api-version={API_VER}"
    system = (
        "Du bist ein vorsichtiger juristischer Assistent (DE). "
        "Antworte präzise in Deutsch und zitiere immer die relevanten Paragraphen "
        "aus dem Kontext als (§ Nummer – Titel). Antworte nicht außerhalb des Kontextes; "
        "wenn der Kontext nicht reicht, sage knapp, was fehlt."
    )
    user = (
        f"Frage:\n{question}\n\n"
        f"Kontextauszüge (deutsche Gesetzestexte):\n{context}\n\n"
        "Anweisung: Gib eine kurze, sachliche Antwort mit Zitaten in Klammern, "
        "z.B. (§ 242 – Diebstahl)."
    )
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": 0.2,
        "max_tokens": 450,
    }
    r = requests.post(url,
                      headers={"api-key": AOAI_API_KEY, "Content-Type": "application/json"},
                      json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Chat {r.status_code}: {r.text}")
    return r.json()["choices[0]"]["message"]["content"] if "choices[0]" in r.text else r.json()["choices"][0]["message"]["content"]

def answer(question: str, k=8):
    docs = retrieve(question, k=k, law="StGB")
    ctx  = build_context(docs)
    out  = ask_llm(question, ctx)
    # Show quick citations list for debugging
    cites = [f"§ {d['sec']} {d['title']}" for d in docs[:k]]
    return out, cites

if __name__ == "__main__":
    q = "Welche Vorschrift regelt Diebstahl und was sind typische Qualifikationen?"
    text, cites = answer(q, k=8)
    print("\nAntwort:\n", text)
    print("\nKontext-Quellen (Top-K):")
    for c in cites:
        print(" -", c)
