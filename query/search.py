# query/search.py
import os
import psycopg2
import requests
from pathlib import Path
from dotenv import load_dotenv

# load .env from project root
load_dotenv(Path("/home/noe/Desktop/Ai_Legal research_assistant/.env"))

# --- Azure embeddings ---
ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY    = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOYMENT = os.environ["AZURE_EMBED_DEPLOYMENT"]  # your embedding deployment name
API_VER    = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")

def embed(text: str):
    url = f"{ENDPOINT}/openai/deployments/{DEPLOYMENT}/embeddings?api-version={API_VER}"
    r = requests.post(url, headers={"api-key": API_KEY, "Content-Type": "application/json"},
                      json={"input": [text]}, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]

def vec_str(v):
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"

def search(query: str, k: int = 5):
    qvec = vec_str(embed(query))

    conn = psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT", "5432"),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        sslmode=os.getenv("PGSSLMODE", "require"),
    )

    with conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT section_number,
                   section_title,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM legal.chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (qvec, qvec, k))
        rows = cur.fetchall()

    conn.close()
    return rows

if __name__ == "__main__":
    q = "Welche Vorschrift regelt Diebstahl?"
    top = search(q, k=5)
    print(f"\nQuery: {q}\nTop {len(top)} Treffer:")
    for sec, title, sim in top:
        print(f"  § {sec} {title}  |  similarity={sim:.3f}")
