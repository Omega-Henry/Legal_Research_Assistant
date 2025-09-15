# embed/test_embed_20.py
import os, json, math
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv(Path("/home/noe/Desktop/Ai_Legal research_assistant/.env"))

NDJSON = Path("/home/noe/Desktop/Ai_Legal research_assistant/data/processed/stgb_sections.ndjson")  # adjust if needed
K = 5  # top-k results

# Azure config
ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY    = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOYMENT = os.environ["AZURE_EMBED_DEPLOYMENT"]
API_VER    = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")

def embed_texts(texts):
    url = f"{ENDPOINT}/openai/deployments/{DEPLOYMENT}/embeddings?api-version={API_VER}"
    headers = {"Content-Type":"application/json", "api-key": API_KEY}
    payload = {"input": texts}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()["data"]
    # preserve original order by index
    data_sorted = sorted(data, key=lambda d: d["index"])
    embs = [d["embedding"] for d in data_sorted]
    # quick sanity
    dim = len(embs[0])
    print(f"[OK] Received {len(embs)} embeddings with dim={dim}")
    return embs

def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na * nb + 1e-12)

def main():
    # 1) load 20 sections
    rows = []
    with NDJSON.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 20: break
            rec = json.loads(line)
            rows.append(rec)

    texts = [r["full_text"] for r in rows]
    # 2) embed the 20 sections
    section_embs = embed_texts(texts)

    # 3) embed one query
    query = "Wie lang ist die Kündigungsfrist bei einem Arbeitsverhältnis?"
    q_emb = embed_texts([query])[0]

    # 4) score & print top-k
    scored = []
    for r, e in zip(rows, section_embs):
        scored.append((cosine_sim(q_emb, e), r))
    scored.sort(reverse=True, key=lambda x: x[0])

    print("\nTop-Ergebnisse:")
    for score, r in scored[:K]:
        print(f"  score={score:.3f} | § {r['section_number']} {r['section_title']}")

if __name__ == "__main__":
    main()
