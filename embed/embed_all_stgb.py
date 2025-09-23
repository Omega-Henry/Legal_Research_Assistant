# save as embed_all_stgb.py and run:
#   python embed_all_stgb.py --input data/interim/stgb_sections.ndjson --out data/processed/stgb_sections_with_vecs.ndjson

import os, json, math, time, argparse, sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import requests

# ---------- config ----------
BATCH_SIZE = 64          # safe batch for embeddings
MAX_RETRIES = 5          # for 429/5xx
RETRY_BASE_SEC = 2       # exponential backoff base

# ---------- utils ----------
def eprint(*a, **k): print(*a, file=sys.stderr, **k)

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na*nb + 1e-12)

def load_ndjson(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def save_ndjson(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def call_azure_embeddings(texts: List[str], endpoint: str, deployment: str, api_key: str, api_version: str) -> List[List[float]]:
    url = f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    payload = {"input": texts}

    for attempt in range(1, MAX_RETRIES+1):
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json()["data"]
            # sort by index to preserve order
            data_sorted = sorted(data, key=lambda d: d["index"])
            return [d["embedding"] for d in data_sorted]

        if r.status_code in (429, 500, 502, 503, 504):
            wait = RETRY_BASE_SEC ** attempt
            eprint(f"[warn] embeddings HTTP {r.status_code} (attempt {attempt}/{MAX_RETRIES}) → sleep {wait}s")
            time.sleep(wait)
            continue

        # hard error
        eprint("[error] embeddings failed:", r.status_code, r.text[:500])
        r.raise_for_status()

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")

# ---------- main ----------
def main():
    load_dotenv()  # expects .env in project root
    endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key    = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_EMBED_DEPLOYMENT")
    api_ver    = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")

    if not all([endpoint, api_key, deployment]):
        raise RuntimeError("Missing Azure env vars. Check .env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_EMBED_DEPLOYMENT.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to NDJSON with sections")
    parser.add_argument("--out", required=True, help="output NDJSON (will include an 'embedding' field)")
    parser.add_argument("--resume", action="store_true", help="resume if output exists (skip already embedded)")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.out)

    rows = load_ndjson(in_path)
    eprint(f"[info] loaded {len(rows)} sections from {in_path}")

    # resume logic: if out exists and resume flag used, load done rows into a dict
    done = {}
    out_rows: List[Dict[str, Any]] = []
    if args.resume and out_path.exists():
        eprint(f"[info] resume mode: reading existing {out_path}")
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                key = f"{r.get('law_abbr')}-{r.get('section_number')}"
                done[key] = r
        eprint(f"[info] already embedded: {len(done)}")
        # we will rewrite final file at the end; keep them in memory for now

    # process in batches
    dim = None
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]

        # filter out records already done (resume)
        todo = []
        for r in batch:
            key = f"{r.get('law_abbr')}-{r.get('section_number')}"
            if args.resume and key in done:
                out_rows.append(done[key])
            else:
                todo.append(r)

        if not todo:
            eprint(f"[skip] {i}-{i+len(batch)} already embedded")
            continue

        texts = [r["full_text"] for r in todo]
        embs = call_azure_embeddings(texts, endpoint, deployment, api_key, api_ver)

        if dim is None and embs:
            dim = len(embs[0])
            eprint(f"[ok] embedding dim = {dim}")

        # attach embeddings
        for r, e in zip(todo, embs):
            r_out = dict(r)
            r_out["embedding"] = e
            out_rows.append(r_out)

        eprint(f"[ok] embedded {len(todo)} rows ({i+len(batch)}/{len(rows)})")

    # write all results
    save_ndjson(out_path, out_rows)
    eprint(f"[done] wrote {len(out_rows)} rows to {out_path}")

    # quick interactive test
    try:
        q = "Wie lang ist die Kündigungsfrist bei einem Arbeitsverhältnis?"
        eprint(f"[test] query: {q}")
        q_emb = call_azure_embeddings([q], endpoint, deployment, api_key, api_ver)[0]
        # score all
        scored = []
        for r in out_rows:
            scored.append((cosine(q_emb, r["embedding"]), r))
        scored.sort(key=lambda x: x[0], reverse=True)

        print("\nTop-Ergebnisse:")
        for score, r in scored[:5]:
            print(f"  score={score:.3f} | § {r['section_number']} {r['section_title']}")
    except Exception as ex:
        eprint(f"[warn] test query failed: {ex}")

if __name__ == "__main__":
    main()
