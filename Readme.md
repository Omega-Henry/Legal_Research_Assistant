# Legal Research Assistant (DE) — StGB + pgvector + Azure RAG

> **Status:** Prototype for demo (local run).  
> **Scope:** German Criminal Code (StGB) ingested; vector search via pgvector; RAG answer API via Azure Chat model.  
> **Disclaimer:** Not legal advice.

---

## 0) Prerequisites

- **Python** 3.12
- **PostgreSQL (managed)**: Azure Database for PostgreSQL (Flexible Server) with `pgvector` enabled by support
- **psql client** (Ubuntu):
  ```bash
  sudo apt update
  sudo apt install -y postgresql-client

## 1) Project Layout

```bash
Legal_Research_Assistant/
├─ app/
│  └─ api.py                 # FastAPI: POST /ask → {answer, citations}
├─ db/
│  └─ schema.sql             # schemas, tables, indexes (pgvector)
├─ embed/
│  ├─ embed_all_stgb.py      # creates embeddings NDJSON with Azure
│  └─ insert_chunks.py  # bulk inserts (resumable)
├─ query/
│  └─ search.py              # quick CLI vector search
├─ data/
│  ├─ interim/               # NDJSON (cleaned sections)
│  └─ processed/             # NDJSON + embeddings
├─ .env.example              # template for secrets (safe to commit)
├─ requirements.txt          # frozen dependencies
└─ README.md
```

## 2) Setup 
```bash
cd /path/to/Legal_Research_Assistant

# 1) Python venv
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
python -V

# 2) Install deps
pip install -U pip
pip install -r requirements.txt

# 3) Copy and edit env
cp .env.example .env
```
## 3) Database bootstrap

### 3.1 Create schema & indexes
```bash
psql "host=$PGHOST port=$PGPORT dbname=$PGDATABASE user=$PGUSER password=$PGPASSWORD sslmode=$PGSSLMODE" \
  -f db/schema.sql
  ```
## ## 3) Database bootstrap

## 4) Quick search test

```bash 
python query/search.py
```
### Example:

```bash
Query: Welche Vorschrift regelt Diebstahl?
Top 10 Treffer:
 § 242 Diebstahl  |  similarity=0.870
 § 248a Diebstahl und Unterschlagung geringwertiger Sachen  |  similarity=0.863
 § 243 Besonders schwerer Fall des Diebstahls  |  similarity=0.861
 ...
```

## 5) Run the RAG API (local)

### Start server:

```bash
python -m uvicorn app.api:app --reload --port 8000
```
### Query it:

```bash 
curl -sS -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Welche Vorschrift regelt Diebstahl und typische Qualifikationen?","k":8,"law":"StGB"}' | jq
```
### Response:
```bash
{
  "answer": "Die Vorschrift, die den Diebstahl regelt, ist § 242 ... (Zitate) ...",
  "citations": [
    {"section_number":"242","section_title":"Diebstahl","similarity":0.85},
    {"section_number":"243","section_title":"Besonders schwerer Fall des Diebstahls","similarity":0.85},
    ...
  ]
}
```

### HEalth check:

```bash
curl -sS http://127.0.0.1:8000/healthz
# {"ok": true}
```
# 6) Environment variables in Python

### Scripts always load .env from the repo root:

> from pathlib import Path
> from dotenv import load_dotenv
>load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# 7) Troubleshooting

- ***404 Resource not found***
   → Endpoint family mismatch. Use either classic (…openai.azure.com) or Foundry (…models.ai.azure.com) consistently with key + deployments.

- ***401 Access denied***
   → Key doesn’t belong to endpoint. Copy both from same page in Azure.

- ***KeyError: 'AZURE_OPENAI_ENDPOINT'***
    → .env not loaded. Ensure it exists at repo root and load_dotenv points to it.

- ***psql not found***
    → Install: sudo apt install -y postgresql-client.

- ***ivfflat index created with little data***
    → Normal with empty table; rebuild index after bulk load.

# 8) Roadmap (future work)

- Ingest BGB + court judgments

- Hybrid retrieval / reranking

- Simple frontend (GitHub Pages) with chat UI

- Deploy API on Azure Container Apps

- Observability: log queries, latency