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
│  └─ insert_chunks_fast.py  # bulk inserts (resumable)
├─ query/
│  └─ search.py              # quick CLI vector search
├─ data/
│  ├─ interim/               # NDJSON (cleaned sections)
│  └─ processed/             # NDJSON + embeddings
├─ .env.example              # template for secrets (safe to commit)
├─ requirements.txt          # frozen dependencies
└─ README.md
---
