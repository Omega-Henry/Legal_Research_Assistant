# embed/insert_chunks_fast.py
import os, json, time
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

# Load .env explicitly from project root
load_dotenv(Path("/home/noe/Desktop/Ai_Legal research_assistant/.env"))

IN_PATH   = Path("/home/noe/Desktop/Ai_Legal research_assistant/data/processed/stgb_sections_with_vecs.ndjson")
LAW_ABBR  = "StGB"
SOURCE_URI= "BJNR001270871.xml"
BATCH     = 100  # tune 50–200

def vec_str(v):  # pgvector-compatible string
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"

def main(limit=0):
    conn = psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT", "5432"),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        sslmode=os.getenv("PGSSLMODE", "require"),
    )
    conn.autocommit = False
    cur = conn.cursor()

    # Ensure documents row
    cur.execute("""
        INSERT INTO legal.documents (law_abbr, source_uri, lang)
        VALUES (%s, %s, %s)
        ON CONFLICT DO NOTHING
        RETURNING id;
    """, (LAW_ABBR, SOURCE_URI, "de"))
    row = cur.fetchone()
    if row is None:
        cur.execute("SELECT id FROM legal.documents WHERE law_abbr=%s AND COALESCE(source_uri,'')=%s",
                    (LAW_ABBR, SOURCE_URI))
        row = cur.fetchone()
        if row is None:
            conn.rollback()
            cur.close(); conn.close()
            raise RuntimeError("Could not obtain document_id.")
    doc_id = row[0]

    # Resume support: fetch existing section_numbers for this document
    cur.execute("SELECT section_number FROM legal.chunks WHERE document_id=%s", (doc_id,))
    existing = {sn for (sn,) in cur.fetchall()}
    print(f"[resume] found {len(existing)} already inserted rows")

    # Read NDJSON and prepare rows
    to_insert = []
    total = 0
    start = time.time()

    with IN_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sec = r["section_number"]
            if sec in existing:
                continue
            title = r.get("section_title", "")
            text = r["full_text"]
            emb  = vec_str(r["embedding"])
            to_insert.append((doc_id, sec, title, text, emb))
            total += 1

            # Flush in batches
            if len(to_insert) >= BATCH:
                _flush(cur, to_insert)
                conn.commit()
                print(f"[progress] inserted so far: {len(existing)} + {total}")
                to_insert.clear()

    # Final flush
    if to_insert:
        _flush(cur, to_insert)
        conn.commit()
        print(f"[progress] final total added this run: {total}")

    cur.close()
    conn.close()
    dur = time.time() - start
    print(f"[done] inserted {total} new rows in {dur:.1f}s")

def _flush(cur, rows):
    # Bulk insert with server-side cast to ::vector
    template = "(%s, %s, %s, %s, %s::vector)"
    execute_values(cur, """
        INSERT INTO legal.chunks
          (document_id, section_number, section_title, full_text, embedding)
        VALUES %s
        ON CONFLICT (document_id, section_number) DO NOTHING
    """, rows, template=template, page_size=len(rows))

if __name__ == "__main__":
    # limit=0 → load ALL; you can safely re-run to resume
    main(limit=0)
