-- Enable pgvector + base schema
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS legal;

-- documents table (no expression in table-level UNIQUE)
CREATE TABLE IF NOT EXISTS legal.documents (
  id          BIGSERIAL PRIMARY KEY,
  law_abbr    TEXT NOT NULL,            -- 'StGB', 'BGB', ...
  source_uri  TEXT,                     -- e.g., 'BJNR001270871.xml' (nullable)
  lang        TEXT DEFAULT 'de',
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- unique index on expression to treat NULL source_uri as ''
CREATE UNIQUE INDEX IF NOT EXISTS documents_law_source_uidx
  ON legal.documents (law_abbr, (COALESCE(source_uri, '')));

-- chunks table (one row per § with its embedding)
CREATE TABLE IF NOT EXISTS legal.chunks (
  id              BIGSERIAL PRIMARY KEY,
  document_id     BIGINT REFERENCES legal.documents(id) ON DELETE CASCADE,
  section_number  TEXT NOT NULL,
  section_title   TEXT,
  full_text       TEXT NOT NULL,
  embedding       VECTOR(1536) NOT NULL,
  created_at      TIMESTAMPTZ DEFAULT now(),
  UNIQUE (document_id, section_number)
);

-- vector index (cosine)
CREATE INDEX IF NOT EXISTS chunks_embedding_ivf
  ON legal.chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- helpful secondary indexes
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON legal.chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_sec_idx ON legal.chunks (section_number);

ANALYZE legal.chunks;
