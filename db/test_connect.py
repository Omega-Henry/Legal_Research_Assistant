# db/test_connect.py
import os, psycopg2
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path("/home/noe/Desktop/Ai_Legal research_assistant/.env"))
host = os.getenv("PGHOST")
port = os.getenv("PGPORT")
db   = os.getenv("PGDATABASE")
user = os.getenv("PGUSER")
pwd  = os.getenv("PGPASSWORD")
ssl  = os.getenv("PGSSLMODE", "require")

print("PGHOST =", host)
print("PGPORT =", port)
print("PGDATABASE =", db)
print("PGUSER =", user)
print("PGSSLMODE =", ssl)
print("PWD set? ", "yes" if pwd else "NO")

if not all([host, port, db, user, pwd]):
    raise RuntimeError("Missing one or more DB env vars. Check your .env is in project root and values are correct.")

# Try to connect
conn = psycopg2.connect(
    host=host, port=port, dbname=db, user=user, password=pwd, sslmode=ssl
)
with conn, conn.cursor() as cur:
    cur.execute("SELECT current_database(), version();")
    print("Connected:", cur.fetchone()[0])
    cur.execute("SELECT extname FROM pg_extension;")
    print("Extensions:", [r[0] for r in cur.fetchall()])
conn.close()