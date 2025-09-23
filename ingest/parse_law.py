# save as parse_law.py and run `python parse_law.py`
import xml.etree.ElementTree as ET
import json, re
from pathlib import Path

XML_IN  = Path("/home/noe/Desktop/Ai_Legal research_assistant/data/raw/BJNR001270871.xml")          # StGB XML
NDJSON_OUT = Path("/home/noe/Desktop/Ai_Legal research_assistant/data/interim/stgb_sections.ndjson")  # output

def raw_text(el):
    return "".join(el.itertext()) if el is not None else ""

def clean(s):
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()

tree = ET.parse(XML_IN.as_posix())
root = tree.getroot()

with NDJSON_OUT.open("w", encoding="utf-8") as f:
    for norm in root.findall(".//norm"):
        md = norm.find("./metadaten")
        if md is None: 
            continue
        enbez = (md.findtext("./enbez") or "").strip()
        if not enbez.startswith("§"):     # skip TOC etc.
            continue

        title = (md.findtext("./titel") or "").strip()
        sec_num = enbez.replace("§", "").strip()
        text_el = norm.find("./textdaten/text")
        body = clean(raw_text(text_el))

        # Compose a clean text (keep the § header to help semantics)
        full_text = clean(f"§ {sec_num} {title}\n\n{body}")

        rec = {
            "law_abbr":   (md.findtext("./jurabk") or "StGB").strip(),
            "section_number": sec_num,
            "section_title": title,
            "full_text":  full_text,
            "source_uri": XML_IN.name,
            "lang":       "de"
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"OK: wrote NDJSON → {NDJSON_OUT}")
