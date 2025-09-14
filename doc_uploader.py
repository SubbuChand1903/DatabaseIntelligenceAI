#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import json
import time
import logging
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from datetime import datetime, timezone
from hashlib import md5

import pandas as pd
from dotenv import load_dotenv

# --- Azure Search (upload) ---
from sentence_transformers import SentenceTransformer
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# =========================
# Tunables
# =========================
FUZZ_THRESHOLD = 0.86       # table/view fuzzy similarity threshold
MAX_CONTEXT = 180           # characters of line context to store
LOG_LEVEL = logging.INFO

# Only include GPS rows whose Schema (or New Schema) starts with these prefixes (case-insensitive)
ALLOW_SCHEMA_PREFIXES = ["partner", "pm", "sales"]

# Upload batches for Azure Search
UPLOAD_BATCH_SIZE = 100

# =========================
# Logging
# =========================
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("impact")

# =========================
# Utils: normalization & fuzzy
# =========================
def letters_digits_only(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())

def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, letters_digits_only(a), letters_digits_only(b)).ratio()

def strip_path_ext(s: str) -> str:
    s = (s or '').replace('\\', '/')
    s = s.split('/')[-1]
    s = re.sub(r'\.parquet$', '', s, flags=re.I)
    return s

def strip_schema(s: str) -> str:
    parts = (s or '').split('.')
    return parts[-1] if len(parts) > 1 else s

def norm_obj_name(s: str) -> str:
    s = (s or '').strip().lower()
    s = strip_path_ext(s)
    s = s.replace('"','').replace('`','').replace("'",'')
    s = s.replace('[','').replace(']','')
    for pre in ('dbo.', 'dw.', 'intake.', 'staging.'):
        s = s.replace(pre, '')
    return s

def base_name_variants(schema: str, view: str) -> set:
    """Generate robust variations so GPS view matches lineage tables."""
    full = f"{schema}.{view}".strip('.')
    v = norm_obj_name(full)
    out = {v}

    noschema = strip_schema(v)
    out.add(noschema)

    no_vw = re.sub(r'^vw_', '', noschema)
    out.update({no_vw, no_vw.replace('_', '')})

    core = no_vw.replace('_', '')
    for p in ['ocpmart_', 'gpsmart_', 'gpsadls_', 'ocpmart', 'gpsmart', 'gpsadls']:
        out.add(f"{p}{no_vw}")
        out.add(f"{p}{core}")

    for s in ['dw.', 'dbo.', 'intake.', 'staging.']:
        out.add(f"{s}{no_vw}")
        out.add(f"{s}{core}")

    return {w.strip('.') for w in out if w}

def fuzzy_match_any(target: str, candidates: set) -> bool:
    t = norm_obj_name(target)
    if t in candidates or strip_schema(t) in candidates:
        return True
    for c in candidates:
        if sim(t, c) >= FUZZ_THRESHOLD:
            return True
    return False

# =========================
# Utils: column search
# =========================
def build_col_regex(col: str) -> re.Pattern:
    esc = re.escape(col)
    # [col], "col", `col`, bare col, table.col
    pat = rf"(?<!\w)(?:\[\s*{esc}\s*\]|\"{esc}\"|`{esc}`|{esc}|(?:\w+\.){esc})(?!\w)"
    return re.compile(pat, flags=re.IGNORECASE)

def find_column_locations(text: str, col_names):
    """Search multiple column names; return unique line hits."""
    out = []
    if not text:
        return out
    regexes = [build_col_regex(c) for c in {c for c in col_names if c}]
    if not regexes:
        return out
    for i, line in enumerate(text.splitlines(), start=1):
        ln = line.strip()
        for rx in regexes:
            if rx.search(ln):
                out.append({"line_number": i, "context": ln[:MAX_CONTEXT]})
                break
    # dedupe by line number
    seen = set()
    dedup = []
    for p in out:
        if p["line_number"] not in seen:
            seen.add(p["line_number"])
            dedup.append(p)
    return dedup

# =========================
# Change-type parsing & consolidation
# =========================
def classify_change(change_type, old_dt, new_dt, old_col, new_col, new_view):
    """
    Returns: (canonical_what_changes, final_new_column_name, flags dict)
    flags: {'deprecated','rename','datatype','merged','new'}
    """
    ct = (change_type or '').strip()
    ctl = ct.lower()
    flags = {'deprecated': False, 'rename': False, 'datatype': False, 'merged': False, 'new': False}

    if 'deprec' in ctl:
        flags['deprecated'] = True
    if 'rename' in ctl:
        flags['rename'] = True
    if 'data type' in ctl or 'datatype' in ctl:
        flags['datatype'] = True
    if 'merged' in ctl:
        flags['merged'] = True
    if 'addition' in ctl or ('new column' in ctl) or (ctl == 'new column'):
        flags['new'] = True

    if flags['deprecated']:
        return "Deprecated", "(Deprecated)", flags

    parts = []
    if flags['rename'] or flags['merged']:
        if new_col:
            parts.append(f"Rename: {old_col} → {new_col}")
        elif 'merged' in ctl:
            m = re.search(r'merged\s+(?:in|with)\s+([A-Za-z0-9_.]+)', ct, flags=re.I)
            tgt = m.group(1) if m else (new_view or '').strip()
            parts.append(f"Merged: {old_col} → {tgt}" if tgt else "Merged")

    if flags['datatype'] and (old_dt or new_dt) and (old_dt != new_dt):
        parts.append(f"Data Type: {old_dt or 'unknown'} → {new_dt or 'unknown'}")

    if flags['new'] and not parts:
        parts.append("New Column")

    what = " + ".join(parts) if parts else (ct or "No Change")
    final_new_col = "(Deprecated)" if flags['deprecated'] else (new_col or old_col)
    return what, final_new_col, flags

def choose_canonical_change(rows_for_key):
    """
    rows_for_key: list of GPS rows (dicts)
    Returns a single consolidated dict for the key with canonical what_changes/new_column_name.
    Priority: Deprecated > (Rename/Merged) > DataType > New > No Change
    """
    PRIOR = {'deprecated': 5, 'rename': 4, 'merged': 4, 'datatype': 3, 'new': 2, 'none': 1}

    best_payload = None
    best_tuple = None
    best_score = -1

    for r in rows_for_key:
        what, final_new, flags = classify_change(
            r['Change Type'], r['Datatype'], r['New Datatype'],
            r['Column Name'], r['New Column Name'], r['New View Name']
        )
        if flags['deprecated']: score = PRIOR['deprecated']
        elif flags['rename'] or flags['merged']: score = PRIOR['rename']
        elif flags['datatype']: score = PRIOR['datatype']
        elif flags['new']: score = PRIOR['new']
        else: score = PRIOR['none']

        # prefer richer datatype when tied
        if score > best_score or (score == best_score and flags['datatype'] and r['New Datatype']):
            best_score = score
            best_tuple = (what, final_new)
            best_payload = r

    what_changes, new_col_name = best_tuple
    
    # BUILD NEW ENTITY NAME FROM NEW SCHEMA + NEW VIEW
    old_entity = f"{best_payload['Schema Name']}.{best_payload['View Name']}".strip('.')
    
    # Use New Schema Name and New View Name if available, otherwise fall back to original
    new_schema = best_payload.get('New Schema Name') or best_payload['Schema Name']
    new_view = best_payload.get('New View Name') or best_payload['View Name']
    new_entity = f"{new_schema}.{new_view}".strip('.')
    
    return {
        "old_entity_name": old_entity,
        "old_column_name": best_payload["Column Name"] or "",
        "new_column_name": new_col_name or "",
        "new_entity_name": new_entity,
        "what_changes": what_changes
    }

# =========================
# Loader + Builder
# =========================
class ImpactBuilder:
    def __init__(self, base_path):
        self.base = base_path
        self.gps_rows = []
        self.sql_lineage = []   # from 1st
        self.nb_lineage = []    # from 2nd
        self.sql_code = {}      # stem_lower -> {content, full_path}
        self.nb_code = {}       # stem_lower -> {content, full_path}

    @staticmethod
    def _schema_allowed(s: str) -> bool:
        s = (s or '').strip().lower().rstrip('.')  # tolerate trailing '.'
        return any(s.startswith(pref) for pref in ALLOW_SCHEMA_PREFIXES)

    def load_gps(self):
        p = os.path.join(self.base, "3rd_GPS_Schema_Changes_Input")
        xlsx = glob.glob(os.path.join(p, "*.xlsx"))
        if not xlsx:
            raise FileNotFoundError(f"No GPS Excel in {p}")
        f = xlsx[0]
        log.info(f"Loading GPS changes from {f}")
        df = pd.read_excel(f, na_filter=False)
        cols = ["StreamName","Schema Name","View Name","Column Name","Datatype","Change Type",
                "New Schema Name","New View Name","New Column Name","New Datatype"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"GPS Excel missing columns: {missing}")

        loaded, kept = 0, 0
        for idx, row in df.iterrows():
            loaded += 1
            r = {k: str(row.get(k, "")).strip() for k in cols}
            r["__row_index"] = int(idx)

            # schema filter (either current or new schema allowed)
            curr_ok = self._schema_allowed(r["Schema Name"])
            new_ok  = self._schema_allowed(r["New Schema Name"])
            if not (curr_ok or new_ok):
                continue
            kept += 1

            r["_schema_l"] = r["Schema Name"].lower()
            r["_view_l"] = r["View Name"].lower()
            r["_col_l"] = r["Column Name"].lower()
            r["_variants"] = base_name_variants(r["Schema Name"], r["View Name"])
            self.gps_rows.append(r)

        log.info(f"Loaded {loaded} GPS rows; kept {kept} after schema filter {ALLOW_SCHEMA_PREFIXES}")

    def load_lineage(self):
        # 1st — sprocs
        p1 = os.path.join(self.base, "1st_IRE_SQL_Synapse_Stored_Procs_Output")
        for csv in glob.glob(os.path.join(p1, "*.csv")):
            df = pd.read_csv(csv, na_filter=False, on_bad_lines='skip')
            for _, r in df.iterrows():
                self.sql_lineage.append({
                    "sproc": str(r.get("sproc name", "")).strip(),
                    "table": str(r.get("table name", "")).strip(),
                    "column": str(r.get("column name", "")).strip(),
                    "src": os.path.basename(csv)
                })
        log.info(f"Loaded {len(self.sql_lineage)} SQL lineage rows")

        # 2nd — notebooks
        p2 = os.path.join(self.base, "2nd_IRE_Databricks_Output")
        for csv in glob.glob(os.path.join(p2, "*.csv")):
            try:
                df = pd.read_csv(csv, header=0, na_filter=False, on_bad_lines='skip')
                cols = {c.lower(): c for c in df.columns}
                if not {"sno","notebook name","table name","column name"}.issubset({c.lower() for c in df.columns}):
                    raise ValueError("Missing headers, fallback to provided schema.")
            except Exception:
                df = pd.read_csv(csv, header=None,
                                 names=["sno","notebook name","table name","column name"],
                                 na_filter=False, on_bad_lines='skip')
            for _, r in df.iterrows():
                self.nb_lineage.append({
                    "notebook": str(r.get("notebook name", "")).strip(),
                    "table": str(r.get("table name", "")).strip(),
                    "column": str(r.get("column name", "")).strip(),
                    "src": os.path.basename(csv)
                })
        log.info(f"Loaded {len(self.nb_lineage)} notebook lineage rows")

    def load_code(self):
        # 4th — SQL code under metadata tree
        p4 = os.path.join(self.base, "4th_IRE_Metadata_Output")
        for sql in glob.glob(os.path.join(p4, "**", "*.sql"), recursive=True):
            try:
                with open(sql, "r", encoding="utf-8", errors="ignore") as f:
                    self.sql_code[Path(sql).stem.lower()] = {
                        "content": f.read(),
                        "full_path": os.path.relpath(sql, p4)
                    }
            except Exception as e:
                log.warning(f"Failed reading SQL file {sql}: {e}")
        # 5th — .py notebooks
        p5 = os.path.join(self.base, "5th_ADLS_Notebooks_Output")
        for py in glob.glob(os.path.join(p5, "**", "*.py"), recursive=True):
            try:
                with open(py, "r", encoding="utf-8", errors="ignore") as f:
                    self.nb_code[Path(py).stem.lower()] = {
                        "content": f.read(),
                        "full_path": os.path.relpath(py, p5)
                    }
            except Exception as e:
                log.warning(f"Failed reading notebook {py}: {e}")

        log.info(f"Loaded {len(self.sql_code)} SQL code files; {len(self.nb_code)} notebook .py files")

    def _lookup_sql_code(self, name):
        k = (name or "").strip().lower()
        if k in self.sql_code:
            return self.sql_code[k]["full_path"], self.sql_code[k]["content"]
        best, score = None, 0.0
        for key in self.sql_code.keys():
            r = sim(k, key)
            if r > score:
                best, score = key, r
        if best and score >= 0.8:
            return self.sql_code[best]["full_path"], self.sql_code[best]["content"]
        return "", ""

    def _lookup_nb_code(self, name):
        k = (name or "").strip().lower()
        if k in self.nb_code:
            return self.nb_code[k]["full_path"], self.nb_code[k]["content"]
        best, score = None, 0.0
        for key in self.nb_code.keys():
            r = sim(k, key)
            if r > score:
                best, score = key, r
        if best and score >= 0.8:
            return self.nb_code[best]["full_path"], self.nb_code[best]["content"]
        return "", ""

    def build_consolidated_impacts(self, out_json="impact_analysis_report.json"):
        if not self.gps_rows:
            log.warning("No GPS rows after schema filter; nothing to do.")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump([], f, indent=4, ensure_ascii=False)
            return []

        # Group GPS rows by (schema, view, column)
        groups = defaultdict(list)
        for r in self.gps_rows:
            groups[(r["_schema_l"], r["_view_l"], r["_col_l"])].append(r)

        log.info(f"Consolidating {len(groups)} GPS keys (schema-filtered to {ALLOW_SCHEMA_PREFIXES})")

        results = []

        for (_, _, _), rows in groups.items():
            header = choose_canonical_change(rows)
            old_entity = header["old_entity_name"]
            new_entity = header["new_entity_name"]
            old_col = header["old_column_name"]
            new_col = header["new_column_name"]

            # Use variants from the first row (same schema/view across group)
            variants = rows[0]["_variants"]

            # Collect lineage matches (sprocs/notebooks), dedup
            sproc_names = set()
            for row in self.sql_lineage:
                if not row["table"] or not row["column"]:
                    continue
                c = row["column"].strip().lower()
                if c not in {old_col.lower(), new_col.lower()}:
                    continue
                if fuzzy_match_any(row["table"], variants):
                    sproc_names.add(row["sproc"])

            nb_names = set()
            for row in self.nb_lineage:
                if not row["table"] or not row["column"]:
                    continue
                c = row["column"].strip().lower()
                if c not in {old_col.lower(), new_col.lower()}:
                    continue
                if fuzzy_match_any(row["table"], variants):
                    nb_names.add(row["notebook"])

            # Find code hits (search both old and new if they differ and not Deprecated)
            search_cols = {old_col}
            if new_col and new_col not in {"(Deprecated)", old_col}:
                search_cols.add(new_col)

            affected_sprocs = []
            for nm in sorted(sproc_names):
                path, text = self._lookup_sql_code(nm)
                places = find_column_locations(text, search_cols)
                affected_sprocs.append({
                    "name": nm,
                    "path": path,
                    "places_of_use": places
                })

            affected_nbs = []
            for nm in sorted(nb_names):
                path, text = self._lookup_nb_code(nm)
                places = find_column_locations(text, search_cols)
                affected_nbs.append({
                    "name": nm,
                    "path": path,
                    "places_of_use": places
                })

            results.append({
                "old_entity_name": old_entity,
                "old_column_name": old_col,
                "new_column_name": new_col,
                "new_entity_name": new_entity,
                "what_changes": header["what_changes"],
                "affected_stored_procedures": affected_sprocs,
                "affected_notebooks": affected_nbs
            })

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        log.info(f"Wrote {len(results)} consolidated rows to {out_json}")
        return results

# =========================
# Azure Search: doc mapping + upload
# =========================
def impact_to_doc(rec):
    """
    Map one consolidated impact record to your index schema.
    """
    old_entity = rec.get("old_entity_name", "")
    new_entity = rec.get("new_entity_name", "")
    
    if "." in old_entity:
        old_schema_name, old_table_name = old_entity.split(".", 1)
    else:
        old_schema_name, old_table_name = "", old_entity
        
    if "." in new_entity:
        new_schema_name, new_table_name = new_entity.split(".", 1)
    else:
        new_schema_name, new_table_name = "", new_entity

    old_col = rec.get("old_column_name", "")
    new_col = rec.get("new_column_name", "")
    what    = rec.get("what_changes", "")

    wt = what.lower()
    if "deprecated" in wt:
        change_type = "Deprecated"
    elif "rename" in wt and "data type" in wt:
        change_type = "Rename+Datatype"
    elif "rename" in wt:
        change_type = "Rename"
    elif "merged" in wt:
        change_type = "Merged"
    elif "data type" in wt:
        change_type = "DataType"
    elif "new column" in wt:
        change_type = "NewColumn"
    else:
        change_type = "NoChange"

    content = (
        f"[IMPACT] {old_entity} :: {old_col}"
        f"\nWhat: {what}"
        f"\nNew entity: {new_entity}"
        f"\nNew column: {new_col}"
        f"\nStored procedures: {len(rec.get('affected_stored_procedures', []))}"
        f"\nNotebooks: {len(rec.get('affected_notebooks', []))}"
    )

    metadata_blob = json.dumps({
        "what_changes": rec.get("what_changes", ""),
        "old_entity_name": rec.get("old_entity_name", ""),
        "new_entity_name": rec.get("new_entity_name", ""),
        "affected_stored_procedures": rec.get("affected_stored_procedures", []),
        "affected_notebooks": rec.get("affected_notebooks", []),
    }, ensure_ascii=False)

    doc_id = md5((old_entity + "|" + old_col + "|" + new_col + "|" + new_entity + "|" + what).encode("utf-8")).hexdigest()

    return {
        "id": doc_id,
        "content": content,
        "source_type": "gps_impact",
        "object_type": "column_change_impact",
        "object_name": "",
        "source_file": "impact_analysis_report.json",
        "schema_name": old_schema_name,
        "table_name": old_table_name,
        "columns": old_col if old_col else "",
        "change_type": change_type,
        "metadata": metadata_blob,
        'created_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    }

def upload_impacts_to_azure(index_name, endpoint, api_key, impacts, batch_size=UPLOAD_BATCH_SIZE):
    """
    Embed 'content' with all-MiniLM-L6-v2, attach to 'content_vector', and upload to Azure Search.
    """
    if not impacts:
        log.info("Nothing to upload.")
        return

    # init
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = SearchClient(endpoint=endpoint, index_name=index_name,
                          credential=AzureKeyCredential(api_key))

    docs = [impact_to_doc(r) for r in impacts]

    for i in range(0, len(docs), batch_size):
        chunk = docs[i:i+batch_size]
        try:
            embs = model.encode([d["content"] for d in chunk], show_progress_bar=False)
        except Exception as e:
            log.error(f"Embedding failed: {e}")
            raise

        for d, e in zip(chunk, embs):
            d["content_vector"] = e.tolist()

        # retry upload
        for attempt in range(3):
            try:
                _ = client.upload_documents(chunk)
                log.info(f"Uploaded {i+1}-{i+len(chunk)} / {len(docs)}")
                break
            except Exception as e:
                wait = 2 ** attempt
                log.warning(f"Upload attempt {attempt+1} failed: {e} (retry in {wait}s)")
                time.sleep(wait)
        else:
            log.error(f"Failed to upload batch {i//batch_size + 1} after retries.")

# =========================
# Entrypoint
# =========================
def main():
    load_dotenv()
    BASE_PATH = os.getenv("BASE_PATH") or r"C:\Users\SubbuChand\Desktop\NGPO\metadata (1)\metadata"

    if not os.path.isdir(BASE_PATH):
        print(f"❌ base path not found: {BASE_PATH}")
        return

    builder = ImpactBuilder(BASE_PATH)
    builder.load_gps()       # applies schema filter: partner.*, pm.*, sales.*
    builder.load_lineage()
    builder.load_code()
    impacts = builder.build_consolidated_impacts(out_json="impact_analysis_report.json")

    print("\n✅ Consolidation done. impact_analysis_report.json ready.")

    # --- Upload toggle ---
    UPLOAD = True  # set to False if you only want the JSON

    if UPLOAD:
        AZURE_SEARCH_ENDPOINT = "https://.search.windows.net"#os.getenv("AZURE_SEARCH_ENDPOINT")
        AZURE_SEARCH_API_KEY  = ""#os.getenv("AZURE_SEARCH_API_KEY")
        INDEX_NAME            = "dw-impact-analysis-index-new"#os.getenv("AZURE_SEARCH_INDEX_NAME") or "dw-impact-analysis-index"

        if not (AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY):
            print("⚠️  Missing AZURE_SEARCH_ENDPOINT/AZURE_SEARCH_API_KEY; skipping upload.")
            return

        upload_impacts_to_azure(
            index_name=INDEX_NAME,
            endpoint=AZURE_SEARCH_ENDPOINT,
            api_key=AZURE_SEARCH_API_KEY,
            impacts=impacts,
            batch_size=UPLOAD_BATCH_SIZE
        )

        print("\n🚀 Upload complete.")

if __name__ == "__main__":
    main()