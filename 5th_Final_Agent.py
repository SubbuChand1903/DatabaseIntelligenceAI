import os
import re
import json
import time
import random
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.exceptions import HttpResponseError

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# -----------------------------
# 0) Configuration / Clients
# -----------------------------
load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = "dw-impact-analysis-index-new"

# UI defaults
DEFAULT_TOP_N = 20

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_search_client():
    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )

# -----------------------------
# 1) LangGraph Agent State & Schema
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    identifiers: List[str]
    raw_docs: List[Dict[str, Any]]
    processed_impacts: List[Dict[str, Any]]
    filtered_impacts: List[Dict[str, Any]]
    final_results: List[Dict[str, Any]]
    summary: Dict[str, int]
    strict_filter: bool
    top_n: int
    step_log: List[str]

# -----------------------------
# 2) Core Utilities (Enhanced with better logging)
# -----------------------------
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]*")

STOP_WORDS = {
    "what","which","tables","columns","column","table","impact","changes","seems","tell",
    "downstream","changing","change","changed","rename","renamed","deprecate","deprecated",
    "drop","dropped","remove","removed","add","added","update","updated","type","datatype",
    "the","of","in","is","are","was","were","and","or","but","for","with","from","to","at","on"
}

def _is_stop(t: str) -> bool:
    return (t or "").lower() in STOP_WORDS

_USER_NOTEBOOK_RE = re.compile(r"(?i)(?:^|[\\/]|dbfs:/)users[\\/]")
def _is_user_notebook(path: str) -> bool:
    return bool(path and _USER_NOTEBOOK_RE.search(str(path)))

_BACKUP_OLD_RE = re.compile(r"(?i)\b(?:backup|bkp|bak|old|copy|temp)\b")
_ALIAS_DATE_RE = re.compile(r"(?i)_alias_(?:\d{8}|\d{4}[-_]\d{2}[-_]\d{2})")
_TRAILING_DATE_RE = re.compile(r"(?i)(?:^|[_-])(?:\d{8}|\d{4}[-_]\d{2}[-_]\d{2})(?:$|[._])")
def _is_backup_or_alias_or_date(name: str, path: str) -> bool:
    n = str(name or "")
    p = str(path or "")
    return (
        _BACKUP_OLD_RE.search(n) or _BACKUP_OLD_RE.search(p) or
        _ALIAS_DATE_RE.search(n) or _ALIAS_DATE_RE.search(p) or
        _TRAILING_DATE_RE.search(n) or _TRAILING_DATE_RE.search(p)
    )

def _safe_json_loads(maybe_json):
    if maybe_json is None:
        return {}
    if isinstance(maybe_json, dict):
        return maybe_json
    try:
        return json.loads(maybe_json)
    except Exception:
        return {}

# -----------------------------
# 3) Enhanced LangGraph Agent Nodes with Terminal Logging
# -----------------------------

def identifier_extraction_agent(state: AgentState) -> AgentState:
    """Agent: Extract identifiers from query text - SIMPLIFIED"""
    query = state["query"]
    print(f"\n🔍 IDENTIFIER EXTRACTION")
    print(f"Original Query: '{query}'")
    
    state["step_log"].append(f"🔍 Extracting identifiers from: {query[:50]}...")
    
    hits = IDENTIFIER_RE.findall(query or "")
    print(f"Raw regex matches: {hits}")
    
    cleaned = []
    seen = set()

    for t in hits:
        if len(t) < 3 or _is_stop(t) or t in seen:
            continue
        cleaned.append(t)
        seen.add(t)

    print(f"Cleaned identifiers: {cleaned}")

    # Simple enhancement for common variations
    enhanced_tokens = []
    for token in cleaned:
        enhanced_tokens.append(token)
        
        # Handle ID variations
        if token.lower().endswith('id'):
            enhanced_tokens.append(token.replace('ID', '_ID'))
            enhanced_tokens.append(token.replace('Id', '_Id'))
        
        # Handle org variations
        if 'org' in token.lower():
            enhanced_tokens.append(token.replace('Org', 'Organization'))
            enhanced_tokens.append(token.replace('org', 'organization'))

    # Add semantic keywords based on query content
    query_lower = query.lower()
    if any(word in query_lower for word in ['merge', 'consolidate', 'combine']):
        enhanced_tokens.extend(['merge', 'merged'])
    if any(word in query_lower for word in ['change', 'impact', 'consequence']):
        enhanced_tokens.extend(['change', 'changed'])

    unique = []
    for t in enhanced_tokens:
        if t not in unique and not _is_stop(t):
            unique.append(t)
    
    print(f"Final identifiers: {unique}")
    
    state["identifiers"] = unique
    state["step_log"].append(f"📝 Found identifiers: {unique}")
    state["messages"].append(AIMessage(content=f"Extracted {len(unique)} identifiers: {', '.join(unique[:5])}" + ("..." if len(unique) > 5 else "")))
    
    return state

def search_agent(state: AgentState) -> AgentState:
    """Agent: Perform hybrid search on Azure index with detailed logging"""
    query = state["query"]
    identifiers = state["identifiers"]
    
    print(f"\n🔎 HYBRID SEARCH")
    print(f"Search Query: '{query}'")
    print(f"Using identifiers: {identifiers}")
    
    state["step_log"].append(f"🔎 Performing hybrid search...")
    
    embedder = get_embedding_model()
    search_client = get_search_client()
    
    # Enhance search query with identifiers
    search_terms = [query]
    for identifier in identifiers:
        if len(identifier) > 2 and not _is_stop(identifier):
            search_terms.append(identifier)
    
    qt = " ".join(search_terms).strip() or "schema change impact"
    print(f"Enhanced search text: '{qt}'")
    
    vec = embedder.encode(qt).tolist()
    vq = VectorizedQuery(vector=vec, k_nearest_neighbors=80, fields="content_vector")

    selectable = [
        "id", "content", "source_type", "source_file", "object_name", 
        "table_name", "schema_name", "columns", "change_type", "object_type", 
        "metadata", "created_at"
    ]
    searchable_fields = ["content", "columns", "table_name", "object_name"]

    try:
        results = search_client.search(
            search_text=qt,
            vector_queries=[vq],
            top=80,
            select=selectable,
            search_fields=searchable_fields,
        )
        docs = list(results)
    except HttpResponseError as e:
        if "search field list is not searchable" in str(e):
            results = search_client.search(
                search_text=qt,
                vector_queries=[vq],
                top=80,
                select=selectable
            )
            docs = list(results)
        else:
            raise

    print(f"Retrieved {len(docs)} documents from Azure Search")
    
    # Log first few results for debugging
    for i, doc in enumerate(docs[:3]):
        print(f"Doc {i+1}: {doc.get('table_name', 'N/A')} - {doc.get('columns', 'N/A')} - {doc.get('change_type', 'N/A')}")
        content_preview = (doc.get('content', '') or '')[:100]
        print(f"  Content: {content_preview}...")

    state["raw_docs"] = docs
    state["step_log"].append(f"📊 Retrieved {len(docs)} documents from search")
    state["messages"].append(AIMessage(content=f"Found {len(docs)} relevant documents from search index"))
    
    return state

def impact_processing_agent(state: AgentState) -> AgentState:
    """Agent: Process raw docs into normalized impact records with enhanced logging"""
    docs = state["raw_docs"]
    print(f"\n⚙️ IMPACT PROCESSING")
    print(f"Processing {len(docs)} documents into impact records...")
    
    state["step_log"].append(f"⚙️ Processing {len(docs)} documents into impact records...")
    
    def extract_user_notebook_path(path_or_name: str) -> str:
        if not path_or_name:
            return path_or_name
        path = str(path_or_name)
        if 'users' in path.lower():
            if path.startswith('/'):
                return path
            elif '\\' in path:
                return path.replace('\\', '/')
        return path

    def extract_sql_proc_name(name_or_path: str) -> str:
        return str(name_or_path) if name_or_path else name_or_path

    def build_impact_from_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
        schema = (doc.get("schema_name") or "").strip()
        table = (doc.get("table_name") or "").strip()
        entity = f"{schema}.{table}" if schema and table else table or schema or ""
        column = (doc.get("columns") or "").strip()
        change_type = (doc.get("change_type") or "").strip()

        meta = _safe_json_loads(doc.get("metadata"))
        what_changes = meta.get("what_changes") or doc.get("content", "").splitlines()[1:2]
        if isinstance(what_changes, list):
            what_changes = " ".join(what_changes).strip()

        if not what_changes:
            m = re.search(r"(?i)what\s*:\s*(.+)", doc.get("content", ""))
            what_changes = m.group(1).strip() if m else (change_type or "Schema change")

        sprocs = meta.get("affected_stored_procedures") or []
        notebooks = meta.get("affected_notebooks") or []

        def norm_places(lst):
            out = []
            for p in lst or []:
                places = []
                for z in p.get("places_of_use") or []:
                    places.append({
                        "line_number": z.get("line_number"),
                        "context": z.get("context", "")
                    })

                name = extract_user_notebook_path(p.get("name", ""))
                path = extract_user_notebook_path(p.get("path", ""))
                if 'stored_procedures' in str(lst):
                    name = extract_sql_proc_name(p.get("name", ""))
                    path = extract_sql_proc_name(p.get("path", ""))

                out.append({
                    "name": name,
                    "path": path,
                    "places_of_use": places
                })
            return out

        sprocs = norm_places(sprocs)
        notebooks = norm_places(notebooks)

        # Apply visibility rules
        sprocs = [p for p in sprocs if not _is_backup_or_alias_or_date(p.get("name",""), p.get("path",""))]
        notebooks = [p for p in notebooks if not _is_user_notebook(p.get("path",""))]

        impact_record = {
            "doc_id": doc.get("id"),
            "entity_name": entity,
            "table_name": table,
            "schema_name": schema,
            "column_name": column,
            "what_changed": what_changes,
            "sproc_count": len(sprocs),
            "notebook_count": len(notebooks),
            "sprocs": sprocs,
            "notebooks": notebooks,
            "raw_source_type": doc.get("source_type", ""),
            "raw_object_type": doc.get("object_type", ""),
            "content": doc.get("content", ""),
        }
        
        return impact_record

    impacts = [build_impact_from_doc(d) for d in docs]
    
    print(f"Created {len(impacts)} impact records")
    for i, impact in enumerate(impacts[:5]):  # Log first 5 for debugging
        print(f"Impact {i+1}: {impact['entity_name']} - {impact['column_name']} - {impact['what_changed']}")
    
    state["processed_impacts"] = impacts
    state["step_log"].append(f"🔄 Processed into {len(impacts)} impact records")
    state["messages"].append(AIMessage(content=f"Processed {len(impacts)} impact records from documents"))
    
    return state

def filtering_agent(state: AgentState) -> AgentState:
    """Agent: Apply semantic filtering with enhanced paraphrasing support - FIXED"""
    impacts = state["processed_impacts"]
    identifiers = state["identifiers"]
    strict_filter = state["strict_filter"]
    query = state["query"]
    
    print(f"\n🎯 FILTERING")
    print(f"Input impacts: {len(impacts)}")
    print(f"Identifiers: {identifiers}")
    print(f"Strict filter: {strict_filter}")
    print(f"Original query: '{query}'")
    
    if not strict_filter or not identifiers:
        state["filtered_impacts"] = impacts
        state["step_log"].append(f"⚡ No filtering applied - returning all {len(impacts)} impacts")
        state["messages"].append(AIMessage(content=f"Skipped filtering - returning all {len(impacts)} impacts"))
        print(f"No filtering applied - returning all {len(impacts)} impacts")
        return state
    
    state["step_log"].append(f"🎯 Applying semantic filtering with tokens: {identifiers}")
    
    # Enhanced filtering with semantic understanding
    canon = sorted({t.upper() for t in identifiers if t and len(t) >= 3 and not _is_stop(t)}, key=len, reverse=True)
    print(f"Canonical tokens: {canon}")
    
    if not canon:
        state["filtered_impacts"] = impacts
        state["step_log"].append(f"⚡ No valid canonical tokens - returning unfiltered")
        print("No valid canonical tokens - returning unfiltered")
        return state

    # Enhanced pattern detection for various phrasings
    query_lower = query.lower()
    
    # Detect merge-related queries with multiple phrasings - ENHANCED FOR "WITH" PATTERNS
    merge_patterns = [
        r"merged?\s+(?:in|into|to)\s+(\w+)",
        r"merged?\s+with\s+(\w+)",
        r"consolidat\w*\s+(?:in|into|to|with)\s+(\w+)",
        r"combin\w*\s+(?:in|into|to|with)\s+(\w+)",
        r"mov\w*\s+(?:in|into|to|with)\s+(\w+)",
        r"transfer\w*\s+(?:in|into|to|with)\s+(\w+)",
        r"migrat\w*\s+(?:in|into|to|with)\s+(\w+)",
    ]
    
    # Also look for change-related queries about specific columns
    change_patterns = [
        r"(?:what|how).*changed?\s+(?:in|with|to)\s+(\w+)",     # "what changed with X"
        r"(?:impact|effect|consequence).*(?:of|from)\s+(\w+)",  # "impact of X"
        r"(?:affect|change).*(?:in|to|with)\s+(\w+)",           # "changes in/with X"
        r"downstream.*(?:of|from|with)\s+(\w+)",                # "downstream of/with X"
    ]
    
    # FIXED: Properly initialize variables
    target_table = None
    merge_intent = False
    
    # Try to extract target from merge patterns
    for pattern in merge_patterns:
        match = re.search(pattern, query_lower)
        if match:
            target_table = match.group(1).upper()
            merge_intent = True
            print(f"Detected merge intent for target: {target_table}")
            break
    
    # If no merge intent, try change patterns
    if not merge_intent:
        for pattern in change_patterns:
            match = re.search(pattern, query_lower)
            if match:
                target_table = match.group(1).upper()
                print(f"Detected change reference for: {target_table}")
                break

    matched = []
    for impact in impacts:
        col_uc = (impact.get("column_name","") or "").upper()
        ent_uc = (impact.get("entity_name","") or "").upper()
        what_uc = (impact.get("what_changed","") or "").upper()
        content_uc = (impact.get("content","") or "").upper()
        table_uc = (impact.get("table_name","") or "").upper()
        
        match_found = False
        match_reasons = []
        
        # Priority 1: Semantic merge matching with target - ENHANCED FOR EXACT MATCHING
        if merge_intent and target_table:
            merge_keywords = ["MERGED", "CONSOLIDAT", "COMBIN", "INTEGRAT", "TRANSFER", "MIGRAT"]
            has_merge_keyword = any(keyword in what_uc or keyword in content_uc for keyword in merge_keywords)
            
            # Enhanced: Look for exact column name in what_changed for precise matching
            has_exact_column_match = target_table in what_uc and (":" in what_uc or "→" in what_uc or "->" in what_uc)
            
            if has_merge_keyword and has_exact_column_match:
                matched.append(impact)
                match_reasons.append(f"Exact merge operation with {target_table}")
                match_found = True
                print(f"✓ Matched on exact merge operation with {target_table}: {impact['entity_name']} - {impact['column_name']} - {impact['what_changed']}")
                continue
        
        # Priority 2: Direct target reference matching
        if target_table and not merge_intent:
            if target_table in ent_uc or target_table in table_uc or target_table in what_uc or target_table in content_uc:
                matched.append(impact)
                match_reasons.append(f"Target reference match: {target_table}")
                match_found = True
                print(f"✓ Matched on target reference '{target_table}': {impact['entity_name']} - {impact['column_name']}")
                continue
        
        # Priority 3: Exact column name match
        for t in canon:
            if col_uc == t:
                matched.append(impact)
                match_reasons.append(f"Exact column match: {t}")
                match_found = True
                print(f"✓ Matched on exact column '{t}': {impact['entity_name']} - {impact['column_name']}")
                break
        
        if match_found:
            continue
            
        # Priority 4: Enhanced semantic pattern matching - RESTRICTED when merge intent exists
        if not match_found:
            semantic_patterns = []
            for t in canon:
                # Handle different identifier patterns
                if re.fullmatch(r"[A-Z0-9_]+", t):
                    semantic_patterns.append(r"\b" + re.escape(t) + r"\b")
                else:
                    semantic_patterns.append(r"(?<![A-Za-z0-9_])" + re.escape(t) + r"(?![A-Za-z0-9_])")
                
                # Add semantic variations for common terms - ONLY if no specific merge intent
                if not merge_intent:
                    if t == "MERGE" or "MERGE" in t:
                        semantic_patterns.extend([
                            r"\bCONSOLIDAT\w*\b", r"\bCOMBIN\w*\b", r"\bINTEGRAT\w*\b"
                        ])
                    elif t == "IMPACT" or "IMPACT" in t:
                        semantic_patterns.extend([
                            r"\bEFFECT\w*\b", r"\bCONSEQUENCE\w*\b", r"\bAFFECT\w*\b"
                        ])
            
            if semantic_patterns:
                pat = re.compile("|".join(semantic_patterns))
                # When merge intent exists, only match if target is also present
                if merge_intent and target_table:
                    if pat.search(what_uc) or pat.search(content_uc) or pat.search(ent_uc):
                        if target_table in what_uc or target_table in content_uc or target_table in col_uc:
                            matched.append(impact)
                            match_reasons.append("Semantic pattern match with target")
                            print(f"✓ Matched on semantic pattern with target: {impact['entity_name']} - {impact['column_name']} - {impact['what_changed']}")
                else:
                    if pat.search(what_uc) or pat.search(content_uc) or pat.search(ent_uc):
                        matched.append(impact)
                        match_reasons.append("Semantic pattern match")
                        print(f"✓ Matched on semantic pattern: {impact['entity_name']} - {impact['column_name']} - {impact['what_changed']}")

    filtered_results = matched if matched else impacts
    
    print(f"Filtering result: {len(filtered_results)} impacts matched out of {len(impacts)}")
    if matched:
        print("Top matches:")
        for i, match in enumerate(matched[:3]):
            print(f"  {i+1}. {match['entity_name']} - {match['column_name']} - {match['what_changed']}")
    
    state["filtered_impacts"] = filtered_results
    state["step_log"].append(f"🎯 Filtered to {len(filtered_results)} matching impacts")
    state["messages"].append(AIMessage(content=f"Applied semantic filtering: {len(filtered_results)} impacts match your criteria"))
    
    return state

def ranking_agent(state: AgentState) -> AgentState:
    """Agent: Rank and deduplicate impacts by risk with enhanced logging"""
    impacts = state["filtered_impacts"]
    top_n = state["top_n"]
    
    print(f"\n🏆 RANKING")
    print(f"Input impacts for ranking: {len(impacts)}")
    
    state["step_log"].append(f"🏆 Ranking and deduplicating {len(impacts)} impacts...")
    
    # De-dup by (entity, column, what_changed)
    dedup = {}
    for r in impacts:
        k = (r["entity_name"], r["column_name"], r["what_changed"])
        if k not in dedup:
            dedup[k] = r
        else:
            if (r["sproc_count"] + r["notebook_count"]) > (dedup[k]["sproc_count"] + dedup[k]["notebook_count"]):
                dedup[k] = r
    
    impacts = list(dedup.values())
    print(f"After deduplication: {len(impacts)} impacts")
    
    # Risk-based sorting
    def risk_key(r):
        risk = 0
        what_changed = r["what_changed"].lower()
        
        if re.search(r"(?i)merge", what_changed):
            risk += 4  # High priority for merge operations
        if re.search(r"(?i)deprecat", what_changed):
            risk += 3
        if re.search(r"(?i)rename", what_changed):
            risk += 2
        if re.search(r"(?i)data\s*type", what_changed):
            risk += 1
        
        risk += min(5, r["sproc_count"] + r["notebook_count"])
        return -risk

    impacts.sort(key=risk_key)
    top_impacts = impacts[:top_n]
    
    print(f"Final ranking - top {len(top_impacts)} impacts:")
    for i, impact in enumerate(top_impacts):
        risk_score = -risk_key(impact)
        print(f"  {i+1}. {impact['entity_name']} - {impact['column_name']} - {impact['what_changed']} (Risk: {risk_score})")
    
    state["final_results"] = top_impacts
    state["step_log"].append(f"🏆 Final ranking: {len(top_impacts)} top impacts selected")
    state["messages"].append(AIMessage(content=f"Ranked and selected top {len(top_impacts)} highest-risk impacts"))
    
    return state

def summarization_agent(state: AgentState) -> AgentState:
    """Agent: Generate summary statistics with enhanced logging"""
    impacts = state["final_results"]
    
    print(f"\n📈 SUMMARIZATION")
    print(f"Generating summary for {len(impacts)} final impacts")
    
    state["step_log"].append(f"📈 Generating summary for {len(impacts)} impacts...")
    
    total = len(impacts)
    sproc_total = sum(r["sproc_count"] for r in impacts)
    nb_total = sum(r["notebook_count"] for r in impacts)
    
    high_risk = sum(
        1 for r in impacts
        if re.search(r"(?i)deprecat|rename|data\s*type|merge", r["what_changed"])
        or (r["sproc_count"] + r["notebook_count"] > 0)
    )
    
    summary = {
        "total": total,
        "sproc_total": sproc_total,
        "nb_total": nb_total,
        "high_risk": high_risk,
    }
    
    print(f"Summary - Total: {total}, High Risk: {high_risk}, Sprocs: {sproc_total}, Notebooks: {nb_total}")
    
    state["summary"] = summary
    state["step_log"].append(f"📈 Summary: {total} total, {high_risk} high-risk, {sproc_total} sprocs, {nb_total} notebooks")
    state["messages"].append(AIMessage(content=f"Analysis complete: {total} impacts found, {high_risk} are high-risk"))
    
    return state

# -----------------------------
# 4) LangGraph Workflow Definition
# -----------------------------

def create_impact_analysis_agent():
    """Create the LangGraph-based impact analysis workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("extract_identifiers", identifier_extraction_agent)
    workflow.add_node("search_index", search_agent)
    workflow.add_node("process_impacts", impact_processing_agent)
    workflow.add_node("filter_impacts", filtering_agent)
    workflow.add_node("rank_impacts", ranking_agent)
    workflow.add_node("summarize", summarization_agent)
    
    # Define workflow edges
    workflow.set_entry_point("extract_identifiers")
    workflow.add_edge("extract_identifiers", "search_index")
    workflow.add_edge("search_index", "process_impacts")
    workflow.add_edge("process_impacts", "filter_impacts")
    workflow.add_edge("filter_impacts", "rank_impacts")
    workflow.add_edge("rank_impacts", "summarize")
    workflow.add_edge("summarize", END)
    
    return workflow.compile()

@st.cache_resource
def get_impact_agent():
    """Cached agent instance"""
    return create_impact_analysis_agent()

# -----------------------------
# 5) Quick Search Utility
# -----------------------------
def quick_hybrid_search(search_client: SearchClient, embedder: SentenceTransformer, query_text: str, top_fetch: int = 15) -> List[Dict[str, Any]]:
    """Quick search for global index peek"""
    qt = (query_text or "").strip() or "schema change impact"
    vec = embedder.encode(qt).tolist()
    vq = VectorizedQuery(vector=vec, k_nearest_neighbors=top_fetch, fields="content_vector")

    selectable = [
        "id", "content", "source_type", "source_file", "object_name", 
        "table_name", "schema_name", "columns", "change_type", "object_type", 
        "metadata", "created_at"
    ]
    searchable_fields = ["content", "columns", "table_name", "object_name"]

    try:
        results = search_client.search(
            search_text=qt,
            vector_queries=[vq],
            top=top_fetch,
            select=selectable,
            search_fields=searchable_fields,
        )
        return list(results)
    except HttpResponseError as e:
        if "search field list is not searchable" in str(e):
            results = search_client.search(
                search_text=qt,
                vector_queries=[vq],
                top=top_fetch,
                select=selectable
            )
            return list(results)
        raise

# -----------------------------
# 6) Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="NGPO GPS Migration Agent(GPT 5 Powered)",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* Hero */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #fff;
    border-radius: 16px;
    padding: 26px 32px;
    margin: 18px 0 8px;
    box-shadow: 0 10px 30px rgba(0,0,0,.12);
}
.hero h1 {
    margin: 0 0 4px;
    font-weight: 800;
    letter-spacing:.3px;
}
.hero p {
    opacity:.95;
    margin: 0;
}

/* Cards row */
.kb-row {
    display: grid;
    grid-template-columns: repeat(5,1fr);
    gap: 18px;
    margin: 10px 0 26px;
}
.kb-card {
    border-radius: 14px;
    padding: 18px;
    color:#fff;
    text-align:center;
    box-shadow: 0 8px 24px rgba(0,0,0,.12);
}
.card-1 { background: linear-gradient(135deg, #667eea, #764ba2); }
.card-2 { background: linear-gradient(135deg, #f093fb, #f5576c); }
.card-3 { background: linear-gradient(135deg, #4facfe, #00f2fe); }
.card-4 { background: linear-gradient(135deg, #43e97b, #38f9d7); }
.card-5 { background: linear-gradient(135deg, #fa709a, #fee140); }

/* Stat cards */
.stats {
    display:grid;
    grid-template-columns: repeat(4,1fr);
    gap:18px;
    margin: 16px 0 8px;
}
.stat {
    border-radius:14px;
    padding:18px;
    color:#fff;
    text-align:center;
    box-shadow:0 8px 24px rgba(0,0,0,.12);
}
.stat h2 {
    margin:.2rem 0 .2rem;
    font-size: 2.0rem;
}
.s1 { background: linear-gradient(135deg, #43e97b, #38f9d7); }
.s2 { background: linear-gradient(135deg, #f093fb, #f5576c); }
.s3 { background: linear-gradient(135deg, #4facfe, #00f2fe); }
.s4 { background: linear-gradient(135deg, #fa709a, #fee140); }

/* Touchpoints */
.tp {
    background:#f8f9fa;
    border-left: 4px solid #667eea;
    padding:8px 10px;
    border-radius:8px;
    margin:6px 0;
}
.tp .ln {
    background:#667eea;
    color:#fff;
    padding:2px 6px;
    border-radius:4px;
    margin-right:8px;
    font-size:.8rem;
}

/* Minor tweaks */
.block {
    background:#fff;
    border-radius:14px;
    padding:16px;
    box-shadow:0 5px 18px rgba(0,0,0,.06);
}
hr {
    border:none;
    border-top:1px solid #eee;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <h1>🚀 NGPO GPS Migration Agent(GPT 5 Powered)</h1>
    <p>GPT-5 AI-Powered Impact Analysis for GPS Migrations</p>
</div>
""", unsafe_allow_html=True)

# Knowledge base cards
st.markdown("### 🔍 AI-Powered Knowledge Base")
st.caption("Comprehensive analysis across your entire data ecosystem")

st.markdown("""
<div class="kb-row">
    <div class="kb-card card-1">
        <div style="font-size:2rem;">📜</div>
        <div style="font-weight:700;">DW Stored Procedures</div>
        <div>Lineage from sprocs → tables/columns</div>
    </div>
    <div class="kb-card card-2">
        <div style="font-size:2rem;">🔗</div>
        <div style="font-weight:700;">Databricks Lineage</div>
        <div>Notebook → sources with transforms</div>
    </div>
    <div class="kb-card card-3">
        <div style="font-size:2rem;">🔄</div>
        <div style="font-weight:700;">GPS Schema Changes</div>
        <div>Renames, deprecations, merges</div>
    </div>
    <div class="kb-card card-4">
        <div style="font-size:2rem;">🏛️</div>
        <div style="font-weight:700;">Full DW Metadata</div>
        <div>Source definitions & schemas</div>
    </div>
    <div class="kb-card card-5">
        <div style="font-size:2rem;">💻</div>
        <div style="font-weight:700;">Notebook Code Base</div>
        <div>Full Databricks code analysis</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Query controls
embedder = get_embedding_model()
search_client = get_search_client()
agent = get_impact_agent()

query = st.text_input(
    "💬 Ask Your Impact Analysis Question",
    value="What are the downstream consequences of the MPNVOrgID column changing?",
    help="Ask about schema changes, a table/view, or a specific column. Now with enhanced paraphrase support!"
)

c1, c2 = st.columns([3,1])
with c1:
    strict = st.checkbox(
        "Filter strictly by column name found in the question", 
        value=True,
        help="Extract tokens from your question (e.g., MPNVOrgID) and only keep impacts that mention them. Enhanced with fuzzy matching!"
    )
with c2:
    top_n = st.number_input("Top N", min_value=5, max_value=50, value=DEFAULT_TOP_N, step=1)

run = st.button("🔎 Analyze Impact", type="primary", use_container_width=False)

# Global search
#st.markdown("### 🧭 Global Search (Index Peek)")
quick = st.text_input("Search across index quickly (name, column, snippet)...", value="", label_visibility="collapsed")

if quick:
    try:
        docs = quick_hybrid_search(search_client, embedder, quick, top_fetch=15)
        if docs:
            peek = []
            for d in docs:
                peek.append({
                    "Type": d.get("source_type",""),
                    "Schema": d.get("schema_name",""),
                    "Table": d.get("table_name",""),
                    "Column": d.get("columns",""),
                    "Change": d.get("change_type",""),
                    "Preview": (d.get("content","")[:120] + "…") if len(d.get("content",""))>120 else d.get("content","")
                })
            st.dataframe(pd.DataFrame(peek), use_container_width=True, hide_index=True)
        else:
            st.info("No results.")
    except Exception as e:
        st.error(f"Global search error: {e}")

st.markdown("---")

# Main Analysis - LangGraph Agent Integration
if run:
    print("\n======== LANGGRAPH AGENT ANALYSIS ========")
    print(f"Query: {query}")
    print(f"Strict Filter: {strict}")
    print(f"Top N: {top_n}")
    
    # Initialize agent state
    initial_state = AgentState(
        messages=[HumanMessage(content=query)],
        query=query,
        identifiers=[],
        raw_docs=[],
        processed_impacts=[],
        filtered_impacts=[],
        final_results=[],
        summary={},
        strict_filter=strict,
        top_n=top_n,
        step_log=[]
    )
    
    with st.spinner("🤖 AI Agent analyzing impact across your data ecosystem..."):
        # Run the agent workflow
        final_state = agent.invoke(initial_state)
        
        # Display agent progress
        with st.expander("🤖 Agent Execution Log", expanded=False):
            for step in final_state["step_log"]:
                st.text(step)
    
    # Extract results from final state
    top_impacts = final_state["final_results"]
    summary = final_state["summary"]

    # Summary cards
    st.markdown("### 📈 Impact Summary")
    st.markdown(f"""
    <div class="stats">
        <div class="stat s1"><div>Total Changes</div><h2>{summary['total']}</h2></div>
        <div class="stat s2"><div>Stored Procedures Impacted</div><h2>{summary['sproc_total']}</h2></div>
        <div class="stat s3"><div>Notebooks Impacted</div><h2>{summary['nb_total']}</h2></div>
        <div class="stat s4"><div>High Risk Changes</div><h2>{summary['high_risk']}</h2></div>
    </div>
    """, unsafe_allow_html=True)

    # Detailed table
    st.markdown("### 📊 Detailed Impact Analysis")
    if not top_impacts:
        st.info("No matching impact rows found for your query/filters.")
    else:
        # Add search functionality for the table
        col1, col2, col3 = st.columns(3)
        with col1:
            table_filter = st.text_input("Filter by Table", value="", key="table_search")
        with col2:
            column_filter = st.text_input("Filter by Column", value="", key="column_search")
        with col3:
            change_filter = st.text_input("Filter by What Changed", value="", key="change_search")
        
        rows = []
        for i, r in enumerate(top_impacts, start=1):
            table_name = r["entity_name"] or r["table_name"]
            column_name = r["column_name"] or "(n/a)"
            what_changed = r["what_changed"]
            
            # Apply filters
            if table_filter and table_filter.lower() not in table_name.lower():
                continue
            if column_filter and column_filter.lower() not in column_name.lower():
                continue
            if change_filter and change_filter.lower() not in what_changed.lower():
                continue
                
            rows.append({
                "S.No": i,
                "Table": table_name,
                "Column": column_name,
                "What Changed": what_changed,
                "SQL Sproc Impact": r["sproc_count"],
                "Notebook Impact": r["notebook_count"],
                "Actions": "🔧 Fix & PR"  # Add fix button column
            })

        if rows:
            df = pd.DataFrame(rows)
            
            # Display dataframe with action handling
            event = st.dataframe(
                df, 
                use_container_width=True, 
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            # Handle fix button clicks
            if event.selection and event.selection["rows"]:
                selected_row = event.selection["rows"][0]
                if selected_row < len(rows):
                    selected_impact = rows[selected_row]
                    st.info(f"🚀 **Future Feature**: Auto-fix for {selected_impact['Table']} → {selected_impact['Column']} will generate pull request with suggested code changes!")
        else:
            st.info("No results match your filters.")

        # Selection for touchpoints
        st.markdown("### 🎯 Object Touchpoints & Line Numbers")
        options = [f"{i+1}. {(r['entity_name'] or r['table_name'] or 'Entity')} → {r['column_name'] or '(n/a)'}" for i, r in enumerate(top_impacts)]
        choice = st.selectbox("Select a row to inspect touchpoints", options) if options else None

        if choice:
            idx = int(choice.split(".")[0]) - 1
            r = top_impacts[idx]

            colA, colB = st.columns(2)

            with colA:
                st.subheader("Stored Procedures")
                if not r["sprocs"]:
                    st.caption("None")
                else:
                    for sp in r["sprocs"]:
                        with st.expander(f"📜 {sp['name']} ({sp.get('path','')})"):
                            if not sp.get("places_of_use"):
                                st.caption("No specific line references found")
                            else:
                                for pp in sp.get("places_of_use", []):
                                    col_context, col_button = st.columns([4, 1])
                                    with col_context:
                                        st.markdown(
                                            f"<div class='tp'><span class='ln'>Line {pp.get('line_number', 'N/A')}</span>"
                                            f"<code>{(pp.get('context') or '').strip()}</code></div>",
                                            unsafe_allow_html=True
                                        )
                                    with col_button:
                                        if st.button("🔧 Fix & PR", key=f"fix_sp_{sp['name']}_{pp.get('line_number', 'na')}", help="Generate fix and create PR"):
                                            st.info(f"🚀 **Future Feature**: Auto-fix for line {pp.get('line_number', 'N/A')} in {sp['name']} will generate pull request!")

            with colB:
                st.subheader("Notebooks") 
                if not r["notebooks"]:
                    st.caption("None")
                else:
                    for nb in r["notebooks"]:
                        with st.expander(f"💻 {nb['name']} ({nb.get('path','')})"):
                            if not nb.get("places_of_use"):
                                st.caption("No specific line references found")
                            else:
                                for pp in nb.get("places_of_use", []):
                                    col_context, col_button = st.columns([4, 1])
                                    with col_context:
                                        st.markdown(
                                            f"<div class='tp'><span class='ln'>Line {pp.get('line_number', 'N/A')}</span>"
                                            f"<code>{(pp.get('context') or '').strip()}</code></div>",
                                            unsafe_allow_html=True
                                        )
                                    with col_button:
                                        if st.button("🔧 Fix & PR", key=f"fix_nb_{nb['name']}_{pp.get('line_number', 'na')}", help="Generate fix and create PR"):
                                            st.info(f"🚀 **Future Feature**: Auto-fix for line {pp.get('line_number', 'N/A')} in {nb['name']} will generate pull request!")
else:
    st.info("Enter a question, optionally enable strict column filter, set Top N, and click **Analyze Impact**.")