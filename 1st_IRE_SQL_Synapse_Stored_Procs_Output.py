#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""
llm_lineage_to_excel_streamlit.py (LLM-Only Version with Streamlit UI + LangGraph Agent) - ENHANCED UI
------------------------------------------------------------------------------
Streamlit UI for extracting lineage information from stored procedures using minimal LangGraph agents.
 
This version relies exclusively on Azure OpenAI for parsing and does not
contain a regex fallback. Uses minimal LangGraph framework for LLM calls only.
 
Requires:
  pip install openai pandas pyodbc tqdm xlsxwriter streamlit plotly networkx langgraph langchain-openai
 
Env vars (set once in PowerShell):
  setx AZURE_OPENAI_ENDPOINT "https://<your>.cognitiveservices.azure.com/"
  setx AZURE_OPENAI_API_KEY  "your-key"
  setx AZURE_OPENAI_API_VERSION "2024-12-01-preview"
  setx AZURE_OPENAI_DEPLOYMENT "gpt-5-chat"
"""
 
import os
import re
import json
import time
import logging
from typing import List, Dict, Tuple, Optional, Set, TypedDict
 
import pandas as pd
import streamlit as st
from openai import AzureOpenAI
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

# Minimal LangGraph imports
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
 
try:
    import pyodbc
except Exception:
    pyodbc = None
 
# -------------------------
# Azure OpenAI config
# -------------------------
ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "https://.openai.azure.com/")
API_KEY    = os.getenv("AZURE_OPENAI_API_KEY","")
API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

# Token pricing (adjust based on your model)
INPUT_TOKEN_COST = 0.000002
OUTPUT_TOKEN_COST = 0.000008
CACHED_TOKEN_COST = 0.000001
 
if not ENDPOINT or not API_KEY:
    st.error("Missing Azure OpenAI config. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY env vars.")
    st.stop()
 
client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VER,
    azure_endpoint=ENDPOINT,
)

# LangChain client for agent
llm_agent = AzureChatOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version=API_VER,
    azure_deployment=DEPLOYMENT,
    temperature=0,
    max_tokens=4096,
)

# -------------------------
# Minimal Agent State & Workflow
# -------------------------
class LLMState(TypedDict):
    messages: List
    response: Optional[str]
    parsed_json: Optional[Dict]
    cost_info: Dict
    attempt: int
    max_retries: int
    error: Optional[str]

def call_llm_node(state: LLMState) -> LLMState:
    try:
        result = llm_agent.invoke(state["messages"])
        state["response"] = result.content
        
        # Extract cost info
        if hasattr(result, 'usage_metadata') and result.usage_metadata:
            usage = result.usage_metadata
            state["cost_info"]["input_tokens"] = usage.get('input_tokens', 0)
            state["cost_info"]["output_tokens"] = usage.get('output_tokens', 0)
            state["cost_info"]["total_tokens"] = usage.get('total_tokens', 0)
            
            input_cost = state["cost_info"]["input_tokens"] * INPUT_TOKEN_COST
            output_cost = state["cost_info"]["output_tokens"] * OUTPUT_TOKEN_COST
            state["cost_info"]["total_cost"] = input_cost + output_cost
        
        # Try to parse JSON
        try:
            state["parsed_json"] = json.loads(result.content)
            return state
        except json.JSONDecodeError as e:
            state["error"] = f"JSON parsing error: {e}"
            state["attempt"] += 1
            if state["attempt"] < state["max_retries"]:
                time.sleep(2.0 * (2 ** (state["attempt"] - 1)))
                return call_llm_node(state)
            return state
            
    except Exception as e:
        state["error"] = str(e)
        state["attempt"] += 1
        if state["attempt"] < state["max_retries"]:
            time.sleep(2.0 * (2 ** (state["attempt"] - 1)))
            return call_llm_node(state)
        return state

# Create simple workflow
workflow = StateGraph(LLMState)
workflow.add_node("call_llm", call_llm_node)
workflow.set_entry_point("call_llm")
workflow.add_edge("call_llm", END)
agent_app = workflow.compile()

# -------------------------
# Database helpers (UNCHANGED)
# -------------------------
def connect_sqlserver(server: str, database: str, auth_method: str = "ActiveDirectoryIntegrated", db_type: str = "Microsoft SQL Server"):
    if pyodbc is None:
        raise RuntimeError("pyodbc is not installed. Run: pip install pyodbc")
    
    if db_type in ["Microsoft SQL Server", "Microsoft Synapse"]:
        if auth_method == "ActiveDirectoryIntegrated":
            conn_str = (
                "DRIVER={ODBC Driver 18 for SQL Server};"
                f"SERVER={server};DATABASE={database};"
                "Authentication=ActiveDirectoryIntegrated;Encrypt=yes;TrustServerCertificate=yes"
            )
        elif auth_method == "ActiveDirectoryInteractive":
            conn_str = (
                "DRIVER={ODBC Driver 18 for SQL Server};"
                f"SERVER={server};DATABASE={database};"
                "Authentication=ActiveDirectoryInteractive;Encrypt=yes;TrustServerCertificate=yes"
            )
        else:
            conn_str = (
                "DRIVER={ODBC Driver 18 for SQL Server};"
                f"SERVER={server};DATABASE={database};"
                "Trusted_Connection=yes;Encrypt=yes;TrustServerCertificate=yes"
            )
    else:
        raise NotImplementedError(f"Database type {db_type} not yet implemented")
    
    return pyodbc.connect(conn_str, autocommit=True)

def list_procedures(conn, schemas: Optional[List[str]] = None, db_type: str = "Microsoft SQL Server") -> List[Tuple[str, str, str]]:
    cur = conn.cursor()
    
    if db_type in ["Microsoft SQL Server", "Microsoft Synapse"]:
        if schemas:
            q = f"""
            SELECT s.name, o.name, sm.definition
            FROM sys.objects o
            JOIN sys.schemas s ON s.schema_id = o.schema_id
            JOIN sys.sql_modules sm ON sm.object_id = o.object_id
            WHERE o.type = 'P' AND s.name IN ({",".join("?"*len(schemas))})
            ORDER BY s.name, o.name
            """
            cur.execute(q, schemas)
        else:
            q = """
            SELECT s.name, o.name, sm.definition
            FROM sys.objects o
            JOIN sys.schemas s ON s.schema_id = o.schema_id
            JOIN sys.sql_modules sm ON sm.object_id = o.object_id
            WHERE o.type = 'P'
            ORDER BY s.name, o.name
            """
            cur.execute(q)
    else:
        raise NotImplementedError(f"Database type {db_type} not yet implemented")
    
    return [(r[0], r[1], r[2] or "") for r in cur.fetchall()]

def get_database_er_diagram(conn, db_type: str = "Microsoft SQL Server") -> pd.DataFrame:
    cur = conn.cursor()
    
    if db_type in ["Microsoft SQL Server", "Microsoft Synapse"]:
        query = """
        SELECT 
            s.name as schema_name,
            t.name as table_name,
            c.name as column_name,
            ty.name as data_type,
            c.max_length,
            c.is_nullable,
            c.is_identity,
            CASE WHEN pk.is_primary_key IS NULL THEN 0 ELSE pk.is_primary_key END as is_primary_key,
            fk.foreign_table,
            fk.foreign_column
        FROM sys.tables t
        JOIN sys.schemas s ON t.schema_id = s.schema_id
        JOIN sys.columns c ON t.object_id = c.object_id
        JOIN sys.types ty ON c.user_type_id = ty.user_type_id
        LEFT JOIN (
            SELECT ic.object_id, ic.column_id, 1 as is_primary_key
            FROM sys.index_columns ic
            JOIN sys.indexes i ON ic.object_id = i.object_id AND ic.index_id = i.index_id
            WHERE i.is_primary_key = 1
        ) pk ON c.object_id = pk.object_id AND c.column_id = pk.column_id
        LEFT JOIN (
            SELECT 
                fkc.parent_object_id,
                fkc.parent_column_id,
                SCHEMA_NAME(rt.schema_id) + '.' + rt.name as foreign_table,
                rc.name as foreign_column
            FROM sys.foreign_key_columns fkc
            JOIN sys.tables rt ON fkc.referenced_object_id = rt.object_id
            JOIN sys.columns rc ON fkc.referenced_object_id = rc.object_id AND fkc.referenced_column_id = rc.column_id
        ) fk ON c.object_id = fk.parent_object_id AND c.column_id = fk.parent_column_id
        ORDER BY s.name, t.name, c.column_id
        """
    else:
        raise NotImplementedError(f"Database type {db_type} not yet implemented")
    
    try:
        cur.execute(query)
        columns = [desc[0] for desc in cur.description]
        data = cur.fetchall()
        
        if not data:
            return pd.DataFrame(columns=columns)
        
        df_data = []
        for row in data:
            row_list = list(row)
            if len(row_list) != len(columns):
                while len(row_list) < len(columns):
                    row_list.append(None)
                row_list = row_list[:len(columns)]
            df_data.append(row_list)
        
        return pd.DataFrame(df_data, columns=columns)
        
    except Exception as e:
        try:
            simple_query = """
            SELECT 
                s.name as schema_name,
                t.name as table_name,
                c.name as column_name,
                ty.name as data_type
            FROM sys.tables t
            JOIN sys.schemas s ON t.schema_id = s.schema_id
            JOIN sys.columns c ON t.object_id = c.object_id
            JOIN sys.types ty ON c.user_type_id = ty.user_type_id
            ORDER BY s.name, t.name, c.column_id
            """
            cur.execute(simple_query)
            columns = [desc[0] for desc in cur.description]
            data = cur.fetchall()
            return pd.DataFrame(data, columns=columns)
        except Exception:
            return pd.DataFrame()
    finally:
        cur.close()

def create_er_diagram(er_df: pd.DataFrame) -> go.Figure:
    if er_df.empty:
        return go.Figure()
    
    G = nx.Graph()
    
    table_info = er_df.groupby(['schema_name', 'table_name']).agg({
        'column_name': 'count',
        'is_primary_key': 'sum'
    }).reset_index()
    table_info.columns = ['schema_name', 'table_name', 'column_count', 'pk_count']
    
    for _, row in table_info.iterrows():
        table_full_name = f"{row['schema_name']}.{row['table_name']}"
        G.add_node(table_full_name, 
                  columns=row['column_count'], 
                  primary_keys=row['pk_count'])
    
    relationships = er_df[er_df['foreign_table'].notna()].copy()
    for _, row in relationships.iterrows():
        source = f"{row['schema_name']}.{row['table_name']}"
        target = row['foreign_table']
        if target in G.nodes:
            G.add_edge(source, target)
    
    try:
        pos = nx.spring_layout(G, k=3, iterations=50)
    except:
        pos = nx.random_layout(G)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                           line=dict(width=2, color='#888'),
                           hoverinfo='none',
                           mode='lines')
    
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        columns = G.nodes[node].get('columns', 0)
        pks = G.nodes[node].get('primary_keys', 0)
        connections = len(list(G.neighbors(node)))
        
        node_info.append(f"{node}<br>Columns: {columns}<br>Primary Keys: {pks}<br>Relationships: {connections}")
        node_colors.append(columns)
    
    node_trace = go.Scatter(x=node_x, y=node_y,
                           mode='markers+text',
                           hoverinfo='text',
                           hovertext=node_info,
                           text=node_text,
                           textposition="middle center",
                           marker=dict(size=[min(50, max(20, c*2)) for c in node_colors],
                                     color=node_colors,
                                     colorscale='Viridis',
                                     showscale=True,
                                     colorbar=dict(title="Number of Columns"),
                                     line=dict(width=2, color='black')))
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                        title=dict(text='Interactive ER Diagram', font=dict(size=16)),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Click and drag to explore relationships",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor="left", yanchor="bottom",
                            font=dict(color="gray", size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600))
    
    return fig

def get_single_procedure(conn, schema: str, name: str, db_type: str = "Microsoft SQL Server") -> Optional[Tuple[str, str, str]]:
    cur = conn.cursor()
    
    if db_type in ["Microsoft SQL Server", "Microsoft Synapse"]:
        cur.execute("""
            SELECT s.name, o.name, sm.definition
            FROM sys.objects o
            JOIN sys.schemas s ON s.schema_id = o.schema_id
            JOIN sys.sql_modules sm ON sm.object_id = o.object_id
            WHERE o.type = 'P' AND s.name = ? AND o.name = ?
        """, (schema, name))
    else:
        raise NotImplementedError(f"Database type {db_type} not yet implemented")
    
    result = cur.fetchone()
    return (result[0], result[1], result[2] or "") if result else None
 
# -------------------------
# Prompt (UNCHANGED)
# -------------------------
SYSTEM_MSG = (
    "You are a 15+ years experienced SQL Senior Developer and Database Architect. Produce STRICT JSON only (no prose, no code fences).\n"
    "\n"
    "Goal: Understand the entire database and Analyze the stored procedure line by line with deep expertise. Understand CTEs, multi-join subqueries, complex alias usage, and trace columns back to their source tables.\n"
    "\n"
    "OUTPUT FORMAT (strict JSON):\n"
    "{\n"
    '  "sproc": "schema.name",\n'
    '  "refs": [\n'
    '    {"table": "schema.tablename", "columns": ["col1","col2", ...]}\n'
    "  ]\n"
    "}\n"
    "\n"
    "EXPERT ANALYSIS RULES:\n"
    "1) Only JSON. No explanations, comments, or extra keys.\n"
    "2) Get into every stored procedure deeply - understand the complete flow.\n"
    "3) Understand line by line execution, including CTEs, temp tables, table variables.\n"
    "4) Master complex multi-join subqueries and trace column origins through aliases.\n"
    "5) Identify ALL base table references from SELECT, INSERT, UPDATE, DELETE, MERGE operations.\n"
    "6) List out every column that are related to all base tables from SELECT, INSERT, UPDATE, DELETE, MERGE operations\n"
    "7) Apply 15+ years of SQL expertise to handle complex scenarios and edge cases.\n"
    "8) Deduplicate case-insensitively and present unique, sorted column lists per table.\n"
    "9) Understand the need, you need to get into the entire code, then start analyzing it deeply, then start preparing a table of all base table references and their columns that are actively used in all stored procs.\n"
    "10) Ignore the stored procs with _bkp or _old or _bak or _date(for example: [ctrl].[uspLoad_MappingSubscriptionPK_kh_20211115]) in their names.\n"
    "11) Never use *, try to list out all columns that are used in stored procs.\n"
    "12) If stored procedure is using dynamic SQL, analyze the dynamic SQL separately and include its base table references and columns.\n"
    "13) If a stored procedure is calling other stored procedures, analyze those as well and include their base table references and columns.\n"\
    "14) If a stored procedure directly uses * in any SELECT, INSERT, UPDATE, MERGE, DELETE then you can use * in column name and tag it to its base table.\n"
    "15) Always cross check your output with the original SQL code to ensure accuracy before adding to final dataframe."
)
 
USER_TMPL = """Stored procedure code:
 
{code}
 
 
Return STRICT JSON for this one procedure only.
"""
 
CHUNK_SIZE_LINES = 2500
 
USER_TMPL_CHUNK = """This is a chunk of a larger stored procedure. The procedure's CREATE statement is included for context.
Analyze ONLY the provided SQL code chunk to identify base tables and their referenced columns.
 
Stored procedure chunk:
{code}
 
 
Return STRICT JSON for this chunk only.
"""

# -------------------------
# Modified LLM function (MINIMAL CHANGE)
# -------------------------
def ask_llm(definition: str, retries: int = 4, backoff: float = 2.0, is_chunk: bool = False, llm_model: str = "GPT-5") -> Tuple[Optional[Dict], Dict]:
    user_content = (USER_TMPL_CHUNK if is_chunk else USER_TMPL).format(code=definition)
    
    initial_state = LLMState(
        messages=[
            SystemMessage(content=SYSTEM_MSG),
            HumanMessage(content=user_content)
        ],
        response=None,
        parsed_json=None,
        cost_info={
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        },
        attempt=0,
        max_retries=retries,
        error=None
    )
    
    final_state = agent_app.invoke(initial_state)
    
    if final_state["error"] and not final_state["parsed_json"]:
        st.warning(f"LLM error: {final_state['error']}")
    
    return final_state["parsed_json"], final_state["cost_info"]
 
def process_procedure_with_chunking(defn: str, progress_bar=None, llm_model: str = "GPT-5") -> Tuple[Dict[str, Set[str]], Dict]:
    lines = defn.splitlines()
    aggregated_refs: Dict[str, Set[str]] = {}
    total_cost_info = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
        "total_cost": 0.0
    }
 
    if len(lines) <= 2500:
        llm_obj, cost_info = ask_llm(defn, llm_model=llm_model)
        for key in total_cost_info:
            total_cost_info[key] += cost_info[key]
            
        if llm_obj and isinstance(llm_obj, dict) and "refs" in llm_obj:
            for r in llm_obj.get("refs", []):
                tbl = (r.get("table") or "").strip()
                if not tbl:
                    continue
                cols = [c for c in (r.get("columns") or []) if isinstance(c, str)]
                aggregated_refs.setdefault(tbl.lower(), set()).update([c.lower() for c in cols])
        return aggregated_refs, total_cost_info

    st.info(f"Procedure is long ({len(lines)} lines), splitting into 3 parts...")
   
    header_end_index = 0
    for i, line in enumerate(lines):
        if re.search(r"\b(AS|BEGIN)\b", line, re.IGNORECASE):
            header_end_index = i + 1
            break
    header = "\n".join(lines[:header_end_index])
    body_lines = lines[header_end_index:]
    
    body_length = len(body_lines)
    chunk_size = body_length // 3
    remainder = body_length % 3
    
    chunk_boundaries = []
    start = 0
    for i in range(3):
        size = chunk_size + (1 if i < remainder else 0)
        end = start + size
        chunk_boundaries.append((start, end))
        start = end
    
    for chunk_idx, (start, end) in enumerate(chunk_boundaries):
        chunk_body = "\n".join(body_lines[start:end])
        chunk_full_code = header + "\n" + chunk_body
       
        chunk_num = chunk_idx + 1
        st.info(f"Processing chunk {chunk_num}/3...")
        if progress_bar:
            progress_bar.progress(chunk_num / 3)
            
        llm_obj, cost_info = ask_llm(chunk_full_code, is_chunk=True, llm_model=llm_model)
        
        for key in total_cost_info:
            total_cost_info[key] += cost_info[key]
 
        if llm_obj and isinstance(llm_obj, dict) and "refs" in llm_obj:
            for r in llm_obj.get("refs", []):
                tbl = (r.get("table") or "").strip()
                if not tbl:
                    continue
                cols = [c for c in (r.get("columns") or []) if isinstance(c, str)]
                aggregated_refs.setdefault(tbl.lower(), set()).update([c.lower() for c in cols])
 
    return aggregated_refs, total_cost_info

# -------------------------
# Enhanced Streamlit UI (UNCHANGED)
# -------------------------
def create_gradient_background():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    }
    .main-header {
        background: linear-gradient(90deg, #56ab2f 0%, #7dd87d 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
    }
    .sidebar .element-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .metric-container {
        background: linear-gradient(135deg, #66d9a5 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .status-success {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border-left: 5px solid #28a745;
    }
    .status-error {
        background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border-left: 5px solid #dc3545;
    }
    .info-panel {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .connection-status {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 10px 0;
    }
    .status-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    .status-dot.success {
        background-color: #28a745;
    }
    .status-dot.error {
        background-color: #dc3545;
    }
    .status-dot.warning {
        background-color: #ffc107;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def create_animated_header():
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI-Powered Database Navigator Agent</h1>
        <p style="font-size: 1.2em; margin-top: 1rem; opacity: 0.9;">
            Discover, Analyze & Map Your Database Universe with Advanced AI
        </p>
        <div style="margin-top: 1rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">🔍 ER Diagrams</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">📊 Lineage Analysis</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">🤖 AI-Powered</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_feature_cards():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Smart ER Analysis</h3>
            <p>Automatically discover database relationships and create interactive visualizations</p>
            <div style="color: #2d5016; font-weight: bold;">• Interactive Diagrams</div>
            <div style="color: #2d5016; font-weight: bold;">• Relationship Mapping</div>
            <div style="color: #2d5016; font-weight: bold;">• Schema Discovery</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Lineage Extraction</h3>
            <p>AI-powered analysis of stored procedures to trace data lineage with precision</p>
            <div style="color: #3e6b2e; font-weight: bold;">• Deep Code Analysis</div>
            <div style="color: #3e6b2e; font-weight: bold;">• Column Tracing</div>
            <div style="color: #3e6b2e; font-weight: bold;">• Dependency Mapping</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 AI Intelligence</h3>
            <p>Leverage cutting-edge LLM models for expert-level SQL analysis and insights</p>
            <div style="color: #ff6b6b; font-weight: bold;">• Multiple LLM Support</div>
            <div style="color: #ff6b6b; font-weight: bold;">• Expert Analysis</div>
            <div style="color: #ff6b6b; font-weight: bold;">• Cost Optimization</div>
        </div>
        """, unsafe_allow_html=True)

def create_status_panel(config_status, db_type, llm_model):
    st.markdown("""
    <div class="info-panel">
        <h3 style="color: #333; margin-bottom: 1rem;">🎛️ System Status</h3>
    """, unsafe_allow_html=True)
    
    if ENDPOINT and API_KEY:
        st.markdown("""
        <div class="connection-status">
            <div class="status-dot success"></div>
            <strong style="color: #28a745;">Azure OpenAI Connected</strong>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="margin-left: 22px; color: #666;">
            🤖 Model: <strong>{llm_model}</strong><br>
            🗄️ Database: <strong>{db_type}</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="connection-status">
            <div class="status-dot error"></div>
            <strong style="color: #dc3545;">Azure OpenAI Not Configured</strong>
        </div>
        """, unsafe_allow_html=True)
        
        st.code("""
setx AZURE_OPENAI_ENDPOINT "https://your-resource.openai.azure.com/"
setx AZURE_OPENAI_API_KEY "your-api-key"
setx AZURE_OPENAI_DEPLOYMENT "your-deployment-name"
        """, language="bash")
    
    if db_type not in ["Microsoft SQL Server", "Microsoft Synapse"]:
        st.markdown(f"""
        <div class="connection-status">
            <div class="status-dot warning"></div>
            <strong style="color: #ffc107;">{db_type} Support Coming Soon</strong>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def create_enhanced_sidebar():
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #56ab2f 0%, #7dd87d 100%); 
                padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;">
        <h2>⚙️ Configuration</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🔗 Database Connection")
    
    db_type = st.sidebar.selectbox(
        "Database Type",
        ["Microsoft SQL Server", "Microsoft Synapse", "Snowflake", "PostgreSQL", "MySQL"],
        help="Select your database platform"
    )
    
    server = st.sidebar.text_input(
        "Server", 
        placeholder="your-server.database.windows.net",
        help="Database server address"
    )
    
    database = st.sidebar.text_input(
        "Database", 
        placeholder="your-database-name",
        help="Target database name"
    )
    
    if db_type in ["Microsoft SQL Server", "Microsoft Synapse"]:
        auth_method = st.sidebar.selectbox(
            "Authentication Method", 
            ["ActiveDirectoryIntegrated", "ActiveDirectoryInteractive", "Windows"],
            help="Choose authentication method"
        )
    else:
        auth_method = "Windows"
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 🎯 Analysis Options")
    
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["ER Diagram Analysis", "Stored Procedure Lineage", "Both"],
        help="Choose what type of analysis to perform"
    )
    
    schema_name = st.sidebar.text_input(
        "Schema Name", 
        value="ctrl", 
        placeholder="ctrl",
        help="Target schema for analysis"
    )
    
    llm_model = st.sidebar.selectbox(
        "LLM Model",
        ["GPT-5", "GPT-4", "GPT-3.5", "Claude 4", "Gemini 2.5 Pro"],
        index=0,
        help="Select AI model for analysis"
    )
    
    if analysis_mode in ["Stored Procedure Lineage", "Both"]:
        process_mode = st.sidebar.radio(
            "Processing Mode",
            ["Single Stored Procedure", "All Procedures in Schema"],
            help="Analyze single procedure or entire schema"
        )
        
        sproc_name = ""
        if process_mode == "Single Stored Procedure":
            sproc_name = st.sidebar.text_input(
                "Stored Procedure Name", 
                placeholder="your_procedure_name",
                help="Name of the specific stored procedure"
            )
    else:
        process_mode = "All Procedures in Schema"
        sproc_name = ""
    
    return db_type, server, database, auth_method, analysis_mode, schema_name, llm_model, process_mode, sproc_name

def create_enhanced_metrics(df_len, unique_tables, unique_procs, total_cost):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin: 0; font-size: 2em;">{df_len:,}</h3>
            <p style="margin: 0.5rem 0 0 0;">Total Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin: 0; font-size: 2em;">{unique_tables}</h3>
            <p style="margin: 0.5rem 0 0 0;">Unique Tables</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin: 0; font-size: 2em;">{unique_procs}</h3>
            <p style="margin: 0.5rem 0 0 0;">Procedures</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin: 0; font-size: 2em;">${total_cost:.4f}</h3>
            <p style="margin: 0.5rem 0 0 0;">Analysis Cost</p>
        </div>
        """, unsafe_allow_html=True)

def create_enhanced_er_metrics(er_df):
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin: 0; font-size: 2em;">{er_df['table_name'].nunique()}</h3>
            <p style="margin: 0.5rem 0 0 0;">Total Tables</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin: 0; font-size: 2em;">{len(er_df):,}</h3>
            <p style="margin: 0.5rem 0 0 0;">Total Columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_c:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin: 0; font-size: 2em;">{er_df['schema_name'].nunique()}</h3>
            <p style="margin: 0.5rem 0 0 0;">Schemas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_d:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin: 0; font-size: 2em;">{er_df['foreign_table'].notna().sum()}</h3>
            <p style="margin: 0.5rem 0 0 0;">Relationships</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="AI Database Navigator",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    create_gradient_background()
    create_animated_header()
    create_feature_cards()
    
    db_type, server, database, auth_method, analysis_mode, schema_name, llm_model, process_mode, sproc_name = create_enhanced_sidebar()
    
    main_col, status_col = st.columns([3, 1])
    
    with status_col:
        create_status_panel("", db_type, llm_model)
    
    with main_col:
        if analysis_mode in ["ER Diagram Analysis", "Both"]:
            st.markdown("### 📊 ER Diagram Analysis")
            
            if st.button("🎨 Generate Interactive ER Diagram", key="er_button", help="Create visual database relationships"):
                if not server or not database:
                    st.error("🚫 Please provide server and database names")
                    return
                
                if db_type not in ["Microsoft SQL Server", "Microsoft Synapse"]:
                    st.error(f"🚫 Database type '{db_type}' is not yet implemented")
                    return
                
                try:
                    with st.spinner("🔄 Connecting to database and analyzing ER diagram..."):
                        conn = connect_sqlserver(server, database, auth_method, db_type)
                        er_df = get_database_er_diagram(conn, db_type)
                        conn.close()
                    
                    if not er_df.empty:
                        st.success(f"✅ ER Diagram analysis complete! Found {er_df['table_name'].nunique()} tables")
                        
                        create_enhanced_er_metrics(er_df)
                        
                        st.markdown("#### 🌐 Interactive Database Visualization")
                        er_fig = create_er_diagram(er_df)
                        st.plotly_chart(er_fig, use_container_width=True)
                        
                        with st.expander("📋 Detailed Schema Information", expanded=False):
                            st.dataframe(
                                er_df, 
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        er_csv = er_df.to_csv(index=False)
                        er_filename = f"er_diagram_{database}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        
                        st.download_button(
                            label="📥 Download ER Analysis Results",
                            data=er_csv,
                            file_name=er_filename,
                            mime="text/csv",
                            help="Download complete ER diagram data as CSV"
                        )
                    else:
                        st.warning("⚠️ No ER diagram data found")
                        
                except Exception as e:
                    st.error(f"❌ ER Diagram Error: {str(e)}")
        
        if analysis_mode == "Both":
            st.markdown("---")
        
        if analysis_mode in ["Stored Procedure Lineage", "Both"]:
            st.markdown("### 🔍 Stored Procedure Lineage Analysis")
            
            button_label = "🚀 Extract AI-Powered Lineage" if analysis_mode == "Stored Procedure Lineage" else "🔬 Analyze Stored Procedures"
            
            if st.button(button_label, key="lineage_button", help="AI-powered stored procedure analysis"):
                if not server or not database:
                    st.error("🚫 Please provide server and database names")
                    return
                
                if db_type not in ["Microsoft SQL Server", "Microsoft Synapse"]:
                    st.error(f"🚫 Database type '{db_type}' is not yet implemented")
                    return
                    
                if process_mode == "Single Stored Procedure" and not sproc_name:
                    st.error("🚫 Please provide stored procedure name")
                    return
                
                if not ENDPOINT or not API_KEY:
                    st.error("🚫 Azure OpenAI configuration is missing. Please set environment variables.")
                    return
                
                try:
                    with st.spinner("🔗 Establishing database connection..."):
                        conn = connect_sqlserver(server, database, auth_method, db_type)
                    
                    if process_mode == "Single Stored Procedure":
                        with st.spinner(f"🔍 Locating procedure {schema_name}.{sproc_name}..."):
                            proc_result = get_single_procedure(conn, schema_name, sproc_name, db_type)
                        if not proc_result:
                            st.error(f"🚫 Stored procedure '{schema_name}.{sproc_name}' not found.")
                            conn.close()
                            return
                        procs = [proc_result]
                        st.success(f"✅ Found stored procedure: **{schema_name}.{sproc_name}**")
                    else:
                        with st.spinner(f"🔍 Discovering procedures in schema '{schema_name}'..."):
                            procs = list_procedures(conn, [schema_name], db_type)
                        st.success(f"✅ Found **{len(procs)}** stored procedures in schema **'{schema_name}'**")
                    
                    if not procs:
                        st.warning("⚠️ No stored procedures found")
                        conn.close()
                        return
                    
                    st.markdown("#### 🤖 AI Analysis in Progress")
                    rows: List[Dict[str, any]] = []
                    total_cost = 0.0
                    
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        cost_display = st.empty()
                    
                    for i, (sch, name, defn) in enumerate(procs):
                        sproc_full = f"{sch}.{name}"
                        
                        status_text.markdown(f"""
                        <div class="progress-container">
                            <h4>🔄 Processing: {sproc_full}</h4>
                            <p>Progress: {i+1}/{len(procs)} procedures</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        refs, cost_info = process_procedure_with_chunking(defn, progress_bar, llm_model)
                        total_cost += cost_info["total_cost"]
                        
                        cost_display.info(f"💰 Running cost: ${total_cost:.6f}")
                        
                        if not refs:
                            st.warning(f"⚠️ No references found for {sproc_full}")
                            continue
                        
                        for tbl_key, cols in refs.items():
                            table_out = tbl_key if "." in tbl_key else f"dbo.{tbl_key}"
                            sorted_cols = sorted(list(cols))
                           
                            if sorted_cols:
                                for c in sorted_cols:
                                    rows.append({
                                        "sproc name": sproc_full,
                                        "table name": table_out,
                                        "column name": c,
                                        "input tokens": cost_info["input_tokens"],
                                        "output tokens": cost_info["output_tokens"],
                                        "cached tokens": cost_info["cached_tokens"],
                                        "total tokens": cost_info["total_tokens"],
                                        "total cost ($)": f"${cost_info['total_cost']:.6f}"
                                    })
                            else:
                                rows.append({
                                    "sproc name": sproc_full,
                                    "table name": table_out,
                                    "column name": "*",
                                    "input tokens": cost_info["input_tokens"],
                                    "output tokens": cost_info["output_tokens"],
                                    "cached tokens": cost_info["cached_tokens"],
                                    "total tokens": cost_info["total_tokens"],
                                    "total cost ($)": f"${cost_info['total_cost']:.6f}"
                                })
                        
                        progress_bar.progress((i + 1) / len(procs))
                    
                    progress_bar.empty()
                    status_text.empty()
                    cost_display.empty()
                    
                    conn.close()
                    
                    if rows:
                        df = pd.DataFrame(rows)
                        
                        st.balloons()
                        st.success(f"🎉 Analysis Complete! Extracted **{len(df):,}** lineage records")
                        
                        create_enhanced_metrics(len(df), df["table name"].nunique(), df["sproc name"].nunique(), total_cost)
                        
                        st.markdown("#### 📈 Lineage Analysis Results")
                        
                        search_term = st.text_input("🔍 Search results", placeholder="Filter by table, procedure, or column name...")
                        if search_term:
                            filtered_df = df[df.apply(lambda row: search_term.lower() in row.astype(str).str.lower().to_string(), axis=1)]
                            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
                        else:
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv = df.to_csv(index=False)
                            filename = f"lineage_{schema_name}_{sproc_name if sproc_name else 'all'}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            st.download_button(
                                label="📥 Download Complete Results (CSV)",
                                data=csv,
                                file_name=filename,
                                mime="text/csv",
                                help="Download complete lineage analysis results"
                            )
                        
                        with col2:
                            summary = f"""
# Lineage Analysis Summary

**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Schema:** {schema_name}
**Processing Mode:** {process_mode}
**AI Model:** {llm_model}

## Results Summary
- **Total Records:** {len(df):,}
- **Unique Tables:** {df["table name"].nunique()}
- **Stored Procedures:** {df["sproc name"].nunique()}
- **Total Analysis Cost:** ${total_cost:.6f}

## Top Tables by Usage
{df.groupby('table name').size().sort_values(ascending=False).head(10).to_string()}
                            """
                            
                            st.download_button(
                                label="📄 Download Summary Report (MD)",
                                data=summary,
                                file_name=f"summary_{schema_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown",
                                help="Download analysis summary report"
                            )
                    else:
                        st.warning("⚠️ No lineage data extracted")
                        
                except pyodbc.Error as e:
                    st.error(f"❌ Database connection error: {str(e)}")
                    st.info("💡 Please check your server name, database name, and authentication method.")
                except Exception as e:
                    st.error(f"❌ Analysis Error: {str(e)}")
                    with st.expander("🔧 Debug Information"):
                        import traceback
                        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()