#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
notebook_lineage_streamlit_connector.py
------------------------------------------------------------------------------
A Streamlit web application that connects to data sources like Databricks,
fetches notebooks, and uses Azure OpenAI with LangGraph agents to extract data lineage.

Requires:
  pip install streamlit pandas openai databricks-sdk langgraph langchain-openai
  (for AAD) pip install azure-identity
"""

import os
import json
import base64
import pandas as pd
import streamlit as st
from openai import AzureOpenAI
from databricks.sdk.service.workspace import ObjectType, ExportFormat
from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI

# Optional Azure Identity for AAD Integrated auth
try:
    from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, ClientSecretCredential
    AZURE_IDENTITY_INSTALLED = True
except ImportError:
    AZURE_IDENTITY_INSTALLED = False

# Try to import the Databricks SDK, handle if not installed
try:
    from databricks.sdk import WorkspaceClient
    DATABRICKS_SDK_INSTALLED = True
except ImportError:
    DATABRICKS_SDK_INSTALLED = False

# --- 1. Azure OpenAI Configuration ---
ENDPOINT   = st.secrets.get("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT"))
API_KEY    = st.secrets.get("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY"))
DEPLOYMENT = st.secrets.get("AZURE_OPENAI_DEPLOYMENT", os.getenv("AZURE_OPENAI_DEPLOYMENT"))
API_VER    = "2024-02-01"

INPUT_TOKEN_COST = 0.005 / 1000
OUTPUT_TOKEN_COST = 0.015 / 1000

if not all([ENDPOINT, API_KEY, DEPLOYMENT]):
    st.error("Azure OpenAI configuration is missing. Please set it in .streamlit/secrets.toml")
    st.stop()

client = AzureOpenAI(api_key=API_KEY, api_version=API_VER, azure_endpoint=ENDPOINT)

# Initialize LangChain Azure OpenAI for agents
llm = AzureChatOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version=API_VER,
    azure_deployment=DEPLOYMENT,
    temperature=0.0
)

# --- 2. System Prompt ---
SYSTEM_MSG = (
    "You are a 15+ years experienced Spark Developer and ETL/ELT Architect. Produce STRICT JSON only (no prose, no code fences).\n"
    "\n"
    "Goal: Understand the entire notebook and Analyze the notebooks line by line with deep expertise. Understand each cell, multi-join dataframes, complex alias usage, and trace columns back to their source tables.\n"
    "\n"
    "OUTPUT FORMAT (strict JSON):\n"
    "{\n"
    '  "notebook": "schema.name",\n'
    '  "refs": [\n'
    '    {"table": "schema.tablename", "columns": ["col1","col2", ...]}\n'
    "  ]\n"
    "}\n"
    "\n"
    "EXPERT ANALYSIS RULES:\n"
    "1) Only JSON. No explanations, comments, or extra keys.\n"
    "2) Get into every notebook deeply - understand the complete flow.\n"
    "3) Identify ALL base dataframe references from read operations (e.g., spark.read.table, spark.read.csv, etc.) and write operations (e.g., df.write.saveAsTable).\n"
    "4) List out every column that is related to all base tables and destinations.\n"
    "5) Apply 15+ years of spark expertise to handle complex scenarios.\n"
    "6) Deduplicate case-insensitively and present unique, sorted column lists per table.\n"
    "7) For file-based sources/destinations (CSV, Parquet, Delta), treat the file path or table name as the 'table'."
)

# --- 3. LangGraph State Definition ---
class LineageState(TypedDict):
    endpoint: str
    token: str
    notebooks: List[Dict[str, str]]
    current_notebook_idx: int
    lineage_results: List[Dict[str, Any]]
    total_cost: float
    error_messages: List[str]
    status: str

# --- 4. LangGraph Agent Functions ---
def fetch_notebooks_agent(state: LineageState) -> LineageState:
    """Agent to fetch notebooks from Databricks."""
    if not DATABRICKS_SDK_INSTALLED:
        state["error_messages"].append("The `databricks-sdk` is not installed. Please run `pip install databricks-sdk`.")
        state["status"] = "error"
        return state
    
    notebooks = []
    try:
        w = WorkspaceClient(host=state["endpoint"], token=state["token"])
        
        # Test connection
        try:
            list(w.workspace.list("/", limit=1))
        except Exception as e:
            state["error_messages"].append(f"Connection test failed: {str(e)}")
            state["status"] = "error"
            return state

        queue = ["/"]
        paths_to_export = []
        
        # Collect all notebook paths
        while queue:
            current_path = queue.pop(0)
            try:
                for item in w.workspace.list(current_path):
                    if item.object_type == ObjectType.DIRECTORY:
                        queue.append(item.path)
                    elif item.object_type == ObjectType.NOTEBOOK:
                        paths_to_export.append(item.path)
            except Exception as e:
                state["error_messages"].append(f"Could not list objects in path '{current_path}': {e}")

        if not paths_to_export:
            state["error_messages"].append("No notebooks found in the workspace.")
            state["status"] = "error"
            return state
        
        # Export notebook content
        for path in paths_to_export:
            try:
                resp = w.workspace.export(path=path, format=ExportFormat.SOURCE)

                if isinstance(resp, (bytes, bytearray)):
                    exported_bytes = bytes(resp)
                elif hasattr(resp, "content"):
                    if isinstance(resp.content, (bytes, bytearray)):
                        exported_bytes = bytes(resp.content)
                    else:
                        exported_bytes = base64.b64decode(resp.content or "")
                else:
                    exported_bytes = str(resp).encode("utf-8")

                content = exported_bytes.decode("utf-8", errors="ignore")
                notebooks.append({"path": path, "content": content})
                
            except Exception as e:
                state["error_messages"].append(f"Could not export notebook '{path}': {e}")
                continue
        
        state["notebooks"] = notebooks
        state["status"] = "notebooks_fetched"
        
    except Exception as e:
        state["error_messages"].append(f"Failed to connect or fetch from Databricks: {e}")
        state["status"] = "error"
    
    return state

def analyze_notebook_agent(state: LineageState) -> LineageState:
    """Agent to analyze a single notebook for lineage."""
    if state["current_notebook_idx"] >= len(state["notebooks"]):
        state["status"] = "analysis_complete"
        return state
    
    notebook = state["notebooks"][state["current_notebook_idx"]]
    notebook_path = notebook["path"]
    content = notebook["content"]
    
    user_prompt = f"notebook path: {notebook_path}\nnotebook code:\n\n{content}\n\nReturn STRICT JSON."
    cost_info = {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}
    
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
        )
        
        lineage_info = json.loads(response.choices[0].message.content)
        # Force the notebook name to just the base name from the workspace path
        lineage_info["notebook"] = os.path.basename(notebook_path)
        
        usage = response.usage
        if usage:
            cost_info["input_tokens"] = usage.prompt_tokens
            cost_info["output_tokens"] = usage.completion_tokens
            cost_info["total_cost"] = (usage.prompt_tokens * INPUT_TOKEN_COST) + (usage.completion_tokens * OUTPUT_TOKEN_COST)
        
        state["lineage_results"].append(lineage_info)
        state["total_cost"] += cost_info["total_cost"]
        
    except json.JSONDecodeError as e:
        state["error_messages"].append(f"Failed to parse LLM response for {notebook_path}: {e}")
    except Exception as e:
        state["error_messages"].append(f"LLM request failed for {notebook_path}: {e}")
    
    state["current_notebook_idx"] += 1
    
    # Check if more notebooks to process
    if state["current_notebook_idx"] >= len(state["notebooks"]):
        state["status"] = "analysis_complete"
    else:
        state["status"] = "analyzing"
    
    return state

def should_continue_analysis(state: LineageState) -> str:
    """Conditional edge to determine if analysis should continue."""
    if state["status"] == "analysis_complete":
        return "end"
    elif state["status"] == "analyzing":
        return "continue"
    elif state["status"] == "error":
        return "end"
    else:
        return "continue"

# --- 5. Build LangGraph Workflow ---
def create_lineage_workflow():
    """Create the LangGraph workflow for notebook lineage analysis."""
    workflow = StateGraph(LineageState)
    
    # Add nodes
    workflow.add_node("fetch_notebooks", fetch_notebooks_agent)
    workflow.add_node("analyze_notebook", analyze_notebook_agent)
    
    # Define edges
    workflow.set_entry_point("fetch_notebooks")
    workflow.add_edge("fetch_notebooks", "analyze_notebook")
    workflow.add_conditional_edges(
        "analyze_notebook",
        should_continue_analysis,
        {
            "continue": "analyze_notebook",
            "end": END
        }
    )
    
    return workflow.compile()

# --- 6. Enhanced AAD Authentication Functions ---
def _get_aad_bearer_token_with_tenant(tenant_id: str = None) -> str:
    """Acquire an AAD access token for Azure Databricks resource with specific tenant."""
    if not AZURE_IDENTITY_INSTALLED:
        raise RuntimeError("azure-identity is not installed. Run: pip install azure-identity")

    scope = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default"
    
    try:
        if tenant_id:
            credential = InteractiveBrowserCredential(tenant_id=tenant_id)
            token = credential.get_token(scope)
            return token.token
        else:
            try:
                dac = DefaultAzureCredential(exclude_interactive_browser_credential=True)
                token = dac.get_token(scope)
                return token.token
            except Exception:
                ibc = InteractiveBrowserCredential()
                token = ibc.get_token(scope)
                return token.token
    except Exception as e:
        raise RuntimeError(f"Failed to acquire AAD token: {str(e)}")

def test_databricks_connection(endpoint: str, token: str) -> bool:
    """Test the Databricks connection before proceeding."""
    if not DATABRICKS_SDK_INSTALLED:
        return False
    
    try:
        w = WorkspaceClient(host=endpoint, token=token)
        list(w.workspace.list("/", limit=1))
        return True
    except Exception as e:
        st.error(f"Connection test failed: {str(e)}")
        return False

# --- 7. Agent Integration Function ---
def run_lineage_analysis_agent(endpoint: str, token: str):
    """Main function to run the LangGraph agent workflow."""
    workflow = create_lineage_workflow()
    
    initial_state: LineageState = {
        "endpoint": endpoint,
        "token": token,
        "notebooks": [],
        "current_notebook_idx": 0,
        "lineage_results": [],
        "total_cost": 0.0,
        "error_messages": [],
        "status": "starting"
    }
    
    try:
        final_state = workflow.invoke(initial_state)
        return final_state
    except Exception as e:
        st.error(f"Agent workflow failed: {str(e)}")
        return initial_state

def get_notebooks_from_source(notebook_type, endpoint, auth_type, token, tenant_id=None):
    """Router function to call the correct connector using agents."""
    if notebook_type == "Azure Databricks":
        if auth_type == "Access Token":
            if not token:
                st.error("Please provide a Databricks Access Token.")
                return None
            return run_lineage_analysis_agent(endpoint, token)
        elif auth_type == "AD Integrated":
            try:
                aad_token = _get_aad_bearer_token_with_tenant(tenant_id)
                return run_lineage_analysis_agent(endpoint, aad_token)
            except Exception as e:
                st.error(f"Failed to sign in with AAD Integrated auth: {e}")
                st.info("💡 **Troubleshooting Tips:**")
                st.info("1. Make sure you're logged into Azure CLI: `az login`")
                st.info("2. Try specifying your tenant ID in the sidebar")
                st.info("3. Ensure Azure Databricks is available in your tenant")
                st.info("4. Consider using Access Token authentication instead")
                return None
        else:
            st.error("For Azure Databricks, please select 'Access Token' or 'AD Integrated'.")
            return None
    else:
        st.warning(f"The connector for '{notebook_type}' is not yet implemented.")
        return None

def process_lineage_to_dataframe(all_results: list) -> pd.DataFrame:
    """Converts a list of LLM responses into a single flat DataFrame."""
    rows = []
    for result in all_results:
        if not result: 
            continue
        notebook_name = result.get("notebook", "UnknownNotebook")
        for ref in result.get("refs", []):
            table_name = ref.get("table", "UnknownTable")
            columns = sorted(list(set(col.lower() for col in ref.get("columns", []))))
            for col in columns:
                rows.append({
                    "Notebook Name": notebook_name,
                    "Table/Path": table_name,
                    "Column Name": col,
                })
    
    if not rows: 
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df = df.drop_duplicates().reset_index(drop=True)
    df.insert(0, 'S.No', range(1, 1 + len(df)))
    return df

# --- 8. Enhanced Streamlit UI Functions ---
def create_gradient_background():
    """Applies a custom CSS with a light orange gradient."""
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
    }
    .main-header {
        background: linear-gradient(90deg, #fdc830 0%, #f37335 100%);
        padding: 2rem; border-radius: 10px; color: white; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .metric-container {
        background: linear-gradient(135deg, #fdfcfb 0%, #e2d1c3 100%);
        padding: 1rem; border-radius: 10px; text-align: center;
        color: #333; margin: 0.5rem; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .connection-status {
        padding: 1rem; border-radius: 8px; margin: 1rem 0;
    }
    .success-status {
        background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;
    }
    .error-status {
        background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)

def create_animated_header():
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI-Powered Notebook Lineage Analyzer Agent</h1>
        <p style="font-size: 1.2em; margin-top: 1rem; opacity: 0.9;">
            Connect to your workspace and instantly map data lineage across all notebooks.
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_enhanced_sidebar():
    """Creates the new sidebar for connection inputs."""
    st.sidebar.markdown("## ⚙️ Connection Configuration")
    
    notebook_type = st.sidebar.selectbox("Notebook Type",
        ("Azure Databricks", "Azure Synapse", "Snowpark", "Microsoft Fabric"))
    
    endpoint = st.sidebar.text_input("Endpoint / Workspace URL",
        placeholder="https://adb-....azuredatabricks.net",
        help="Your Databricks workspace URL")

    auth_type = st.sidebar.selectbox("Authentication Type",
        ("Access Token", "AD Integrated"))
    
    token = ""
    tenant_id = ""
    
    if auth_type == "Access Token":
        token = st.sidebar.text_input("Access Token", type="password",
            help="Generate a personal access token from your Databricks workspace User Settings",
            value=st.secrets.get("DATABRICKS_TOKEN", ""))
    elif auth_type == "AD Integrated":
        tenant_id = st.sidebar.text_input("Azure Tenant ID (Optional)",
            placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            help="Specify your Azure AD tenant ID if authentication fails",
            value=st.secrets.get("AZURE_TENANT_ID", ""))
        
        st.sidebar.info("💡 **AD Integrated Tips:**\n- Ensure you're logged in via Azure CLI\n- Your tenant must have Azure Databricks enabled\n- Try specifying Tenant ID if auth fails")

    st.sidebar.markdown("---")
    st.sidebar.selectbox("LLM Model (UI Only)",
        ("GPT 4", "GPT 4 Turbo", "Claude 3.5 Sonnet", "Gemini Pro"), disabled=False)
    st.sidebar.info(f"Analysis will be performed by: **{DEPLOYMENT}**")
    
    return notebook_type, endpoint, auth_type, token, tenant_id

def create_enhanced_metrics(df: pd.DataFrame, total_cost: float, num_notebooks: int):
    """Displays key metrics from the analysis results."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-container"><h3>{len(df):,}</h3><p>Column References</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-container"><h3>{df["Table/Path"].nunique() if not df.empty else 0}</h3><p>Unique Tables</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-container"><h3>{num_notebooks}</h3><p>Notebooks Processed</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-container"><h3>${total_cost:.4f}</h3><p>Analysis Cost</p></div>', unsafe_allow_html=True)

def display_connection_status(is_connected: bool, endpoint: str):
    """Display connection status with appropriate styling."""
    if is_connected:
        st.markdown(f'''
        <div class="connection-status success-status">
            <strong>✅ Connection Successful</strong><br>
            Connected to: {endpoint}
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="connection-status error-status">
            <strong>❌ Connection Failed</strong><br>
            Could not connect to: {endpoint}
        </div>
        ''', unsafe_allow_html=True)

# --- 9. Main Application ---
def main():
    st.set_page_config(page_title="AI Lineage Analyzer", page_icon="🚀", layout="wide")
    create_gradient_background()
    create_animated_header()

    notebook_type, endpoint, auth_type, token, tenant_id = create_enhanced_sidebar()

    # Add a connection test button
    if st.button("🔍 Test Connection", use_container_width=True):
        if not endpoint:
            st.warning("Please provide an Endpoint / Workspace URL first.")
        else:
            with st.spinner("Testing connection..."):
                if auth_type == "Access Token":
                    if not token:
                        st.error("Please provide an Access Token for testing.")
                    else:
                        is_connected = test_databricks_connection(endpoint, token)
                        display_connection_status(is_connected, endpoint)
                elif auth_type == "AD Integrated":
                    try:
                        aad_token = _get_aad_bearer_token_with_tenant(tenant_id if tenant_id else None)
                        is_connected = test_databricks_connection(endpoint, aad_token)
                        display_connection_status(is_connected, endpoint)
                    except Exception as e:
                        st.error(f"Authentication failed: {str(e)}")
                        display_connection_status(False, endpoint)

    st.markdown("---")

    if st.button("🔗 Connect and Analyze All Notebooks", use_container_width=True):
        if not endpoint:
            st.warning("Please provide an Endpoint / Workspace URL to begin.")
            st.stop()
        
        with st.spinner("Running LangGraph agents to fetch and analyze notebooks..."):
            final_state = get_notebooks_from_source(notebook_type, endpoint, auth_type, token, tenant_id)

        if final_state and final_state.get("notebooks"):
            notebooks = final_state["notebooks"]
            lineage_results = final_state["lineage_results"]
            total_cost = final_state["total_cost"]
            error_messages = final_state["error_messages"]
            
            if error_messages:
                for error in error_messages:
                    st.warning(f"⚠️ {error}")
            
            st.success(f"Successfully processed {len(notebooks)} notebooks using LangGraph agents!")
            
            results_df = process_lineage_to_dataframe(lineage_results)

            if not results_df.empty:
                st.balloons()
                st.success(f"🎉 Analysis Complete! Processed {len(notebooks)} notebooks successfully.")
                create_enhanced_metrics(results_df, total_cost, len(notebooks))
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["📊 Full Results", "📈 Table Summary", "📋 Notebook Summary"])
                
                with tab1:
                    st.markdown("#### Complete Data Lineage Results")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                with tab2:
                    if not results_df.empty:
                        table_summary = results_df.groupby('Table/Path').agg({
                            'Column Name': 'count',
                            'Notebook Name': lambda x: ', '.join(x.unique())
                        }).reset_index()
                        table_summary.columns = ['Table/Path', 'Column Count', 'Referenced in Notebooks']
                        table_summary = table_summary.sort_values('Column Count', ascending=False).reset_index(drop=True)
                        table_summary.insert(0, 'Rank', range(1, len(table_summary) + 1))
                        st.dataframe(table_summary, use_container_width=True, hide_index=True)
                
                with tab3:
                    if not results_df.empty:
                        notebook_summary = results_df.groupby('Notebook Name').agg({
                            'Table/Path': 'nunique',
                            'Column Name': 'count'
                        }).reset_index()
                        notebook_summary.columns = ['Notebook Name', 'Unique Tables', 'Total Column References']
                        notebook_summary = notebook_summary.sort_values('Total Column References', ascending=False).reset_index(drop=True)
                        notebook_summary.insert(0, 'Rank', range(1, len(notebook_summary) + 1))
                        st.dataframe(notebook_summary, use_container_width=True, hide_index=True)
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Download Full Results (CSV)", data=csv,
                        file_name="lineage_analysis_full.csv", mime="text/csv", use_container_width=True)
                
                with col2:
                    if 'table_summary' in locals():
                        table_csv = table_summary.to_csv(index=False).encode('utf-8')
                        st.download_button(label="📥 Download Table Summary (CSV)", data=table_csv,
                            file_name="lineage_table_summary.csv", mime="text/csv", use_container_width=True)
                
            else:
                st.warning("⚠️ Analysis completed, but no data lineage references were found across all notebooks.")
                st.info("This could happen if:\n- Notebooks don't contain recognizable Spark/SQL code\n- The notebooks are empty or contain only comments\n- The LLM couldn't identify table/column references")
        else:
            st.error("No notebooks were processed by the agents. Please check your connection settings.")

if __name__ == "__main__":
    main()