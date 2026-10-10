from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from schemasense.ai_agent import (
    AIAgentError,
    AIProviderError,
    AIQuotaExceededError,
    AITimeoutError,
    SchemaSenseAIAgent,
    extract_sql_from_text,
    get_session_agent,
    load_api_keys,
    verify_schema_fidelity,
)
from schemasense.connection import (
    AccountPrivilegeReport,
    ConcurrencyLimitExceeded,
    ConnectionConfigError,
    DatabaseConnectionError,
    DriverMissingError,
    PrivilegedAccountError,
    create_safe_engine,
    dispose_engine,
    inspect_account_privileges,
)
from schemasense.profiler import (
    build_safe_profiling_query,
    profile_dataframe,
)
from schemasense.rules_engine import (
    PDFIngestionError,
    extract_rules_from_pdf,
    format_rules_for_prompt,
)
from schemasense.schema_inspector import (
    inspect_database_schema,
)
from schemasense.sql_guard import (
    QueryExecutionError,
    QueryTimeoutError,
    SQLGuard,
    SQLSecurityViolation,
    execute_safe_query,
)

# ──────────────────────────────────────────────
# CORE CONFIGURATION & PAGE METADATA
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SchemaSense — Enterprise Database Intelligence & Security Console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# ENTERPRISE SECURITY CONSOLE STYLING
# Cyan (#38BDF8) and Violet (#8B5CF6) accents, dark obsidian canvas.
# Zero emojis. Clean borders, strong typography, responsive panels.
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-canvas: #080B12;
    --bg-surface: #0E1424;
    --bg-panel: #11192C;
    --border-subtle: rgba(148, 163, 184, 0.14);
    --border-cyan: rgba(56, 189, 248, 0.3);
    --border-violet: rgba(139, 92, 246, 0.3);
    --accent-cyan: #38BDF8;
    --accent-cyan-hover: #0EA5E9;
    --accent-violet: #8B5CF6;
    --accent-emerald: #10B981;
    --accent-amber: #F59E0B;
    --accent-rose: #F43F5E;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.01em;
}

code, kbd, samp, pre, .stCodeBlock, [data-testid="stCode"] {
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stAppViewContainer"] {
    background-color: var(--bg-canvas) !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(14, 165, 233, 0.05) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.05) 0px, transparent 50%);
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] {
    background-color: #0A0F1B !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.8rem;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1440px !important;
}

/* SECURITY CONSOLE HEADER */
.console-header {
    background: linear-gradient(180deg, #10182D 0%, #0C1220 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.75rem;
    box-shadow: 0 4px 16px -4px rgba(0, 0, 0, 0.5);
}

.console-brand-group {
    display: flex;
    align-items: baseline;
    gap: 12px;
}

.console-brand-title {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 0;
    line-height: 1;
}

.console-brand-subtitle {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-weight: 500;
}

.console-badges {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.console-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.badge-connected {
    background: rgba(16, 185, 129, 0.1);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-disconnected {
    background: rgba(148, 163, 184, 0.08);
    color: #94A3B8;
    border: 1px solid rgba(148, 163, 184, 0.18);
}

.badge-cyan {
    background: rgba(14, 165, 233, 0.1);
    color: var(--accent-cyan);
    border: 1px solid var(--border-cyan);
}

.badge-violet {
    background: rgba(139, 92, 246, 0.1);
    color: var(--accent-violet);
    border: 1px solid var(--border-violet);
}

.badge-warning {
    background: rgba(245, 158, 11, 0.1);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
}
.dot-green { background: #10B981; box-shadow: 0 0 6px #10B981; }
.dot-gray { background: #64748B; }

/* OPERATIONAL SUMMARY STRIP */
.telemetry-strip {
    background: #0E1628;
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 0.65rem 1.1rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.75rem;
    font-size: 0.82rem;
    color: var(--text-secondary);
}

/* FORM CONTROLS */
div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="select"] > div {
    background-color: #0E1526 !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    border-radius: 7px !important;
    color: var(--text-primary) !important;
}

div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 1px var(--accent-cyan) !important;
}

input, textarea {
    color: var(--text-primary) !important;
}

/* BUTTONS */
button[kind="primary"] {
    background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--border-cyan) !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25) !important;
    transition: all 0.2s ease !important;
}

button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0369A1 0%, #075985 100%) !important;
    border-color: var(--accent-cyan) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(2, 132, 199, 0.4) !important;
}

button[kind="secondary"] {
    background: #0E1528 !important;
    color: #CBD5E1 !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    border-radius: 7px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}

button[kind="secondary"]:hover {
    background: #17223C !important;
    border-color: var(--border-cyan) !important;
    color: #F8FAFC !important;
    transform: translateY(-1px) !important;
}

/* STREAMLIT TABS */
div[data-testid="stTabs"] button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.65rem 1.15rem !important;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom-color: var(--accent-cyan) !important;
    background: rgba(56, 189, 248, 0.04) !important;
}

div[data-testid="stTabs"] button:hover {
    color: var(--text-primary) !important;
}

/* EXPANDERS */
[data-testid="stExpander"] {
    background: #0D1424 !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    margin-bottom: 0.6rem !important;
}

/* CHAT MESSAGES */
[data-testid="stChatMessage"] {
    background-color: #0C1322 !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    padding: 0.85rem 1.1rem !important;
    margin-bottom: 0.65rem !important;
}

/* DATAFRAMES */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# GEMINI MODEL CLIENT & SECURE CREDENTIALS (google-genai)
# ──────────────────────────────────────────────
DEFAULT_API_KEYS = load_api_keys()
if "active_key_index" not in st.session_state:
    st.session_state.active_key_index = 0


def _get_agent(
    model_name: str,
    backend_type: str | None = None,
    model_path: str | Path | None = None,
) -> SchemaSenseAIAgent:
    """Retrieve or instantiate a SchemaSenseAIAgent for the active backend.

    Reuses existing agent instances across reruns within the user session when
    configuration is unchanged, avoiding redundant memory mapping of local model weights.
    Fails closed without cloud fallback when local inference is selected.
    """
    return get_session_agent(
        model_name=model_name,
        backend_type=backend_type,
        model_path=model_path,
        default_api_keys=DEFAULT_API_KEYS,
        cache_store=st.session_state,
        timeout_seconds=20.0,
    )


# Backward compatibility aliases
extract_sql_from_response = extract_sql_from_text


def generate_with_fallback(
    model_name: str, system_prompt: str, messages_or_prompt, stream: bool = False
):
    """Compatibility bridge routing legacy callers to the modernized agent."""
    agent = _get_agent(model_name)
    if isinstance(messages_or_prompt, str):
        prompt = messages_or_prompt
        history = None
    else:
        history, prompt = messages_or_prompt
    if stream:
        return agent.generate_stream(
            prompt=prompt, history=history
        ), agent.active_key_index
    resp = agent.generate_structured(prompt=prompt)
    return resp, agent.active_key_index


# ──────────────────────────────────────────────
# APPLICATION STATE INITIALIZATION
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome! Connect a PostgreSQL or MySQL database in the sidebar to inspect live schemas, or ask any SQL and database architecture questions.",
        }
    ]
if "db_engine" not in st.session_state:
    st.session_state.db_engine = None
if "db_schema" not in st.session_state:
    st.session_state.db_schema = None
if "db_type" not in st.session_state:
    st.session_state.db_type = ""
if "last_sql" not in st.session_state:
    st.session_state.last_sql = None
if "business_rules" not in st.session_state:
    st.session_state.business_rules = []
if "privilege_report" not in st.session_state:
    st.session_state.privilege_report = None
if "selected_tables" not in st.session_state:
    st.session_state.selected_tables = []

# ──────────────────────────────────────────────
# SIDEBAR: CONNECTION GATEWAY & SETTINGS
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.2rem 0 0.7rem 0; border-bottom: 1px solid rgba(148, 163, 184, 0.12); margin-bottom: 0.8rem;">
            <div style="font-size:1.2rem; font-weight:800; letter-spacing:-0.02em; color:#F8FAFC;">
                SchemaSense
            </div>
            <div style="font-size:0.75rem; color:#64748B;">
                Database Intelligence & Security Console
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style=\"font-family:'JetBrains Mono',monospace; font-size:0.68rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.1rem;\">Inference Backend</div>",
        unsafe_allow_html=True,
    )
    backend_mode_choice = st.selectbox(
        "Backend Selection",
        [
            "Google Gemini (Cloud)",
            "llama.cpp (Local GGUF)",
        ],
        key="backend_mode_select",
        label_visibility="collapsed",
    )
    is_local_backend = "Local" in backend_mode_choice

    if not is_local_backend:
        st.markdown(
            '<div style="font-size:0.75rem; color:#94A3B8; margin-bottom:0.4rem;">'
            "Provider: <strong style='color:#38BDF8;'>Google Gemini API</strong> (Cloud inference)"
            "</div>",
            unsafe_allow_html=True,
        )
        selected_model = st.selectbox(
            "Model Selection",
            [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash",
                "gemini-3-flash-preview",
            ],
            key="gemini_model_select",
            label_visibility="collapsed",
        )
        custom_key = st.text_input(
            "Gemini API Key (Optional)",
            type="password",
            placeholder="Custom API Key (or use environment)",
            help="Optional API key override. Credentials are kept in memory and never logged.",
            key="gemini_custom_key_input",
        )
        if custom_key.strip():
            st.session_state.custom_api_key = custom_key.strip()
    else:
        st.markdown(
            '<div style="font-size:0.75rem; color:#94A3B8; margin-bottom:0.4rem;">'
            "Provider: <strong style='color:#10B981;'>llama.cpp</strong> (In-process offline inference)"
            "</div>",
            unsafe_allow_html=True,
        )
        local_path_val = st.text_input(
            "Local GGUF Model Path",
            value=st.session_state.get("local_model_path", ""),
            placeholder="e.g. C:/models/model.gguf or /models/model.gguf",
            help="Local filesystem path to a compatible .gguf model file. No automatic downloads or directory scans.",
            key="local_model_path_input",
        )
        if local_path_val.strip():
            st.session_state.local_model_path = local_path_val.strip()
        else:
            st.session_state.local_model_path = ""

        selected_model = (
            Path(local_path_val.strip()).name
            if local_path_val.strip()
            else "local-gguf"
        )

        from schemasense.runtime.backends.llama_cpp import (
            LLAMA_CPP_AVAILABLE,
            LlamaCppBackend,
        )

        if not LLAMA_CPP_AVAILABLE:
            st.caption(
                "Engine: `llama-cpp-python` not installed. Install via: `pip install -r requirements-local.txt`"
            )
        elif not local_path_val.strip():
            st.caption("Engine: Please specify a local `.gguf` file path above.")
        else:
            check_backend = LlamaCppBackend(model_path=local_path_val.strip())
            if not check_backend.is_available():
                st.caption("Status: File not found or not a valid `.gguf` model file.")
            else:
                cached_agent = st.session_state.get("_active_agent")
                if (
                    cached_agent is not None
                    and hasattr(cached_agent.backend, "_llm")
                    and cached_agent.backend._llm is not None
                    and getattr(cached_agent, "model_path", None)
                    == Path(local_path_val.strip())
                ):
                    st.caption("Status: Model loaded in memory (ready for queries).")
                else:
                    st.caption(
                        "Status: Model verified on disk (weights load on first query)."
                    )

        st.caption(
            "Privacy Boundary: Local model inference executes strictly in-process on this machine. Live database queries continue to route over the network to your configured database host."
        )

    st.divider()

    st.markdown(
        "<div style=\"font-family:'JetBrains Mono',monospace; font-size:0.68rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.1rem;\">Database Connection</div>",
        unsafe_allow_html=True,
    )
    db_engine_choice = st.selectbox(
        "Database Engine",
        ["PostgreSQL", "MySQL"],
        label_visibility="collapsed",
    )

    with st.form("db_connect_form"):
        db_host = st.text_input("Host URL / IP", placeholder="e.g. 127.0.0.1")
        db_port = st.text_input(
            "Port", value="5432" if db_engine_choice == "PostgreSQL" else "3306"
        )
        db_name = st.text_input("Database Name", placeholder="database_name")
        db_user = st.text_input("Username", placeholder="username")
        db_pass = st.text_input("Password", type="password")
        enforce_least_priv = st.checkbox(
            "Enforce Least-Privilege Role",
            value=False,
            help="Enforces that connecting credentials belong to a read-only role. Rejects superusers or administrative accounts.",
        )

        connect_btn = st.form_submit_button("Secure Connect", use_container_width=True)

    if connect_btn:
        if not all([db_host, db_port, db_name, db_user, db_pass]):
            st.error("All connection parameters are mandatory.")
        else:
            with st.spinner("Connecting to database and inspecting schema..."):
                try:
                    if st.session_state.db_engine is not None:
                        dispose_engine(st.session_state.db_engine)
                        st.session_state.db_engine = None

                    engine = create_safe_engine(
                        db_type=db_engine_choice,
                        host=db_host,
                        port=db_port,
                        db_name=db_name,
                        user=db_user,
                        password=db_pass,
                        enforce_least_privilege=enforce_least_priv,
                    )
                    schema_obj = inspect_database_schema(engine)
                    report = getattr(
                        engine, "_schemasense_privilege_report", None
                    ) or inspect_account_privileges(engine, db_type=db_engine_choice)

                    st.session_state.db_engine = engine
                    st.session_state.db_schema = schema_obj
                    st.session_state.db_type = db_engine_choice
                    st.session_state.privilege_report = report

                    table_count = len(schema_obj.tables)
                    if report.is_privileged:
                        st.warning(f"Security Advisory: {report.summary}")
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": (
                                    f"Security Advisory: Connected to **{db_name}** with a privileged account "
                                    f"(`{report.current_user}`). {report.summary}. For production deployment, provision a restricted read-only account."
                                ),
                            }
                        )
                    else:
                        st.success(
                            f"Connected: Discovered {table_count} tables with least-privilege role (`{report.current_user}`)."
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": (
                                    f"Connected to **{db_name}** ({db_engine_choice}) with verified least-privilege account "
                                    f"(`{report.current_user}`). Discovered {table_count} tables. AST Guard active for all queries."
                                ),
                            }
                        )

                except PrivilegedAccountError as pae:
                    st.error(f"Security Policy Violation: {pae}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                f"Connection Blocked: Administrative or superuser accounts are prohibited when least-privilege "
                                f"enforcement is active. {pae}. Please configure and use a read-only service account."
                            ),
                        }
                    )
                except (ConnectionConfigError, DriverMissingError) as ce:
                    st.error(f"Configuration / Driver Error: {ce}")
                except DatabaseConnectionError as dce:
                    st.error(f"Connection Failed: {dce}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Unexpected Connection Fault: {e}")

    if st.session_state.get("db_engine") is not None and st.button(
        "Disconnect Database", use_container_width=True
    ):
        dispose_engine(st.session_state.db_engine)
        st.session_state.db_engine = None
        st.session_state.db_schema = None
        st.session_state.privilege_report = None
        st.session_state.last_sql = None
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Disconnect Database completed. Connection pool drained and safely disposed.",
            }
        )
        st.rerun()

    st.divider()

    st.markdown(
        "<div style=\"font-family:'JetBrains Mono',monospace; font-size:0.68rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.1rem;\">Enterprise Knowledge Base</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Business Rules Ingestion (PDF)"):
        st.caption(
            "Upload business policies and domain definitions to ground SQL generation."
        )
        uploaded_pdf = st.file_uploader(
            "Upload Specification", type=["pdf"], label_visibility="collapsed"
        )
        if uploaded_pdf is not None:
            try:
                extracted_rules = extract_rules_from_pdf(uploaded_pdf.read())
                st.session_state.business_rules = extracted_rules
                st.success(
                    f"Ingested {len(extracted_rules)} business constraints from {uploaded_pdf.name}."
                )
            except PDFIngestionError as pie:
                st.error(f"Ingestion Error: {pie}")

    if st.session_state.db_schema:
        st.divider()
        st.markdown(
            "<div style=\"font-family:'JetBrains Mono',monospace; font-size:0.68rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.1rem;\">Schema Privacy Isolation</div>",
            unsafe_allow_html=True,
        )
        all_tables = list(st.session_state.db_schema.tables.keys())
        selected_tables = st.multiselect(
            "Exposed Schema Tables",
            options=all_tables,
            default=all_tables,
            help="Select which tables are transmitted to the external AI model.",
            label_visibility="collapsed",
        )
        st.session_state.selected_tables = selected_tables

    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Welcome! Conversation history cleared. Ready for database intelligence queries.",
            }
        ]
        st.session_state.last_sql = None
        st.rerun()

# ──────────────────────────────────────────────
# SYSTEM PROMPT ARCHITECTURE
# ──────────────────────────────────────────────
schema_prompt_section = (
    st.session_state.db_schema.to_prompt_context(
        table_filter=st.session_state.get("selected_tables")
    )
    if st.session_state.db_schema
    else "No database connected yet. Answer general database and SQL questions accurately, and indicate that a database connection is required for live schema queries."
)
rules_prompt_section = (
    format_rules_for_prompt(st.session_state.business_rules)
    if st.session_state.business_rules
    else ""
)

steward_persona = f"""
You are SchemaSense, a Principal Enterprise Data Architect and Analytics Agent.
Active Dialect: {st.session_state.db_type or "None"}

{schema_prompt_section}

{rules_prompt_section}

Mandatory Operational Guardrails:
1. SCHEMA ACCURACY: When a database is connected, reference only tables, columns, and relations present in the schema context above.
2. READ-ONLY INSTRUCTION: Strictly generate SELECT queries. Never emit DROP, DELETE, INSERT, UPDATE, or ALTER statements.
3. DOMAIN COMPLIANCE: Honor any enterprise business rules provided above.
4. CODE FORMATTING: Wrap all runnable SQL queries inside ```sql ... ``` fences.
"""

# ──────────────────────────────────────────────
# TOP STATUS & TELEMETRY BAR
# All badges derived from real runtime state.
# ──────────────────────────────────────────────
is_connected = st.session_state.db_engine is not None
priv_report: AccountPrivilegeReport | None = st.session_state.privilege_report

if is_connected:
    db_name_val = (
        st.session_state.db_schema.database_name
        if st.session_state.db_schema
        else "Connected"
    )
    conn_badge_html = f'<div class="console-badge badge-connected"><span class="status-dot dot-green"></span><span>Connected: {db_name_val} ({st.session_state.db_type})</span></div>'
    if priv_report and priv_report.is_privileged:
        role_badge_html = f'<div class="console-badge badge-warning"><span>Role: Privileged ({priv_report.current_user})</span></div>'
    elif priv_report:
        role_badge_html = f'<div class="console-badge badge-connected"><span>Role: Least-Privilege ({priv_report.current_user})</span></div>'
    else:
        role_badge_html = '<div class="console-badge badge-disconnected"><span>Role: Unknown</span></div>'
else:
    conn_badge_html = '<div class="console-badge badge-disconnected"><span class="status-dot dot-gray"></span><span>Disconnected</span></div>'
    role_badge_html = '<div class="console-badge badge-disconnected"><span>Role: Not checked</span></div>'

ast_badge_html = '<div class="console-badge badge-cyan"><span>AST Guard: Active (sqlglot)</span></div>'
pool_badge_html = '<div class="console-badge badge-violet"><span>Pool Limit: 4 Concurrent</span></div>'

st.markdown(
    f"""
    <div class="console-header">
        <div class="console-brand-group">
            <h1 class="console-brand-title">SchemaSense</h1>
            <span class="console-brand-subtitle">Enterprise Database Intelligence & Security Console</span>
        </div>
        <div class="console-badges">
            {conn_badge_html}
            {role_badge_html}
            {ast_badge_html}
            {pool_badge_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# MAIN NAVIGATION TABS
# ──────────────────────────────────────────────
tab_chat, tab_schema, tab_profiler, tab_sql = st.tabs(
    [
        "AI Agent Studio",
        "Schema Architecture",
        "Column Profiler & Health",
        "SQL Guard Sandbox",
    ]
)

# ═══════════════════════════════════════════════
# TAB 1: AI AGENT STUDIO
# ═══════════════════════════════════════════════
with tab_chat:
    active_mode_select = st.session_state.get(
        "backend_mode_select", "Google Gemini (Cloud)"
    )
    is_local_active = "Local" in active_mode_select
    engine_badge = (
        f"Local (llama.cpp - {selected_model})"
        if is_local_active
        else f"Cloud (Gemini - {selected_model})"
    )

    if is_connected and st.session_state.db_schema:
        active_tables_count = len(
            st.session_state.get("selected_tables") or st.session_state.db_schema.tables
        )
        active_rules_count = len(st.session_state.business_rules)
        st.markdown(
            f"""
            <div class="telemetry-strip">
                <span>Database: <strong>{st.session_state.db_schema.database_name}</strong> ({st.session_state.db_type})</span>
                <span>Active Schema Context: <strong>{active_tables_count} tables</strong></span>
                <span>Business Constraints: <strong>{active_rules_count} rules active</strong></span>
                <span>Inference: <strong>{engine_badge}</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="telemetry-strip">
                <span>Status: <strong>Database disconnected</strong></span>
                <span>Operating Mode: <strong>General SQL guidance</strong></span>
                <span>Inference: <strong>{engine_badge}</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_mode, col_clear = st.columns([3, 1])
    with col_mode:
        ai_mode = st.radio(
            "Response Mode",
            ["Streaming Dialogue", "Structured Schema Reasoning (Pydantic)"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with col_clear:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Welcome! Connect a PostgreSQL or MySQL database in the sidebar to inspect live schemas, or ask any SQL and database architecture questions.",
                }
            ]
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(
        "Ask about database schemas, join relationships, or SQL queries..."
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                agent = _get_agent(selected_model)
                allowed_tables = (
                    set(
                        st.session_state.get("selected_tables")
                        or st.session_state.db_schema.tables.keys()
                    )
                    if (is_connected and st.session_state.db_schema)
                    else set()
                )

                if ai_mode == "Structured Schema Reasoning (Pydantic)":
                    with st.spinner("Generating validated structured response..."):
                        structured_resp = agent.generate_structured(
                            prompt=prompt,
                            schema_context=schema_prompt_section,
                            business_rules_context=rules_prompt_section,
                            dialect=st.session_state.db_type or "postgres",
                            allowed_tables=allowed_tables,
                        )
                        full_response = structured_resp.explanation
                        if structured_resp.sql_query:
                            full_response += (
                                f"\n\n```sql\n{structured_resp.sql_query}\n```"
                            )
                        if structured_resp.safety_notes:
                            full_response += f"\n\n> **Safety Notes**: {structured_resp.safety_notes}"
                        response_placeholder.markdown(full_response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": full_response}
                        )

                        if structured_resp.sql_query:
                            st.session_state.last_sql = structured_resp.sql_query
                            if structured_resp.is_executable:
                                st.info(
                                    "Structured SQL query verified against schema and loaded into SQL Guard Sandbox."
                                )
                            else:
                                st.warning(
                                    "Query generated with safety restrictions or hallucination warning. Inspect in SQL Guard Sandbox."
                                )
                else:
                    for chunk in agent.generate_stream(
                        prompt=prompt,
                        schema_context=schema_prompt_section,
                        business_rules_context=rules_prompt_section,
                        dialect=st.session_state.db_type or "postgres",
                        history=st.session_state.messages[:-1],
                    ):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "...")

                    response_placeholder.markdown(full_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )

                    sql = extract_sql_from_text(full_response)
                    if sql:
                        st.session_state.last_sql = sql
                        if allowed_tables:
                            from schemasense.sql_guard import SQLGuard

                            try:
                                ast_tables = SQLGuard(
                                    allowed_tables=allowed_tables
                                )._extract_all_table_references(sql)
                                hallucinations = verify_schema_fidelity(
                                    list(ast_tables), allowed_tables
                                )
                                if hallucinations:
                                    st.warning(
                                        f"Schema Grounding Alert: Query references tables not present in active schema: {', '.join(hallucinations)}."
                                    )
                                else:
                                    st.info(
                                        "SQL query verified against active schema and loaded into the SQL Guard Sandbox."
                                    )
                            except Exception:  # noqa: BLE001
                                st.info(
                                    "SQL query detected and loaded into the SQL Guard Sandbox for inspection and execution."
                                )
                        else:
                            st.info(
                                "SQL query detected and loaded into the SQL Guard Sandbox for inspection and execution."
                            )

            except (
                AIQuotaExceededError,
                AITimeoutError,
                AIProviderError,
                AIAgentError,
            ) as ai_err:
                st.error(f"AI Provider Alert: {ai_err}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Agent Execution Fault: {e}")

# ═══════════════════════════════════════════════
# TAB 2: SCHEMA ARCHITECTURE & DATA DICTIONARY
# ═══════════════════════════════════════════════
with tab_schema:
    if not is_connected or st.session_state.db_schema is None:
        st.info(
            "No active database connection. Connect a PostgreSQL or MySQL database in the sidebar to inspect tables, relations, and data dictionary."
        )
    else:
        schema = st.session_state.db_schema
        total_tables = len(schema.tables)
        total_cols = sum(len(t.columns) for t in schema.tables.values())
        total_fks = sum(len(t.foreign_keys) for t in schema.tables.values())
        total_indexes = sum(len(t.indexes) for t in schema.tables.values())

        st.markdown(
            f"""
            <div class="telemetry-strip">
                <span>Database: <strong>{schema.database_name}</strong></span>
                <span>Tables: <strong>{total_tables}</strong></span>
                <span>Total Columns: <strong>{total_cols}</strong></span>
                <span>Foreign Keys: <strong>{total_fks}</strong></span>
                <span>Indexes: <strong>{total_indexes}</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                label="Download Data Dictionary (JSON)",
                data=schema.to_json(),
                file_name=f"{schema.database_name}_data_dictionary.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_exp2:
            st.download_button(
                label="Download Data Dictionary (Markdown)",
                data=schema.to_markdown(),
                file_name=f"{schema.database_name}_data_dictionary.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        for table_name, table_info in schema.tables.items():
            pk_list = (
                ", ".join(table_info.primary_key) if table_info.primary_key else "None"
            )
            with st.expander(
                f"{table_name}  [{len(table_info.columns)} columns | PK: {pk_list}]"
            ):
                col_c1, col_c2 = st.columns([1, 1])
                with col_c1:
                    st.markdown(
                        "<div style='font-family:\"JetBrains Mono\",monospace; font-size:0.75rem; font-weight:700; color:#38BDF8; text-transform:uppercase;'>Column Definitions</div>",
                        unsafe_allow_html=True,
                    )
                    col_records = []
                    for c in table_info.columns:
                        col_records.append(
                            {
                                "Column": c.name,
                                "Type": c.data_type,
                                "Primary Key": "True" if c.is_primary_key else "False",
                                "Nullable": "True" if c.nullable else "False",
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(col_records),
                        use_container_width=True,
                        hide_index=True,
                    )

                with col_c2:
                    st.markdown(
                        "<div style='font-family:\"JetBrains Mono\",monospace; font-size:0.75rem; font-weight:700; color:#8B5CF6; text-transform:uppercase;'>Foreign Keys & Indexes</div>",
                        unsafe_allow_html=True,
                    )
                    if table_info.foreign_keys:
                        fk_records = [
                            {
                                "Local Column": fk.constrained_column,
                                "Foreign Table": fk.referred_table,
                                "Target Column": fk.referred_column,
                            }
                            for fk in table_info.foreign_keys
                        ]
                        st.dataframe(
                            pd.DataFrame(fk_records),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.caption(
                            "No foreign key constraints defined for this entity."
                        )

                    if table_info.indexes:
                        st.markdown(
                            "<div style='font-family:\"JetBrains Mono\",monospace; font-size:0.7rem; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-top:8px;'>Indexes</div>",
                            unsafe_allow_html=True,
                        )
                        idx_records = [
                            {
                                "Index Name": idx.name,
                                "Columns": ", ".join(idx.columns),
                                "Unique": "True" if idx.unique else "False",
                            }
                            for idx in table_info.indexes
                        ]
                        st.dataframe(
                            pd.DataFrame(idx_records),
                            use_container_width=True,
                            hide_index=True,
                        )

# ═══════════════════════════════════════════════
# TAB 3: COLUMN PROFILER & HEALTH
# ═══════════════════════════════════════════════
with tab_profiler:
    if not is_connected or st.session_state.db_schema is None:
        st.info(
            "No active database connection. Connect a database in the sidebar to select tables and compute column statistics."
        )
    else:
        tables = list(st.session_state.db_schema.tables.keys())
        selected_table = st.selectbox("Select Target Table for Health Audit", tables)

        if st.button("Audit Table Health & Column Quality", type="primary"):
            with st.spinner(f"Computing column statistics for {selected_table}..."):
                try:
                    profile_query = build_safe_profiling_query(
                        selected_table,
                        dialect=st.session_state.db_type,
                        sample_size=1000,
                    )
                    allowed_tbls = set(st.session_state.db_schema.tables.keys())

                    sample_df, _ = execute_safe_query(
                        engine=st.session_state.db_engine,
                        query=profile_query,
                        allowed_tables=allowed_tbls,
                        dialect=st.session_state.db_type,
                        timeout_ms=5000,
                        max_rows=1000,
                    )

                    report = profile_dataframe(sample_df, selected_table)

                    st.markdown(
                        f"""
                        <div class="telemetry-strip">
                            <span>Table: <strong>{selected_table}</strong></span>
                            <span>Rows Audited: <strong>{report.row_count:,}</strong></span>
                            <span>Quality Score: <strong>{report.overall_quality_score}%</strong></span>
                            <span>Duplicate Rows: <strong>{report.duplicate_rows_count} ({report.duplicate_rows_percentage}%)</strong></span>
                            <span>Candidate Keys: <strong>{len([c for c in report.columns if c.is_candidate_key])}</strong></span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        "<div style='font-family:\"JetBrains Mono\",monospace; font-size:0.75rem; font-weight:700; color:#38BDF8; text-transform:uppercase; margin-top:0.5rem;'>Attribute Health Breakdown</div>",
                        unsafe_allow_html=True,
                    )
                    summary_records = []
                    for c in report.columns:
                        summary_records.append(
                            {
                                "Column": c.name,
                                "Type": c.data_type,
                                "Null %": f"{c.null_percentage}%",
                                "Distinct Count": c.distinct_count,
                                "Candidate Key": "True"
                                if c.is_candidate_key
                                else "False",
                                "Outliers": max(0, c.outlier_count),
                                "Min Value": str(c.min_value)
                                if c.min_value is not None
                                else "-",
                                "Max Value": str(c.max_value)
                                if c.max_value is not None
                                else "-",
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(summary_records),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Plotly Visualizations with restrained Cyan and Violet theme
                    numeric_cols = sample_df.select_dtypes(
                        include="number"
                    ).columns.tolist()
                    categorical_cols = sample_df.select_dtypes(
                        include=["object", "category", "bool"]
                    ).columns.tolist()

                    if numeric_cols or categorical_cols:
                        st.markdown(
                            "<div style='font-family:\"JetBrains Mono\",monospace; font-size:0.75rem; font-weight:700; color:#8B5CF6; text-transform:uppercase; margin-top:1.25rem;'>Attribute Distributions</div>",
                            unsafe_allow_html=True,
                        )
                        dist_tab_num, dist_tab_cat = st.tabs(
                            ["Numerical Features", "Categorical Features"]
                        )

                        with dist_tab_num:
                            if numeric_cols:
                                for col in numeric_cols[:4]:
                                    fig = px.histogram(
                                        sample_df,
                                        x=col,
                                        title=f"Distribution: {col}",
                                        template="plotly_dark",
                                        color_discrete_sequence=["#38BDF8"],
                                    )
                                    fig.update_layout(
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(14, 20, 36, 0.6)",
                                        font={
                                            "family": "Plus Jakarta Sans, sans-serif",
                                            "color": "#94A3B8",
                                        },
                                        margin={"l": 20, "r": 20, "t": 40, "b": 20},
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.caption(
                                    "No numerical attributes discovered in this table."
                                )

                        with dist_tab_cat:
                            if categorical_cols:
                                for col in categorical_cols[:4]:
                                    counts = sample_df[col].value_counts().head(8)
                                    fig = px.bar(
                                        x=counts.index.astype(str),
                                        y=counts.values,
                                        title=f"Frequency Breakdown: {col}",
                                        template="plotly_dark",
                                        color_discrete_sequence=["#8B5CF6"],
                                    )
                                    fig.update_layout(
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(14, 20, 36, 0.6)",
                                        font={
                                            "family": "Plus Jakarta Sans, sans-serif",
                                            "color": "#94A3B8",
                                        },
                                        margin={"l": 20, "r": 20, "t": 40, "b": 20},
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.caption(
                                    "No categorical attributes discovered in this table."
                                )

                except ConcurrencyLimitExceeded as cle:
                    st.warning(f"Server Concurrency Bound Exceeded: {cle}")
                except (SQLSecurityViolation, QueryExecutionError) as err:
                    st.error(f"Profiling Error: {err}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Unexpected Profiling Fault: {e}")

# ═══════════════════════════════════════════════
# TAB 4: SQL GUARD SANDBOX & VISUALIZER
# ═══════════════════════════════════════════════
with tab_sql:
    if is_connected:
        st.markdown(
            f"""
            <div class="telemetry-strip">
                <span>Active Dialect: <strong>{st.session_state.db_type}</strong></span>
                <span>AST Validator: <strong>Strict SELECT Enforcing</strong></span>
                <span>Timeout Ceiling: <strong>5,000 ms</strong></span>
                <span>Max Rows: <strong>1,000</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="telemetry-strip">
                <span>AST Sandbox: <strong>Offline Validation Mode</strong></span>
                <span>Validation Rule: <strong>Read-Only Enforced</strong></span>
                <span>Connect via sidebar to execute against live data</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    default_sql = st.session_state.last_sql or ""
    sql_input = st.text_area(
        "SQL Query Buffer",
        value=default_sql,
        height=140,
        placeholder="SELECT * FROM table_name LIMIT 50;",
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_btn = st.button(
            "Execute Guarded Query", type="primary", use_container_width=True
        )
    with col_btn2:
        if st.button("Clear Buffer", use_container_width=True):
            st.session_state.last_sql = None
            st.rerun()

    if run_btn and sql_input.strip():
        if not is_connected:
            # Standalone AST validation sandbox
            try:
                validation = SQLGuard.validate(
                    query=sql_input.strip(),
                    dialect=st.session_state.db_type or "postgres",
                )
                if validation.is_valid:
                    st.success(
                        "AST Validation Passed: Statement classified as strictly read-only. "
                        "Connect a live database in the sidebar to execute against real tables."
                    )
                else:
                    st.error(f"Security Block (SQLGuard): {validation.error_message}")
            except Exception as e:  # noqa: BLE001
                st.error(f"AST Validation Error: {e}")
        else:
            allowed_tables = set(
                st.session_state.get("selected_tables")
                or st.session_state.db_schema.tables.keys()
            )
            try:
                with st.spinner("Validating AST and executing query..."):
                    result_df, is_truncated = execute_safe_query(
                        engine=st.session_state.db_engine,
                        query=sql_input.strip(),
                        allowed_tables=allowed_tables,
                        dialect=st.session_state.db_type,
                        timeout_ms=5000,
                        max_rows=1000,
                    )

                if is_truncated:
                    st.warning(
                        "Result exceeded memory limit (1,000 rows) and was truncated."
                    )

                st.success(f"Query executed: {len(result_df):,} rows retrieved.")
                st.dataframe(result_df, use_container_width=True)

                if not result_df.empty:
                    num_cols = result_df.select_dtypes(
                        include="number"
                    ).columns.tolist()
                    cat_cols = result_df.select_dtypes(
                        exclude="number"
                    ).columns.tolist()

                    if num_cols and cat_cols:
                        st.markdown(
                            "<div style='font-family:\"JetBrains Mono\",monospace; font-size:0.75rem; font-weight:700; color:#38BDF8; text-transform:uppercase; margin-top:1.25rem;'>Automated Visualization</div>",
                            unsafe_allow_html=True,
                        )
                        fig = px.bar(
                            result_df.head(30),
                            x=cat_cols[0],
                            y=num_cols[0],
                            template="plotly_dark",
                            color_discrete_sequence=["#38BDF8"],
                        )
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(14, 20, 36, 0.6)",
                            font={
                                "family": "Plus Jakarta Sans, sans-serif",
                                "color": "#94A3B8",
                            },
                            margin={"l": 20, "r": 20, "t": 40, "b": 20},
                        )
                        st.plotly_chart(fig, use_container_width=True)

            except ConcurrencyLimitExceeded as cle:
                st.warning(f"Server Busy: {cle}")
            except SQLSecurityViolation as sec_err:
                st.error(f"Security Block (SQLGuard): {sec_err}")
            except QueryTimeoutError as time_err:
                st.error(f"Statement Timeout: {time_err}")
            except QueryExecutionError as exec_err:
                st.error(f"SQL Execution Error: {exec_err}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Execution Fault: {e}")
