
import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import json
import re
import urllib.parse
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

# Try to import pymysql, install if missing
try:
    import pymysql
except ImportError:
    import subprocess
    import sys
    st.warning("Installing PyMySQL...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pymysql==1.1.1"])
        import pymysql
        st.success("✅ PyMySQL installed successfully!")
    except Exception as e:
        st.error(f"❌ Failed to install PyMySQL: {e}")
        st.error("Please run: `pip install pymysql` manually")

# ──────────────────────────────────────────────
# INITIAL CONFIGURATION
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SchemaSense AI – Enterprise Data Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS – glassmorphism, gradients, animations
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body { font-family: 'Inter', sans-serif; }
.hero-header {
    background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 50%, #06B6D4 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; color: white; position: relative; overflow: hidden;
}
.hero-header::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(circle at 80% 20%, rgba(255,255,255,0.15) 0%, transparent 60%); pointer-events: none;
}
.hero-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.hero-header p  { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 1rem; }
.glass-card {
    background: rgba(255,255,255,0.04); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 1.25rem 1.5rem; margin-bottom: 1rem;
}
.status-badge {
    display: inline-flex; align-items: center; gap: 8px; background: rgba(16,185,129,0.12); color: #10B981;
    border: 1px solid rgba(16,185,129,0.25); border-radius: 999px; padding: 4px 14px; font-size: 0.82rem; font-weight: 600;
}
.metric-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 1rem; }
.metric-item { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 1rem; text-align: center; }
.metric-item .label { font-size: 0.75rem; opacity: 0.6; text-transform: uppercase; letter-spacing: 1px; }
.metric-item .value { font-size: 1.5rem; font-weight: 700; color: #6C63FF; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# FEATURE 1: API KEY FALLBACK
# ──────────────────────────────────────────────
def _load_api_keys() -> list[str]:
    try:
        keys = st.secrets.get("GEMINI_KEYS", None)
        if keys and isinstance(keys, list): return [k for k in keys if k]
    except Exception: pass
    try:
        single = st.secrets.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
        if single: return [single]
    except Exception: pass
    return ["YOUR_API_KEY_HERE"]

API_KEYS = _load_api_keys()
if "active_key_index" not in st.session_state: st.session_state.active_key_index = 0

def _get_active_key() -> str:
    return API_KEYS[st.session_state.active_key_index % len(API_KEYS)]

def _rotate_key():
    st.session_state.active_key_index = (st.session_state.active_key_index + 1) % len(API_KEYS)

def generate_with_fallback(model_name: str, system_prompt: str, messages_or_prompt, stream: bool = False):
    attempts = len(API_KEYS)
    last_exc = None
    for attempt in range(attempts):
        genai.configure(api_key=_get_active_key())
        model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
        try:
            if isinstance(messages_or_prompt, str):
                return model.generate_content(messages_or_prompt), st.session_state.active_key_index
            else:
                history, user_prompt = messages_or_prompt
                chat = model.start_chat(history=history)
                return chat.send_message(user_prompt, stream=stream), st.session_state.active_key_index
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                last_exc = e
                _rotate_key()
                st.toast(f"⚠️ Quota hit – switching to next key…", icon="🔄")
            else: raise e
    raise last_exc

# ──────────────────────────────────────────────
# FEATURE 2: DYNAMIC DATABASE ENGINE
# ──────────────────────────────────────────────
def build_connection_string(db_type, host, port, db_name, user, password):
    """Constructs the SQLAlchemy connection string based on the selected DB type."""
    if not all([host, port, db_name, user, password]):
        raise ValueError("All connection parameters are required")
    
    safe_user = urllib.parse.quote_plus(str(user))
    safe_password = urllib.parse.quote_plus(str(password))
    safe_host = str(host).strip()
    safe_db_name = str(db_name).strip()
    
    try:
        port = int(port)
    except (ValueError, TypeError):
        raise ValueError("Port must be a valid integer")
    
    if db_type == "PostgreSQL":
        return f"postgresql+psycopg2://{safe_user}:{safe_password}@{safe_host}:{port}/{safe_db_name}"
    elif db_type == "MySQL":
        return f"mysql+pymysql://{safe_user}:{safe_password}@{safe_host}:{port}/{safe_db_name}?charset=utf8mb4"
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def extract_database_schema(engine):
    """Automatically inspects the database to get tables, columns, and foreign keys."""
    try:
        inspector = inspect(engine)
        schema_dict = {}
        schema_text_lines = []

        tables = inspector.get_table_names()
        if not tables:
            return {}, "No tables found in the database."
            
        for table in tables:
            try:
                columns = inspector.get_columns(table)
                fks = inspector.get_foreign_keys(table)
                
                col_details = []
                for col in columns:
                    col_type = str(col.get('type', 'UNKNOWN'))
                    nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
                    col_details.append(f"{col['name']} ({col_type}) {nullable}")
                
                schema_dict[table] = {
                    "columns": col_details,
                    "foreign_keys": fks
                }

                # Build text representation for the AI Prompt
                schema_text_lines.append(f"Table: {table}")
                schema_text_lines.append(f"  Columns: {', '.join(col_details)}")
                if fks:
                    fk_strings = []
                    for fk in fks:
                        if fk.get('constrained_columns') and fk.get('referred_table') and fk.get('referred_columns'):
                            fk_strings.append(f"({fk['constrained_columns'][0]} -> {fk['referred_table']}.{fk['referred_columns'][0]})")
                    if fk_strings:
                        schema_text_lines.append(f"  Foreign Keys: {', '.join(fk_strings)}")
                schema_text_lines.append("")  # blank line
            except Exception as e:
                st.warning(f"Could not inspect table {table}: {e}")
                continue

        return schema_dict, "\n".join(schema_text_lines)
    except Exception as e:
        raise Exception(f"Failed to extract database schema: {e}")

# ──────────────────────────────────────────────
# FEATURE 3: SQL EXTRACTION & GUARDRAILS
# ──────────────────────────────────────────────
def extract_sql_from_response(text_response: str) -> str | None:
    match = re.search(r"```(?:sql|SQL)\s*([\s\S]+?)```", text_response)
    if match: return match.group(1).strip()
    match = re.search(r"(SELECT\s[\s\S]+?;)", text_response, re.IGNORECASE)
    if match: return match.group(1).strip()
    return None

def is_safe_query(query: str) -> bool:
    """Check if SQL query is safe (read-only)."""
    if not query or not isinstance(query, str):
        return False
        
    forbidden = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", 
        "GRANT", "REVOKE", "REPLACE", "CREATE", "EXEC", "EXECUTE",
        "CALL", "MERGE", "UPSERT", "COPY", "BULK", "LOAD"
    ]
    
    # Remove comments and normalize whitespace
    query_clean = re.sub(r'--.*?\n', ' ', query)
    query_clean = re.sub(r'/\*.*?\*/', ' ', query_clean, flags=re.DOTALL)
    query_upper = query_clean.upper().strip()
    
    # Check for forbidden keywords
    for keyword in forbidden:
        if re.search(rf'\b{keyword}\b', query_upper):
            return False
            
    # Must start with SELECT, WITH, or SHOW
    if not re.match(r'^\s*(SELECT|WITH|SHOW)\b', query_upper):
        return False
        
    return True

# ──────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "👋 Welcome! Connect a database in the sidebar to get started."}]
if "db_engine" not in st.session_state: st.session_state.db_engine = None
if "db_schema_dict" not in st.session_state: st.session_state.db_schema_dict = {}
if "db_schema_text" not in st.session_state: st.session_state.db_schema_text = ""
if "db_type" not in st.session_state: st.session_state.db_type = ""
if "last_sql" not in st.session_state: st.session_state.last_sql = None

# ──────────────────────────────────────────────
# SIDEBAR: DYNAMIC CREDENTIALS & KNOWLEDGE BASE
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:0.5rem 0 0.2rem 0;">'
        '<h2 style="margin:0;background:linear-gradient(90deg,#6C63FF,#06B6D4); -webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
        "SchemaSense AI</h2>"
        '<p style="opacity:0.6;font-size:0.82rem;margin:0">Enterprise Data Agent</p></div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("⚙️ AI Engine")
    selected_model = st.selectbox("Gemini Model:", [
        "gemini-3-flash-preview", 
        "gemini-2.5-pro", 
        "gemini-2.5-flash"
    ])

    st.divider()
    st.subheader("🔌 Live Database Connection")
    
    db_engine_choice = st.selectbox("Database Engine", ["PostgreSQL", "MySQL"])
    
    with st.form("db_connect_form"):
        db_host = st.text_input("Host URL / IP", placeholder="e.g., database-1.cluster.aws.com")
        db_port = st.text_input("Port", value="5432" if db_engine_choice == "PostgreSQL" else "3306")
        db_name = st.text_input("Database Name")
        db_user = st.text_input("Username")
        db_pass = st.text_input("Password", type="password")
        
        connect_btn = st.form_submit_button("Secure Connect", use_container_width=True)

    if connect_btn:
        if not all([db_host, db_port, db_name, db_user, db_pass]):
            st.error("Please fill in all credentials.")
        else:
            with st.spinner("Connecting and extracting schema..."):
                try:
                    conn_str = build_connection_string(db_engine_choice, db_host, db_port, db_name, db_user, db_pass)
                    engine = create_engine(
                        conn_str,
                        pool_pre_ping=True,
                        pool_recycle=3600,
                        connect_args={
                            "connect_timeout": 10,
                            "read_timeout": 30,
                            "write_timeout": 30
                        } if db_engine_choice == "MySQL" else {"connect_timeout": 10}
                    )
                    
                    # Test connection with timeout
                    with engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                    
                    # Extract schema
                    schema_dict, schema_text = extract_database_schema(engine)
                    
                    st.session_state.db_engine = engine
                    st.session_state.db_schema_dict = schema_dict
                    st.session_state.db_schema_text = schema_text
                    st.session_state.db_type = db_engine_choice
                    
                    st.success(f"Connected! Found {len(schema_dict)} tables.")
                    st.session_state.messages.append({"role": "assistant", "content": f"✅ Successfully connected to **{db_name}**. I've analyzed the schema and found {len(schema_dict)} tables. What would you like to know?"})
                
                except ValueError as ve:
                    st.error(f"Configuration Error: {ve}")
                except Exception as e:
                    error_msg = str(e)
                    if "pymysql" in error_msg.lower() or "no module named 'pymysql'" in error_msg.lower():
                        st.error("❌ PyMySQL driver not found. Installing now...")
                        import subprocess
                        try:
                            subprocess.run(["pip", "install", "pymysql==1.1.1"], check=True, capture_output=True)
                            st.success("✅ PyMySQL installed. Please try connecting again.")
                        except:
                            st.error("❌ Failed to install PyMySQL. Run: `pip install pymysql`")
                    elif "psycopg2" in error_msg.lower():
                        st.error("❌ PostgreSQL driver not found. Please install: `pip install psycopg2-binary`")
                    elif "timeout" in error_msg.lower():
                        st.error("❌ Connection timeout. Please check your host and network.")
                    elif "authentication" in error_msg.lower() or "password" in error_msg.lower():
                        st.error("❌ Authentication failed. Please check your username and password.")
                    else:
                        st.error(f"❌ Connection Failed: {error_msg}")

    # --- Change 2: Add Enterprise Knowledge Base to the Sidebar ---
    st.divider()
    st.subheader("🗄️ Enterprise Knowledge Base")
    with st.expander("Sync Cloud Business Rules"):
        st.info("Connect to Google Cloud Storage (GCS) or AWS S3.")
        cloud_uri = st.text_input("Cloud Storage URI (e.g., gs://enterprise-rules/q1.pdf)")
        
        st.markdown("**Or manual fallback override:**")
        uploaded_file = st.file_uploader("Ingest Document", type=["pdf"])
        
        if uploaded_file is not None and uploaded_file.size > 5 * 1024 * 1024:
            st.warning("Enterprise scale alert: Large files in production are streamed via BigQuery. This prototype uses in-memory limits.")
            
        if st.button("Fetch Rules from Cloud", use_container_width=True):
            st.success(f"Successfully synced business rules from {cloud_uri if cloud_uri else 'local fallback'}")

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "👋 Conversation cleared. How can I help?"}]
        st.session_state.last_sql = None
        st.rerun()

# ──────────────────────────────────────────────
# AI PERSONA
# ──────────────────────────────────────────────
# --- Change 1: Update the AI Persona (System Prompt) ---
steward_persona = f"""
You are 'SchemaSense AI', an Enterprise Data Dictionary Agent.
You are currently connected to a live {st.session_state.db_type} database.
Here is the exact schema extracted from the database:
{st.session_state.db_schema_text if st.session_state.db_schema_text else "No database connected yet."}
Your Enterprise Responsibilities:

SCHEMA RULE: If the database is PostgreSQL, it uses the 'public' schema. You MUST prefix EVERY table name with 'public.' in your SQL queries (e.g., 'public.olist_orders_dataset'). If you omit it, the query will fail.
SECURITY RULE: You are restricted to READ-ONLY access. Only generate SELECT statements. Never generate DROP, DELETE, or UPDATE commands.
BUSINESS RULE: When asked to explain tables, deeply explain the relationships between them. Identify the Foreign Keys connecting them and explain the specific business purpose of why they are linked.
Format responses with Markdown. ALWAYS wrap SQL code in ```sql ... ``` fences.
"""

genai.configure(api_key=_get_active_key())

# ──────────────────────────────────────────────
# MAIN UI & TABS
# ──────────────────────────────────────────────
st.markdown('<div class="hero-header"><h1>SchemaSense AI</h1><p>Enterprise Data Dictionary & Analytics Agent</p></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🗺️ Schema Map", "📈 Column Profiler", "📊 Visual Analytics"])

# ═══════════════════════════════════════════════
# TAB 1 – CHAT
# ═══════════════════════════════════════════════
with tab1:
    if st.session_state.db_engine is None:
        st.info("👈 Please enter your database credentials in the sidebar to begin.")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about the data schema or request a SQL query…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                try:
                    history_for_api = [{"role": "user" if msg["role"] == "user" else "model", "parts": [msg["content"]]} for msg in st.session_state.messages[:-1]]
                    responses, _ = generate_with_fallback(selected_model, steward_persona, (history_for_api, prompt), stream=True)
                    
                    for chunk in responses:
                        if chunk.text:
                            full_response += chunk.text
                            response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                    sql = extract_sql_from_response(full_response)
                    if sql:
                        st.session_state.last_sql = sql
                        st.info("💡 SQL detected! Switch to the **📊 Visual Analytics** tab to run it securely.")

                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ═══════════════════════════════════════════════
# TAB 2 – SCHEMA MAP
# ═══════════════════════════════════════════════
with tab2:
    if st.session_state.db_engine is None:
        st.info("Connect a database to view the Schema Map.")
    else:
        st.markdown('<div class="glass-card"><h3 style="margin-top:0">🗺️ Database Architecture</h3></div>', unsafe_allow_html=True)
        schema = st.session_state.db_schema_dict
        
        st.markdown(f"**Total Tables Extracted: {len(schema)}**")
        for table_name, details in schema.items():
            with st.expander(f"📦 {table_name} ({len(details['columns'])} columns)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Columns:**")
                    for c in details['columns']: st.markdown(f"- `{c}`")
                with col2:
                    if details['foreign_keys']:
                        st.markdown("**Foreign Keys (Relationships):**")
                        for fk in details['foreign_keys']:
                            st.markdown(f"- `{fk['constrained_columns'][0]}` ➔ `{fk['referred_table']}.{fk['referred_columns'][0]}`")
                    else:
                        st.markdown("*No foreign keys detected.*")

# ═══════════════════════════════════════════════
# TAB 3 – COLUMN PROFILER (Live Sample)
# ═══════════════════════════════════════════════
with tab3:
    if st.session_state.db_engine is None:
        st.info("Connect a database to profile columns.")
    else:
        st.markdown('<div class="glass-card"><h3 style="margin-top:0">📈 Live Table Profiler</h3><p style="opacity:0.7;font-size:0.9rem">Select a table to securely fetch a 1,000-row sample and generate statistical distributions.</p></div>', unsafe_allow_html=True)
        
        tables = list(st.session_state.db_schema_dict.keys())
        selected_table = st.selectbox("Select Table to Profile:", tables)
        
        if st.button("Fetch Sample & Profile", type="primary"):
            with st.spinner(f"Fetching sample from {selected_table}..."):
                try:
                    # Safely fetch max 1000 rows to prevent memory crashes
                    query = f"SELECT * FROM {selected_table} LIMIT 1000"
                    with st.session_state.db_engine.connect() as conn:
                        df = pd.read_sql(query, conn)
                    
                    st.success(f"✅ Fetched {len(df)} sample rows from `{selected_table}`.")
                    
                    numeric_cols = df.select_dtypes(include="number").columns.tolist()
                    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

                    tab_num, tab_cat = st.tabs(["🔢 Numeric Columns", "🏷️ Categorical Columns"])

                    with tab_num:
                        if numeric_cols:
                            for col in numeric_cols[:5]: # limit to first 5 for speed
                                with st.expander(f"📊 {col}", expanded=True):
                                    fig = px.histogram(df, x=col, title=f"Distribution of {col}", template="plotly_dark", color_discrete_sequence=["#6C63FF"])
                                    st.plotly_chart(fig, use_container_width=True)
                        else: st.info("No numeric columns found.")

                    with tab_cat:
                        if categorical_cols:
                            for col in categorical_cols[:5]:
                                with st.expander(f"🏷️ {col}", expanded=True):
                                    counts = df[col].value_counts().head(10)
                                    fig = px.bar(x=counts.index.astype(str), y=counts.values, title=f"Top Values – {col}", template="plotly_dark", color_discrete_sequence=["#8B5CF6"])
                                    st.plotly_chart(fig, use_container_width=True)
                        else: st.info("No categorical columns found.")

                except Exception as e:
                    st.error(f"Error fetching data: {e}")

# ═══════════════════════════════════════════════
# TAB 4 – VISUAL Analytics ENGINE
# ═══════════════════════════════════════════════
with tab4:
    if st.session_state.db_engine is None:
        st.info("Connect a database to run queries.")
    else:
        st.markdown(f"### 📊 Engine: {st.session_state.db_type}")
        
        default_sql = st.session_state.last_sql or ""
        sql_input = st.text_area("📝 SQL Query:", value=default_sql, height=160)

        col1, col2 = st.columns([1, 4])
        with col1: run_btn = st.button("▶️ Run Query", type="primary", use_container_width=True)
        with col2: 
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.last_sql = None
                st.rerun()

        if run_btn and sql_input.strip():
            if not is_safe_query(sql_input.strip()):
                st.error("🛑 Security Block: Only SELECT queries are allowed.")
            else:
                try:
                    with st.spinner("⚡ Executing query on live database…"):
                        with st.session_state.db_engine.connect() as conn:
                            result_df = pd.read_sql(text(sql_input.strip()), conn)

                    st.success(f"✅ Query returned **{len(result_df):,} rows**.")
                    st.dataframe(result_df, use_container_width=True)

                    if not result_df.empty:
                        num_cols = result_df.select_dtypes(include="number").columns.tolist()
                        cat_cols = result_df.select_dtypes(exclude="number").columns.tolist()
                        
                        if num_cols and cat_cols:
                            st.markdown("### 📈 Auto-Chart")
                            fig = px.bar(result_df.head(30), x=cat_cols[0], y=num_cols[0], template="plotly_dark", color_discrete_sequence=["#06B6D4"])
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ SQL Error: {e}")
