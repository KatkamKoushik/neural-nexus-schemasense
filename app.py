import streamlit as st
try:
    import PyPDF2
    _PYPDF2_AVAILABLE = True
except ImportError:
    _PYPDF2_AVAILABLE = False
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import json
import sqlite3
import re
from io import StringIO
import streamlit.components.v1 as components

# ──────────────────────────────────────────────
# INITIAL CONFIGURATION
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SchemaSense AI – Intelligent Data Dictionary Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS – glassmorphism, gradients, animations
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Google Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

/* ---------- Gradient hero header ---------- */
.hero-header {
    background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 50%, #06B6D4 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(circle at 80% 20%, rgba(255,255,255,0.15) 0%, transparent 60%);
    pointer-events: none;
}
.hero-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.hero-header p  { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 1rem; }

/* ---------- Glassmorphism cards ---------- */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(108,99,255,0.12);
}

/* ---------- Animated status badge ---------- */
.status-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(16,185,129,0.12);
    color: #10B981;
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.82rem; font-weight: 600;
}
.status-badge::before {
    content: '';
    width: 8px; height: 8px;
    background: #10B981;
    border-radius: 50%;
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }
    70%  { box-shadow: 0 0 0 8px rgba(16,185,129,0); }
    100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
}

/* ---------- Suggestion chips ---------- */
.chip-container { display: flex; flex-wrap: wrap; gap: 8px; margin: 0.75rem 0 1rem 0; }
div.stButton > button[kind="secondary"] {
    border: 1px solid rgba(108,99,255,0.35) !important;
    border-radius: 999px !important;
    font-size: 0.82rem !important;
    padding: 4px 16px !important;
    background: rgba(108,99,255,0.06) !important;
    color: #A5B4FC !important;
    transition: all 0.2s ease !important;
}
div.stButton > button[kind="secondary"]:hover {
    background: rgba(108,99,255,0.18) !important;
    border-color: #6C63FF !important;
    color: #fff !important;
}

/* ---------- Metric card style ---------- */
.metric-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 1rem;
}
.metric-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-item .label { font-size: 0.75rem; opacity: 0.6; text-transform: uppercase; letter-spacing: 1px; }
.metric-item .value { font-size: 1.5rem; font-weight: 700; color: #6C63FF; margin-top: 4px; }

/* ---------- Schema table ---------- */
.schema-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.schema-table th {
    text-align: left; padding: 8px 12px;
    background: rgba(108,99,255,0.1);
    border-bottom: 2px solid rgba(108,99,255,0.25);
    color: #A5B4FC; font-weight: 600;
}
.schema-table td {
    padding: 6px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}

/* ---------- Tab styling ---------- */
div[data-baseweb="tab-list"] { gap: 4px; }
div[data-baseweb="tab"] {
    border-radius: 10px 10px 0 0 !important;
    font-weight: 600 !important;
}

/* ---------- Key rotation info badge ---------- */
.key-info {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(59,130,246,0.12);
    color: #60A5FA;
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.78rem; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# FEATURE 1 – ROBUST API KEY FALLBACK
# ──────────────────────────────────────────────
def _load_api_keys() -> list[str]:
    """Load API keys from st.secrets (list) or fall back to a single env/hardcoded key."""
    try:
        keys = st.secrets.get("GEMINI_KEYS", None)
        if keys and isinstance(keys, list) and len(keys) > 0:
            return [k for k in keys if k and k != "YOUR_KEY_HERE"]
    except Exception:
        pass
    # Fallback: single key from secrets or placeholder
    try:
        single = st.secrets.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
        if single and single != "YOUR_API_KEY_HERE":
            return [single]
    except Exception:
        pass
    return ["YOUR_API_KEY_HERE"]


API_KEYS = _load_api_keys()

# Track which key is active across the session
if "active_key_index" not in st.session_state:
    st.session_state.active_key_index = 0


def _get_active_key() -> str:
    idx = st.session_state.active_key_index % len(API_KEYS)
    return API_KEYS[idx]


def _rotate_key():
    """Rotate to the next API key."""
    st.session_state.active_key_index = (st.session_state.active_key_index + 1) % len(API_KEYS)


def generate_with_fallback(model_name: str, system_prompt: str, messages_or_prompt, stream: bool = False):
    """
    Call model.generate_content() (or chat.send_message) with automatic key rotation on 429 errors.
    Returns (response, key_index_used).
    Raises the last exception if all keys are exhausted.
    """
    attempts = len(API_KEYS)
    last_exc = None

    for attempt in range(attempts):
        current_key = _get_active_key()
        genai.configure(api_key=current_key)
        model = genai.GenerativeModel(model_name, system_instruction=system_prompt)

        try:
            if isinstance(messages_or_prompt, str):
                # Simple single-turn call (e.g. quality scan)
                response = model.generate_content(messages_or_prompt)
                return response, st.session_state.active_key_index

            else:
                # Multi-turn: messages_or_prompt is (history, user_prompt)
                history, user_prompt = messages_or_prompt
                chat = model.start_chat(history=history)
                response = chat.send_message(user_prompt, stream=stream)
                return response, st.session_state.active_key_index

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                last_exc = e
                _rotate_key()  # Try next key
                st.toast(f"⚠️ Key #{attempt + 1} quota hit – switching to next key…", icon="🔄")
            else:
                raise e  # Non-quota errors bubble up immediately

    raise last_exc  # All keys exhausted


# ──────────────────────────────────────────────
# FEATURE 2 – SQLITE MULTI-DATASET ENGINE
# ──────────────────────────────────────────────

def load_csvs_into_sqlite(files) -> tuple[sqlite3.Connection | None, dict]:
    """
    Load a list of uploaded CSV files into an in-memory SQLite DB.
    Returns (connection, schema_info_dict).
    Table names are derived from the file names (sanitized).
    """
    if not files:
        return None, {}

    conn = sqlite3.connect(":memory:")
    schema_info = {}  # {table_name: [col_name, ...]}

    for f in files:
        # Sanitize filename → table name
        table_name = re.sub(r"[^a-zA-Z0-9_]", "_", f.name.rsplit(".", 1)[0])
        table_name = re.sub(r"_+", "_", table_name).strip("_")

        try:
            df = pd.read_csv(f)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            schema_info[table_name] = list(df.columns)
        except Exception as e:
            st.warning(f"Could not load `{f.name}`: {e}")

    return conn, schema_info


def build_sqlite_system_prompt(schema_info: dict) -> str:
    """Build a system prompt section describing the SQLite schema for JOIN-aware AI queries."""
    if not schema_info:
        return ""
    lines = ["The user has loaded the following tables into an in-memory **SQLite** database:"]
    for table, cols in schema_info.items():
        lines.append(f"\n  • **{table}** ( {', '.join(cols)} )")
    lines.append(
        "\n\nWhen writing SQL queries, use **SQLite syntax** (not PostgreSQL). "
        "You can perform JOINs across these tables. "
        "Always reference actual column names exactly as shown above."
    )
    return "\n".join(lines)


def extract_schema_metadata(df: pd.DataFrame) -> dict:
    """Return rich metadata dict from a DataFrame (used for dictionary + sidebar)."""
    meta = {}
    for col in df.columns:
        meta[col] = {
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].notna().sum()),
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(df[col].isna().mean() * 100, 1),
            "unique_values": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(3).tolist(),
        }
    return meta


def build_data_dictionary_json(source_name: str, meta: dict | None) -> str:
    """Build a rich JSON data dictionary."""
    if meta:
        columns = []
        for col, info in meta.items():
            columns.append({
                "column_name": col,
                "data_type": info["dtype"],
                "non_null_count": info["non_null_count"],
                "null_count": info["null_count"],
                "null_percentage": info["null_pct"],
                "unique_values": info["unique_values"],
                "sample_values": info["sample_values"],
            })
    else:
        columns = [
            {"table": t, "status": "schema extracted"}
            for t in [
                "olist_orders_dataset",
                "olist_order_items_dataset",
                "olist_products_dataset",
                "olist_customers_dataset",
                "olist_sellers_dataset",
                "olist_order_payments_dataset",
                "olist_order_reviews_dataset",
                "olist_geolocation_dataset",
                "olist_product_category_name_translation",
            ]
        ]

    dictionary = {
        "generated_by": "SchemaSense AI",
        "dataset": source_name,
        "description": "Auto-generated Intelligent Data Dictionary",
        "columns": columns,
    }
    return json.dumps(dictionary, indent=4, default=str)


def render_schema_table_html(meta: dict) -> str:
    """Render a compact HTML table from schema metadata."""
    rows = ""
    for col, info in meta.items():
        null_badge = (
            f'<span style="color:#EF4444;font-weight:600">{info["null_pct"]}%</span>'
            if info["null_pct"] > 0
            else '<span style="color:#10B981">0%</span>'
        )
        rows += f"""<tr>
            <td><code>{col}</code></td>
            <td>{info['dtype']}</td>
            <td>{info['unique_values']}</td>
            <td>{null_badge}</td>
        </tr>"""
    return f"""<table class="schema-table">
        <thead><tr><th>Column</th><th>Type</th><th>Unique</th><th>Nulls</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


# ──────────────────────────────────────────────
# FEATURE 3 – SQL EXTRACTION HELPER
# ──────────────────────────────────────────────
def extract_sql_from_response(text: str) -> str | None:
    """
    Extract the first SQL query from an AI response.
    Looks for ```sql ... ``` fenced code blocks, or a SELECT/WITH statement.
    Returns the raw SQL string or None.
    """
    # Priority 1: fenced ```sql block
    match = re.search(r"```(?:sql|SQL)\s*([\s\S]+?)```", text)
    if match:
        return match.group(1).strip()

    # Priority 2: fenced ``` block that starts with SELECT/WITH/INSERT/UPDATE
    match = re.search(r"```\s*((?:SELECT|WITH|INSERT|UPDATE|DELETE)[\s\S]+?)```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Priority 3: bare SELECT statement
    match = re.search(r"(SELECT\s[\s\S]+?;)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


# ──────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Data connection established!** I'm your AI Data Steward. Ask me to describe tables, write SQL queries, or analyze data quality.",
        }
    ]
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "uploaded_meta" not in st.session_state:
    st.session_state.uploaded_meta = None
if "sqlite_conn" not in st.session_state:
    st.session_state.sqlite_conn = None
if "sqlite_schema" not in st.session_state:
    st.session_state.sqlite_schema = {}
if "last_sql" not in st.session_state:
    st.session_state.last_sql = None


# ──────────────────────────────────────────────
# LIVE POSTGRESQL CONNECTION  (Olist / Cloud SQL)
# ──────────────────────────────────────────────
# Initialised once per session via Streamlit's native connector.
# Credentials must be set in .streamlit/secrets.toml under [connections.postgresql].
# On Streamlit Cloud, configure the same block in the Secrets manager.
pg_conn = None
pg_conn_error = None
try:
    pg_conn = st.connection("postgresql", type="sql")
    # Lightweight ping to validate the connection immediately
    pg_conn.query("SELECT 1", ttl=0)
except Exception as _pg_err:
    pg_conn_error = str(_pg_err)


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:0.5rem 0 0.2rem 0;">'
        '<span style="font-size:2.2rem">🧠</span>'
        '<h2 style="margin:0;background:linear-gradient(90deg,#6C63FF,#06B6D4);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
        "SchemaSense AI</h2>"
        '<p style="opacity:0.6;font-size:0.82rem;margin:0">Intelligent Data Dictionary Agent</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── AI Engine ──
    st.subheader("⚙️ AI Engine")
    _MODEL_OPTIONS = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
    ]
    _MODEL_LABELS = {
        "gemini-2.5-flash":      "Gemini 3 Flash (Fast)",
        "gemini-2.0-flash":      "Gemini 2 Flash (Balanced)",
        "gemini-2.0-flash-lite": "Gemini 2 Flash Lite (Fastest)",
        "gemini-2.5-flash-lite": "Gemini 3 Flash Lite (Lightweight)",
        "gemini-2.5-pro":        "Gemini 3 Pro (High Reasoning)",
    }
    selected_model = st.selectbox(
        "Gemini Model:",
        options=_MODEL_OPTIONS,
        format_func=lambda m: _MODEL_LABELS.get(m, m),
        help="Switch models if you hit quota limits.",
    )

    st.divider()

    # ── Data Source ──
    with st.sidebar.container():
        st.subheader("🔌 Data Source")
        data_source = st.radio(
            "Select Source:",
            ["Built-in: Olist E-Commerce", "Upload Custom Dataset"],
        )
    
        custom_schema_context = ""
    
        if data_source == "Upload Custom Dataset":
            # FEATURE 2: accept_multiple_files=True
            uploaded_files = st.file_uploader(
                "Upload CSV files (multiple allowed)",
                type=["csv"],
                accept_multiple_files=True,
            )
    
            if uploaded_files:
                # Load into SQLite in-memory DB
                conn, schema_info = load_csvs_into_sqlite(uploaded_files)
                st.session_state.sqlite_conn = conn
                st.session_state.sqlite_schema = schema_info
    
                # For backward compat: store the first file's df for the column profiler
                first_file = uploaded_files[0]
                first_file.seek(0)
                df_first = pd.read_csv(first_file)
                st.session_state.uploaded_df = df_first
                meta = extract_schema_metadata(df_first)
                st.session_state.uploaded_meta = meta
    
                # Build schema context for the AI
                multi_table_info = []
                for fname, cols in schema_info.items():
                    multi_table_info.append(f"Table `{fname}`: columns = [{', '.join(cols)}]")
    
                custom_schema_context = (
                    f"The user uploaded {len(uploaded_files)} CSV file(s) loaded into a SQLite in-memory database.\n"
                    + "\n".join(multi_table_info)
                    + f"\nFirst file sample rows: {df_first.head(3).to_json(orient='records')}"
                )
    
                # Sidebar schema summary
                st.markdown('<span class="status-badge">Schema Ingested</span>', unsafe_allow_html=True)
                if len(uploaded_files) > 1:
                    st.caption(f"📦 {len(uploaded_files)} tables loaded into SQLite")
                    for tname, cols in schema_info.items():
                        with st.expander(f"📋 {tname} ({len(cols)} cols)"):
                            st.code(", ".join(cols), language=None)
                else:
                    st.markdown(render_schema_table_html(meta), unsafe_allow_html=True)
            else:
                st.info("Upload one or more CSVs to begin analysis.")
                st.session_state.sqlite_conn = None
                st.session_state.sqlite_schema = {}
        else:
            # ── Live PostgreSQL connection status ──
            if pg_conn is not None:
                st.success("🟢 **Connected** · Olist PostgreSQL · Cloud SQL", icon=None)
                st.caption("Live database · 9 tables · queries cached 10 min")
            else:
                st.error("🔴 **DB Unavailable** · Could not connect to PostgreSQL")
                if pg_conn_error:
                    with st.expander("Connection error details"):
                        st.code(pg_conn_error, language=None)
    
        st.sidebar.divider()
        pdf_file = st.sidebar.file_uploader("📂 Upload Business Rules (PDF)", type=['pdf'])
        
        business_context = ""
        if st.session_state.biz_rules is not None:
            if _PYPDF2_AVAILABLE:
                try:
                    pdf_reader = PyPDF2.PdfReader(st.session_state.biz_rules)
                    pages_text = []
                    for page in pdf_reader.pages:
                        pages_text.append(page.extract_text() or "")
                    business_context = "\n".join(pages_text).strip()
                    if business_context:
                        st.success(f"✅ PDF ingested · {len(pdf_reader.pages)} page(s) · {len(business_context):,} chars")
                    else:
                        st.warning("⚠️ PDF uploaded but no text could be extracted (may be image-based).")
                except Exception as _pdf_err:
                    business_context = "PDF attached, but text extraction skipped."
                    st.warning("⚠️ PDF attached, but text extraction skipped.")
            else:
                st.error("❌ PyPDF2 not installed. Add `PyPDF2` to requirements.txt and redeploy.")
    
    st.divider()

    # ── Actions ──
    with st.sidebar.container():
        st.subheader("⚡ Actions")
    
        dict_meta = st.session_state.uploaded_meta if data_source == "Upload Custom Dataset" else None
        dict_json = build_data_dictionary_json(data_source, dict_meta)
        st.download_button(
            label="📥 Download Data Dictionary",
            file_name="schemasense_data_dictionary.json",
            mime="application/json",
            data=dict_json,
            use_container_width=True,
        )
    
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "👋 Conversation cleared. How can I help?",
                }
            ]
            st.session_state.last_sql = None
            st.rerun()


# ──────────────────────────────────────────────
# AI PERSONA & MODEL INIT
# ──────────────────────────────────────────────
# Retrieve PDF business rules text
_business_rules_section = (
    f"\n\n---\n**Follow these business rules:**\n{business_context}\n"
    if business_context else ""
)

if data_source == "Built-in: Olist E-Commerce":
    steward_persona = f"""
You are 'SchemaSense AI', an Intelligent Data Dictionary Agent created for the GDG Cloud Delhi hackathon.
You are connected to the **Olist E-commerce** PostgreSQL database hosted on Google Cloud SQL.

Available tables:
  • olist_orders_dataset (order_id, customer_id, order_status, order_purchase_timestamp, order_delivered_customer_date, …)
  • olist_order_items_dataset (order_id, order_item_id, product_id, seller_id, price, freight_value, …)
  • olist_products_dataset (product_id, product_category_name, product_weight_g, product_length_cm, …)
  • olist_customers_dataset (customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state)
  • olist_sellers_dataset (seller_id, seller_zip_code_prefix, seller_city, seller_state)
  • olist_order_payments_dataset (order_id, payment_sequential, payment_type, payment_installments, payment_value)
  • olist_order_reviews_dataset (review_id, order_id, review_score, review_comment_title, review_comment_message)
  • olist_geolocation_dataset (geolocation_zip_code_prefix, geolocation_lat, geolocation_lng, geolocation_city, geolocation_state)
  • olist_product_category_name_translation (product_category_name, product_category_name_english)

Your responsibilities:
  1. Provide **business-friendly** plain-English summaries of tables and columns.
  2. Write **optimized, production-quality SQL** (PostgreSQL dialect).
  3. Perform **data quality analysis** — identify nulls, duplicates, anomalies.
  4. Explain table **relationships and join paths**.
  5. Format responses with Markdown for clarity.
{_business_rules_section}"""
else:
    sqlite_schema_section = build_sqlite_system_prompt(st.session_state.sqlite_schema)
    steward_persona = f"""
You are 'SchemaSense AI', an Intelligent Data Dictionary Agent.
The user has uploaded custom dataset(s) loaded into an in-memory SQLite database.

{sqlite_schema_section}

Additional context:
{custom_schema_context}

Your responsibilities:
  1. Analyze the schema and explain what the data likely represents.
  2. Suggest data quality checks (nulls, outliers, type issues).
  3. Write SQL queries based **ONLY** on the columns above using **SQLite syntax**.
  4. Provide business-friendly descriptions for each column.
  5. When multiple tables exist, suggest useful JOIN queries.
  6. Format responses with Markdown. Always wrap SQL in ```sql ... ``` fences.
{_business_rules_section}"""

# Configure the API key for non-streaming calls (like quality scan)
genai.configure(api_key=_get_active_key())
model = genai.GenerativeModel(selected_model, system_instruction=steward_persona)


# ──────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────
st.markdown(
    '<div class="hero-header">'
    "<h1>🧠 SchemaSense AI</h1>"
    "<p>Your Intelligent Data Dictionary Agent · Powered by Gemini & Google Cloud</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# TABS  ← added Visual Analytics tab
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["💬 Data Steward Chat", "📊 Schema Map & Quality", "📈 Column Profiler", "📊 Visual Analytics"]
)


# ═══════════════════════════════════════════════
# TAB 1 – AI CHAT
# ═══════════════════════════════════════════════
with tab1:
    # Suggestion chips
    st.markdown("**Quick Actions:**")
    chip_cols = st.columns(5)
    suggestions = [
        "📋 Describe all tables",
        "🔗 Show table relationships",
        "🛠️ Write a JOIN query",
        "🔍 Find anomalies",
        "📊 Suggest KPIs",
    ]

    chip_clicked = None
    for i, label in enumerate(suggestions):
        with chip_cols[i]:
            if st.button(label, key=f"chip_{i}", type="secondary", use_container_width=True):
                chip_clicked = label

    if chip_clicked:
        st.session_state.messages.append({"role": "user", "content": chip_clicked})
        st.rerun()

    st.divider()

    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the data schema …"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                # Build multi-turn history for context-aware conversation
                history_for_api = []
                for msg in st.session_state.messages[:-1]:
                    role = "user" if msg["role"] == "user" else "model"
                    history_for_api.append({"role": role, "parts": [msg["content"]]})

                with st.status("🧠 SchemaSense is thinking …", expanded=False):
                    # FEATURE 1: use key-rotation wrapper (streaming)
                    responses, _ = generate_with_fallback(
                        selected_model,
                        steward_persona,
                        (history_for_api, prompt),
                        stream=True,
                    )
                    for chunk in responses:
                        if chunk.text:
                            full_response += chunk.text
                            response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

                # FEATURE 3: Extract SQL and cache it for Visual Analytics tab
                sql = extract_sql_from_response(full_response)
                if sql:
                    st.session_state.last_sql = sql
                    st.info("💡 SQL detected! Switch to the **📊 Visual Analytics** tab to run it.")

            except Exception as e:
                err_msg = str(e)
                if "quota" in err_msg.lower() or "429" in err_msg:
                    st.error(
                        "⚠️ **All API quota keys exhausted.** Please wait a moment and try again, "
                        "or add more keys under `GEMINI_KEYS` in `.streamlit/secrets.toml`."
                    )
                else:
                    st.error(f"❌ Error: {e}")


# ═══════════════════════════════════════════════
# TAB 2 – SCHEMA MAP & QUALITY
# ═══════════════════════════════════════════════
with tab2:
    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown(
            '<div class="glass-card">'
            "<h3 style='margin-top:0'>🛡️ Intelligent Quality Guard</h3>"
            "<p style='opacity:0.7;font-size:0.9rem'>Run an automated AI-powered scan for anomalies, "
            "missing data, and business rule violations.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        if st.button("🚀 Run Data Quality Scan", type="primary", use_container_width=True):
            with st.status("🔬 Analyzing schema health …", expanded=True) as status:
                quality_prompt = (
                    "Execute the Intelligent Quality Guard protocol. "
                    "Generate a comprehensive Data Quality Report for the active dataset. "
                    "Include sections for: "
                    "1) **Overview** – dataset summary stats. "
                    "2) **Completeness** – missing/null analysis per column. "
                    "3) **Consistency** – data type mismatches or formatting issues. "
                    "4) **Anomalies** – outlier risks and statistical health. "
                    "5) **Recommendations** – prioritized actions to improve data quality. "
                    "Use tables, bullet points, and severity emojis (🟢🟡🔴)."
                )
                # FEATURE 1: key rotation wrapper
                response, key_idx = generate_with_fallback(
                    selected_model, steward_persona, quality_prompt
                )
                status.update(label="✅ Scan complete!", state="complete")
            st.markdown(response.text)

    with col_right:
        if data_source == "Built-in: Olist E-Commerce":
            st.markdown(
                '<div class="glass-card">'
                "<h3 style='margin-top:0'>🗺️ Olist Relationship Map</h3>"
                "</div>",
                unsafe_allow_html=True,
            )
            mermaid_code = '''erDiagram
    CUSTOMERS ||--o{ ORDERS : places
    ORDERS ||--o{ ORDER_ITEMS : contains
    ORDERS ||--o{ PAYMENTS : has
    ORDERS ||--o{ REVIEWS : receives
    PRODUCTS ||--o{ ORDER_ITEMS : included_in
    SELLERS ||--o{ ORDER_ITEMS : fulfills
    GEOLOCATION }o--|| CUSTOMERS : locates
    CATEGORY_TRANSLATION ||--|| PRODUCTS : translates'''
            st.components.v1.html(f'<div class="mermaid">{mermaid_code}</div>', height=400)
        else:
            if st.session_state.uploaded_meta:
                st.markdown(
                    '<div class="glass-card">'
                    "<h3 style='margin-top:0'>📋 Schema Summary</h3>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                meta = st.session_state.uploaded_meta
                total_cols = len(meta)
                total_nulls = sum(v["null_count"] for v in meta.values())
                avg_unique = round(sum(v["unique_values"] for v in meta.values()) / max(total_cols, 1))

                st.markdown(
                    f"""<div class="metric-row">
                        <div class="metric-item"><div class="label">Columns</div><div class="value">{total_cols}</div></div>
                        <div class="metric-item"><div class="label">Total Nulls</div><div class="value">{total_nulls:,}</div></div>
                        <div class="metric-item"><div class="label">Avg Unique</div><div class="value">{avg_unique:,}</div></div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # Multi-table schema map
                if len(st.session_state.sqlite_schema) > 1:
                    st.markdown("**📦 Loaded Tables:**")
                    for tname, cols in st.session_state.sqlite_schema.items():
                        st.markdown(f"- `{tname}`: {', '.join([f'`{c}`' for c in cols[:6]])}{'…' if len(cols) > 6 else ''}")
            else:
                st.info("📂 Upload a dataset to generate insights.")


# ═══════════════════════════════════════════════
# TAB 3 – COLUMN PROFILER
# ═══════════════════════════════════════════════
with tab3:
    st.markdown(
        '<div class="glass-card">'
        "<h3 style='margin-top:0'>📈 Column Profiler</h3>"
        "<p style='opacity:0.7;font-size:0.9rem'>Auto-generated distribution charts for every column in your dataset.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    if data_source == "Upload Custom Dataset" and st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df

        # Overview metrics
        st.markdown(
            f"""<div class="metric-row">
                <div class="metric-item"><div class="label">Rows</div><div class="value">{len(df):,}</div></div>
                <div class="metric-item"><div class="label">Columns</div><div class="value">{len(df.columns)}</div></div>
                <div class="metric-item"><div class="label">Memory</div><div class="value">{df.memory_usage(deep=True).sum() / 1024:.0f} KB</div></div>
                <div class="metric-item"><div class="label">Duplicates</div><div class="value">{df.duplicated().sum():,}</div></div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.divider()

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        profile_tab_num, profile_tab_cat = st.tabs(["🔢 Numeric Columns", "🏷️ Categorical Columns"])

        with profile_tab_num:
            if numeric_cols:
                sel_num = st.multiselect(
                    "Select numeric columns to profile:",
                    numeric_cols,
                    default=numeric_cols[:3],
                )
                for col in sel_num:
                    with st.expander(f"📊 {col}", expanded=True):
                        c1, c2 = st.columns(2)
                        with c1:
                            fig = px.histogram(
                                df, x=col, nbins=30,
                                title=f"Distribution of {col}",
                                color_discrete_sequence=["#6C63FF"],
                                template="plotly_dark",
                            )
                            fig.update_layout(
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=20, r=20, t=40, b=20),
                                height=300,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        with c2:
                            fig2 = px.box(
                                df, y=col,
                                title=f"Box Plot – {col}",
                                color_discrete_sequence=["#06B6D4"],
                                template="plotly_dark",
                            )
                            fig2.update_layout(
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=20, r=20, t=40, b=20),
                                height=300,
                            )
                            st.plotly_chart(fig2, use_container_width=True)

                        stats = df[col].describe()
                        st.markdown(
                            f"""<div class="metric-row">
                                <div class="metric-item"><div class="label">Mean</div><div class="value">{stats['mean']:.2f}</div></div>
                                <div class="metric-item"><div class="label">Std</div><div class="value">{stats['std']:.2f}</div></div>
                                <div class="metric-item"><div class="label">Min</div><div class="value">{stats['min']:.2f}</div></div>
                                <div class="metric-item"><div class="label">Max</div><div class="value">{stats['max']:.2f}</div></div>
                                <div class="metric-item"><div class="label">Nulls</div><div class="value">{df[col].isna().sum()}</div></div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
            else:
                st.info("No numeric columns found in the dataset.")

        with profile_tab_cat:
            if categorical_cols:
                sel_cat = st.multiselect(
                    "Select categorical columns to profile:",
                    categorical_cols,
                    default=categorical_cols[:3],
                )
                for col in sel_cat:
                    with st.expander(f"🏷️ {col}", expanded=True):
                        value_counts = df[col].value_counts().head(15)
                        fig = px.bar(
                            x=value_counts.index.astype(str),
                            y=value_counts.values,
                            title=f"Top Values – {col}",
                            labels={"x": col, "y": "Count"},
                            color_discrete_sequence=["#8B5CF6"],
                            template="plotly_dark",
                        )
                        fig.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=320,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown(
                            f"""<div class="metric-row">
                                <div class="metric-item"><div class="label">Unique</div><div class="value">{df[col].nunique()}</div></div>
                                <div class="metric-item"><div class="label">Most Common</div><div class="value">{df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'}</div></div>
                                <div class="metric-item"><div class="label">Nulls</div><div class="value">{df[col].isna().sum()}</div></div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
            else:
                st.info("No categorical columns found in the dataset.")

    elif data_source == "Built-in: Olist E-Commerce":
        st.info(
            "Column Profiling for the massive 100k+ row Olist database is handled via the "
            "AI Quality Guard. To use the interactive micro-profiler, switch to "
            "**Upload Custom Dataset** in the sidebar and upload a CSV."
        )
    else:
        st.info("📂 Upload a CSV in the sidebar to start profiling columns.")


# ═══════════════════════════════════════════════
# TAB 4 – VISUAL ANALYTICS ENGINE  (FEATURE 3)
# Dual-mode: PostgreSQL (Olist) or SQLite (CSV uploads)
# ═══════════════════════════════════════════════
with tab4:
    # ── Determine active backend ──
    _using_postgres = (data_source == "Built-in: Olist E-Commerce")
    _sql_hint = "PostgreSQL" if _using_postgres else "SQLite"

    st.title("📊 Visual Analytics Engine")
    st.write(f"Execute SQL against the **{_sql_hint}** database. Queries are auto-detected from the chat.")

    # ── Branch: Olist PostgreSQL vs CSV SQLite ──
    if _using_postgres:
        # ── PostgreSQL path ──────────────────────────────────────────
        if pg_conn is None:
            st.error(
                "🔴 **PostgreSQL connection unavailable.** "
                "Configure `[connections.postgresql]` in `.streamlit/secrets.toml` "
                "(or in Streamlit Cloud Secrets) and redeploy."
            )
            if pg_conn_error:
                st.code(pg_conn_error, language=None)
        else:
            # Show Olist table reference
            with st.expander('📊 Data Schema Details', expanded=False):
                st.markdown('''
* **olist_orders_dataset** — order_id, customer_id, order_status, order_purchase_timestamp...
* **olist_order_items_dataset** — order_id, product_id, seller_id, price, freight_value...
* **olist_products_dataset** — product_id, product_category_name...
* **olist_customers_dataset** — customer_id, customer_city, customer_state...
* **olist_sellers_dataset** — seller_id, seller_city, seller_state...
* **olist_order_payments_dataset** — order_id, payment_type, payment_value...
* **olist_order_reviews_dataset** — review_id, order_id, review_score...
* **olist_geolocation_dataset** — geolocation_zip_code_prefix, geolocation_lat, geolocation_lng...
* **olist_product_category_name_translation** — product_category_name, product_category_name_english...
''')

            st.divider()

            # SQL Editor pre-filled with AI-detected SQL
            default_sql = st.session_state.last_sql or ""
            if default_sql:
                st.success("✅ SQL auto-detected from the latest AI response. You can edit it below.")

            sql_input = st.text_area(
                "📝 SQL Query (PostgreSQL syntax):",
                value=default_sql,
                height=160,
                placeholder=(
                    "SELECT product_category_name, COUNT(*) AS order_count\n"
                    "FROM olist_order_items_dataset oi\n"
                    "JOIN olist_products_dataset p USING (product_id)\n"
                    "GROUP BY 1 ORDER BY 2 DESC LIMIT 20;"
                ),
            )

            col_run, col_clear = st.columns([1, 4])
            with col_run:
                run_btn = st.button("▶️ Run Query", type="primary", use_container_width=True, key="pg_run")
            with col_clear:
                if st.button("🗑️ Clear SQL", use_container_width=True, key="pg_clear"):
                    st.session_state.last_sql = None
                    st.rerun()

            if run_btn and sql_input.strip():
                try:
                    with st.spinner("⚡ Querying PostgreSQL (Cloud SQL)…"):
                        # conn.query() returns a DataFrame and caches results for 10 minutes
                        result_df = pg_conn.query(sql_input.strip(), ttl="10m")

                    st.success(
                        f"✅ Query returned **{len(result_df):,} rows** × "
                        f"**{len(result_df.columns)} columns** · cached for 10 min"
                    )
                    st.dataframe(result_df, use_container_width=True)

                    st.divider()

                    # ── Auto-Chart ──
                    if result_df.empty:
                        st.warning("Query returned 0 rows — no chart to display.")
                    else:
                        numeric_res_cols = result_df.select_dtypes(include="number").columns.tolist()
                        non_numeric_res_cols = result_df.select_dtypes(exclude="number").columns.tolist()

                        if numeric_res_cols:
                            st.markdown("### 📈 Auto-Generated Bar Chart")

                            if len(result_df.columns) >= 2 and non_numeric_res_cols:
                                chart_df = result_df.set_index(non_numeric_res_cols[0])[numeric_res_cols[0:3]]
                            else:
                                chart_df = result_df[numeric_res_cols[0:3]]

                            chart_df = chart_df.head(30)
                            st.bar_chart(chart_df, use_container_width=True)

                            if non_numeric_res_cols and numeric_res_cols:
                                with st.expander("🎨 Enhanced Plotly Chart", expanded=False):
                                    fig = px.bar(
                                        result_df.head(30),
                                        x=non_numeric_res_cols[0],
                                        y=numeric_res_cols[0],
                                        title=f"{numeric_res_cols[0]} by {non_numeric_res_cols[0]}",
                                        color_discrete_sequence=["#6C63FF"],
                                        template="plotly_dark",
                                    )
                                    fig.update_layout(
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        margin=dict(l=20, r=20, t=50, b=20),
                                        xaxis_tickangle=-30,
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("ℹ️ No numeric columns in the result – displaying table only.")

                except Exception as e:
                    st.error(f"❌ PostgreSQL Error: {e}")
                    st.caption(
                        "Tip: Use PostgreSQL syntax. Table names must match exactly "
                        "(see the Olist tables list above)."
                    )

            elif run_btn:
                st.warning("⚠️ Please enter a SQL query first.")

    else:
        # ── SQLite / CSV upload path ─────────────────────────────────
        sqlite_conn = st.session_state.sqlite_conn

        if sqlite_conn is None:
            st.info(
                "📂 **No database loaded.** Please upload one or more CSV files using "
                "'Upload Custom Dataset' in the sidebar to enable the Visual Analytics Engine."
            )
        else:
            schema = st.session_state.sqlite_schema
            if schema:
                with st.expander("📋 Available Tables & Columns", expanded=False):
                    for tname, cols in schema.items():
                        st.markdown(f"**`{tname}`** — {', '.join([f'`{c}`' for c in cols])}")

            st.divider()

            default_sql = st.session_state.last_sql or ""
            if default_sql:
                st.success("✅ SQL auto-detected from the latest AI response. You can edit it below.")

            sql_input = st.text_area(
                "📝 SQL Query (SQLite syntax):",
                value=default_sql,
                height=160,
                placeholder="SELECT column_name, COUNT(*) as count FROM your_table GROUP BY column_name ORDER BY count DESC LIMIT 20;",
            )

            col_run, col_clear = st.columns([1, 4])
            with col_run:
                run_btn = st.button("▶️ Run Query", type="primary", use_container_width=True, key="sqlite_run")
            with col_clear:
                if st.button("🗑️ Clear SQL", use_container_width=True, key="sqlite_clear"):
                    st.session_state.last_sql = None
                    st.rerun()

            if run_btn and sql_input.strip():
                try:
                    with st.spinner("⚡ Executing query…"):
                        result_df = pd.read_sql(sql_input.strip(), sqlite_conn)

                    st.success(f"✅ Query returned **{len(result_df):,} rows** × **{len(result_df.columns)} columns**")
                    st.dataframe(result_df, use_container_width=True)

                    st.divider()

                    if result_df.empty:
                        st.warning("Query returned 0 rows — no chart to display.")
                    else:
                        numeric_res_cols = result_df.select_dtypes(include="number").columns.tolist()
                        non_numeric_res_cols = result_df.select_dtypes(exclude="number").columns.tolist()

                        if numeric_res_cols:
                            st.markdown("### 📈 Auto-Generated Bar Chart")

                            if len(result_df.columns) >= 2 and non_numeric_res_cols:
                                chart_df = result_df.set_index(non_numeric_res_cols[0])[numeric_res_cols[0:3]]
                            else:
                                chart_df = result_df[numeric_res_cols[0:3]]

                            chart_df = chart_df.head(30)
                            st.bar_chart(chart_df, use_container_width=True)

                            if non_numeric_res_cols and numeric_res_cols:
                                with st.expander("🎨 Enhanced Plotly Chart", expanded=False):
                                    fig = px.bar(
                                        result_df.head(30),
                                        x=non_numeric_res_cols[0],
                                        y=numeric_res_cols[0],
                                        title=f"{numeric_res_cols[0]} by {non_numeric_res_cols[0]}",
                                        color_discrete_sequence=["#6C63FF"],
                                        template="plotly_dark",
                                    )
                                    fig.update_layout(
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        margin=dict(l=20, r=20, t=50, b=20),
                                        xaxis_tickangle=-30,
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("ℹ️ No numeric columns in the result – displaying table only (no chart).")

                except Exception as e:
                    st.error(f"❌ SQL Error: {e}")
                    st.caption("Tip: Make sure your column and table names match exactly. Check the table list above.")

            elif run_btn:
                st.warning("⚠️ Please enter a SQL query first.")


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;opacity:0.4;font-size:0.78rem">'
    "Built with ❤️ for GDG Cloud Delhi Hackathon · Powered by Google Gemini & Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
