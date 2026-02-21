import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import sqlite3
import json
import re

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="SchemaSense AI",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------------------------
# SAFE GLASS UI (NO INTERNAL STREAMLIT OVERRIDE)
# ------------------------------------------------
st.markdown("""
<style>
body { font-family: Inter, sans-serif; }

.hero {
    background: linear-gradient(135deg,#6C63FF,#3B82F6,#06B6D4);
    padding:2rem;
    border-radius:16px;
    color:white;
    margin-bottom:1.5rem;
}

.glass {
    background: rgba(255,255,255,0.05);
    border-radius:14px;
    padding:1.2rem;
    margin-bottom:1rem;
    border:1px solid rgba(255,255,255,0.08);
}

.metric-grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
    gap:12px;
}

.metric-box {
    background:rgba(255,255,255,0.04);
    border-radius:12px;
    padding:1rem;
    text-align:center;
}

.metric-box h4 {
    margin:0;
    font-size:0.8rem;
    opacity:0.6;
}

.metric-box h2 {
    margin:0.3rem 0 0 0;
    color:#6C63FF;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HERO
# ------------------------------------------------
with st.container():
    st.markdown("""
    <div class="hero">
        <h1>🧠 SchemaSense AI</h1>
        <p>Intelligent Data Dictionary Agent · Gemini Powered</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# SESSION STATE
# ------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sqlite_conn" not in st.session_state:
    st.session_state.sqlite_conn = None

if "sqlite_schema" not in st.session_state:
    st.session_state.sqlite_schema = {}

if "last_sql" not in st.session_state:
    st.session_state.last_sql = None

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
with st.sidebar:
    st.header("📂 Data Source")

    uploaded_files = st.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        conn = sqlite3.connect(":memory:")
        schema = {}

        for f in uploaded_files:
            name = re.sub(r"\W+", "_", f.name.split(".")[0])
            df = pd.read_csv(f)
            df.to_sql(name, conn, index=False, if_exists="replace")
            schema[name] = list(df.columns)

        st.session_state.sqlite_conn = conn
        st.session_state.sqlite_schema = schema
        st.success(f"{len(schema)} table(s) loaded")

    st.divider()

    st.subheader("⚙️ AI Settings")
    api_key = st.text_input("Gemini API Key", type="password")

    if api_key:
        genai.configure(api_key=api_key)

# ------------------------------------------------
# TABS
# ------------------------------------------------
tab_chat, tab_schema, tab_profile, tab_visual = st.tabs(
    ["💬 Chat", "📊 Schema", "📈 Column Profiler", "📊 Visual Analytics"]
)

# ------------------------------------------------
# TAB 1 – CHAT
# ------------------------------------------------
with tab_chat:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your data..."):
        st.session_state.messages.append({"role":"user","content":prompt})

        with st.chat_message("assistant"):
            if api_key:
                model = genai.GenerativeModel("gemini-2.0-flash")
                response = model.generate_content(prompt)
                st.markdown(response.text)
                st.session_state.messages.append({"role":"assistant","content":response.text})

                sql_match = re.search(r"(SELECT .*?;)", response.text, re.I | re.S)
                if sql_match:
                    st.session_state.last_sql = sql_match.group(1)
            else:
                st.warning("Enter API key in sidebar")

# ------------------------------------------------
# TAB 2 – SCHEMA VIEW
# ------------------------------------------------
with tab_schema:

    schema = st.session_state.sqlite_schema

    if schema:
        for t, cols in schema.items():
            with st.expander(f"📋 {t}"):
                st.write(", ".join(cols))
    else:
        st.info("Upload dataset to view schema")

# ------------------------------------------------
# TAB 3 – COLUMN PROFILER
# ------------------------------------------------
with tab_profile:

    conn = st.session_state.sqlite_conn

    if conn:
        tables = list(st.session_state.sqlite_schema.keys())
        table = st.selectbox("Select Table", tables)

        df = pd.read_sql(f"SELECT * FROM {table}", conn)

        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-box">
            <h4>Rows</h4><h2>{len(df)}</h2>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        numeric_cols = df.select_dtypes("number").columns

        for col in numeric_cols[:3]:
            fig = px.histogram(df, x=col)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Upload dataset first")

# ------------------------------------------------
# TAB 4 – VISUAL ANALYTICS
# ------------------------------------------------
with tab_visual:

    conn = st.session_state.sqlite_conn

    if not conn:
        st.info("Upload dataset to enable SQL engine")
    else:
        default_sql = st.session_state.last_sql or ""

        sql_input = st.text_area(
            "SQL Query (SQLite)",
            value=default_sql,
            height=150
        )

        if st.button("Run Query"):
            try:
                df = pd.read_sql(sql_input, conn)
                st.dataframe(df, use_container_width=True)

                num_cols = df.select_dtypes("number").columns
                cat_cols = df.select_dtypes(exclude="number").columns

                if len(num_cols) > 0:
                    if len(cat_cols) > 0:
                        fig = px.bar(df.head(30), x=cat_cols[0], y=num_cols[0])
                    else:
                        fig = px.bar(df.head(30), y=num_cols[0])
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"SQL Error: {e}")
