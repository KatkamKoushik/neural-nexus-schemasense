import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import json
import sqlite3
import re
from io import StringIO

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="SchemaSense AI",
    page_icon="🧠",
    layout="wide",
)

# -------------------------------------------------
# CLEAN SAFE CSS (NO INTERNAL OVERRIDES)
# -------------------------------------------------
st.markdown("""
<style>
body { font-family: Inter, sans-serif; }

.hero {
    background: linear-gradient(135deg,#6C63FF,#3B82F6,#06B6D4);
    padding: 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 1.2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}

.metric-grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
    gap:12px;
}
.metric-box {
    background:rgba(255,255,255,0.04);
    padding:1rem;
    border-radius:12px;
    text-align:center;
}
.metric-box h4 { margin:0; font-size:0.8rem; opacity:0.6; }
.metric-box h2 { margin:0.3rem 0 0 0; color:#6C63FF; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HERO SECTION
# -------------------------------------------------
with st.container():
    st.markdown("""
    <div class="hero">
        <h1>🧠 SchemaSense AI</h1>
        <p>Intelligent Data Dictionary Agent</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sqlite_conn" not in st.session_state:
    st.session_state.sqlite_conn = None
if "sqlite_schema" not in st.session_state:
    st.session_state.sqlite_schema = {}
if "last_sql" not in st.session_state:
    st.session_state.last_sql = None

# -------------------------------------------------
# SIDEBAR DATA UPLOAD
# -------------------------------------------------
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
        st.success(f"{len(schema)} table(s) loaded.")

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2 = st.tabs(["💬 Chat", "📊 Visual Analytics"])

# -------------------------------------------------
# TAB 1 – CHAT
# -------------------------------------------------
with tab1:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask something about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            st.markdown("⚠️ AI disabled in debug version.")

# -------------------------------------------------
# TAB 2 – VISUAL ANALYTICS (STABLE VERSION)
# -------------------------------------------------
with tab2:

    if st.session_state.sqlite_conn is None:
        st.info("Upload CSV files in sidebar to enable SQL engine.")
    else:
        schema = st.session_state.sqlite_schema

        with st.expander("📋 Available Tables", expanded=False):
            for t, cols in schema.items():
                st.markdown(f"**{t}** — {', '.join(cols)}")

        default_sql = st.session_state.last_sql or ""

        sql_input = st.text_area(
            "SQL Query (SQLite)",
            value=default_sql,
            height=150
        )

        col1, col2 = st.columns([1,4])
        with col1:
            run = st.button("Run Query", type="primary")
        with col2:
            if st.button("Clear"):
                st.session_state.last_sql = None
                st.rerun()

        if run and sql_input.strip():
            try:
                df = pd.read_sql(sql_input, st.session_state.sqlite_conn)

                st.success(f"{len(df)} rows returned")
                st.dataframe(df, use_container_width=True)

                if not df.empty:
                    num_cols = df.select_dtypes("number").columns.tolist()
                    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

                    if num_cols:
                        st.subheader("Auto Chart")

                        if cat_cols:
                            chart_df = df.set_index(cat_cols[0])[num_cols[:3]]
                        else:
                            chart_df = df[num_cols[:3]]

                        chart_df = chart_df.head(30)
                        st.bar_chart(chart_df, use_container_width=True)

                        if cat_cols:
                            fig = px.bar(
                                df.head(30),
                                x=cat_cols[0],
                                y=num_cols[0],
                                title=f"{num_cols[0]} by {cat_cols[0]}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No numeric columns for charting.")

            except Exception as e:
                st.error(f"SQL Error: {e}")
