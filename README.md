# 🧠 SchemaSense AI — Intelligent Data Dictionary Agent

> **Built for the GDG Cloud Delhi Hackathon** · Powered by **Google Gemini** & **Streamlit**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-AI-4285F4?logo=google&logoColor=white)](https://aistudio.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An autonomous AI Data Steward that connects to enterprise databases and uploaded CSV files to **generate interactive data dictionaries, visualize schemas, run SQL analytics, and perform AI-powered data quality audits** — all in a sleek dark-mode Streamlit interface.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔑 **API Key Rotation** | Accepts a list of Gemini API keys; auto-rotates on 429 quota errors so the app never goes down |
| 📦 **Multi-Dataset SQLite** | Upload multiple CSVs → loaded into in-memory SQLite for JOIN-aware AI queries |
| 💬 **Data Steward Chat** | Multi-turn AI chat powered by Gemini to describe schemas, write SQL, explain relationships, and find anomalies |
| 📊 **Visual Analytics Engine** | Auto-extracts SQL from AI responses → executes against SQLite or PostgreSQL → renders interactive bar charts |
| 📈 **Column Profiler** | Histograms, box plots, and value-count charts for every column in your dataset |
| 🛡️ **Quality Guard** | One-click AI data quality audit with severity ratings (🔴🟡🟢) |
| 📥 **Data Dictionary Export** | Download an auto-generated JSON data dictionary at any time |
| 📄 **Business Rules (PDF)** | Upload a PDF of business rules — ingested via PyPDF2 and injected into the AI context |
| 🔌 **PostgreSQL / Cloud SQL** | Connect to a live PostgreSQL database (e.g. Olist on Google Cloud SQL) for real enterprise queries |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/KatkamKoushik/neural-nexus-schemasense.git
cd neural-nexus-schemasense
pip install -r requirements.txt
```

### 2. Configure API keys

Create `.streamlit/secrets.toml` (**do not commit this file — it is already in `.gitignore`**):

```toml
# Option A — Single key
GEMINI_API_KEY = "AIza..."

# Option B — Multiple keys for automatic quota-rotation failover
GEMINI_KEYS = ["AIza...key1", "AIza...key2", "AIza...key3"]

# Option C — Live PostgreSQL connection (optional)
[connections.postgresql]
dialect  = "postgresql"
host     = "YOUR_CLOUD_SQL_PUBLIC_IP"
port     = 5432
database = "olist"
username = "postgres"
password = "YOUR_PASSWORD"
```

Get free Gemini API keys at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

### 3. Run

```bash
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Fork this repo to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your fork.
3. Set **Main file path** to `app.py`.
4. Under **Advanced settings → Secrets**, paste your `secrets.toml` content.
5. Click **Deploy** — done!

> **Note:** The PostgreSQL connection is optional. The app fully works with CSV uploads only.

---

## 🏗️ Architecture

```
app.py
├── API Key Fallback Layer    — generate_with_fallback() with key rotation
├── SQLite Engine             — load_csvs_into_sqlite() + schema injection
├── PostgreSQL Connector      — st.connection("postgresql") with live ping
├── Chat Interface            — multi-turn Gemini conversation
├── Visual Analytics Tab      — SQL extraction regex + pd.read_sql + st.bar_chart
├── Column Profiler           — Plotly histograms, box plots, bar charts
├── Quality Guard             — single-call AI quality scan with severity emojis
└── Business Rules RAG        — PyPDF2 text extraction injected into system prompt
```

---

## 📁 Project Structure

```
neural-nexus-schemasense/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes secrets.toml & caches
├── .streamlit/
│   ├── config.toml           # Dark-mode theme & headless config
│   └── secrets.toml          # ⚠️ NOT committed — add your keys here
└── README.md
```

---

## 🔧 Requirements

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.30 | Web UI framework |
| `google-generativeai` | ≥ 0.4 | Gemini AI API |
| `pandas` | ≥ 2.0 | Data manipulation |
| `plotly` | ≥ 5.18 | Interactive charts |
| `sqlalchemy` | ≥ 2.0 | SQL toolkit |
| `PyPDF2` | ≥ 3.0 | PDF text extraction |
| `openpyxl` | ≥ 3.1 | Excel file support |
| `sqlite3` | stdlib | In-memory SQL engine (no install needed) |

> **PostgreSQL support:** If you want to connect to a live PostgreSQL database, install `psycopg2-binary` separately:
> ```bash
> pip install psycopg2-binary
> ```

---

## 🤖 Supported AI Models

Switch between these models from the sidebar at any time:

| Model | Best For |
|---|---|
| `gemini-2.0-flash` | Fast responses · **Recommended default** |
| `gemini-2.0-flash-lite` | Highest throughput · Low quota usage |
| `gemini-1.5-flash` | Stable, battle-tested |
| `gemini-1.5-pro` | Complex multi-step reasoning |

---

## 📝 License

MIT — free to use, modify, and distribute.
