# 🧠 SchemaSense AI — Intelligent Data Dictionary Agent

> Built for the **GDG Cloud Delhi Hackathon** · Powered by **Google Gemini** & **Streamlit**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔑 **API Key Rotation** | Accepts a list of Gemini keys; auto-rotates on 429 quota errors |
| 📦 **Multi-Dataset SQLite** | Upload multiple CSVs → loaded into in-memory SQLite for JOIN-aware AI queries |
| 💬 **Data Steward Chat** | Multi-turn AI chat powered by Gemini to describe schemas, write SQL, find anomalies |
| 📊 **Visual Analytics Engine** | Auto-extracts SQL from AI responses → runs against SQLite → bar chart |
| 📈 **Column Profiler** | Histograms, box plots, and value-count charts for uploaded datasets |
| 🛡️ **Quality Guard** | One-click AI data quality audit with severity ratings |
| 📥 **Data Dictionary Export** | Download auto-generated JSON data dictionary |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/schemasense-ai.git
cd schemasense-ai
pip install -r requirements.txt
```

### 2. Configure API keys

Create `.streamlit/secrets.toml` (**do not commit this file**):

```toml
# Single key
GEMINI_API_KEY = "AIza..."

# Or multiple keys for automatic quota-rotation fallback
GEMINI_KEYS = ["AIza...key1", "AIza...key2", "AIza...key3"]
```

Get your Gemini API keys at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

### 3. Run

```bash
streamlit run app.py
```

---

## 🏗️ Architecture

```
app.py
├── API Key Fallback Layer   — generate_with_fallback() with key rotation
├── SQLite Engine            — load_csvs_into_sqlite() + schema injection
├── Chat Interface           — multi-turn Gemini conversation
├── Visual Analytics Tab     — SQL extraction regex + pd.read_sql + st.bar_chart
└── Column Profiler          — Plotly histograms, box plots, bar charts
```

---

## 📁 Project Structure

```
nexus/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes secrets.toml
├── .streamlit/
│   ├── config.toml           # Streamlit theme (dark mode)
│   └── secrets.toml          # ⚠️ NOT committed — add your keys here
└── README.md
```

---

## 🔧 Requirements

- Python 3.10+
- Streamlit ≥ 1.30
- google-generativeai ≥ 0.4
- pandas ≥ 2.0
- plotly ≥ 5.18
- `sqlite3` (Python stdlib — no install needed)

---

## 📝 License

MIT — free to use, modify, and distribute.
