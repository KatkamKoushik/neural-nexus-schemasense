# 🧠 SchemaSense AI — Intelligent Data Dictionary & Enterprise Data Agent

> **Built for Enterprise Relational Databases** · Powered by **Google Gemini**, **SQLAlchemy**, and **Streamlit**

[![SchemaSense CI](https://github.com/KatkamKoushik/neural-nexus-schemasense/actions/workflows/ci.yml/badge.svg)](https://github.com/KatkamKoushik/neural-nexus-schemasense/actions/workflows/ci.yml)
[![Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Security: SQLGuard AST](https://img.shields.io/badge/Security-SQLGuard%20AST-10B981?logo=shield)](docs/security/threat-model.md)
[![Tests: Pytest](https://img.shields.io/badge/Tests-119%20Passed-brightgreen?logo=pytest)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-89%25-brightgreen)](tests/)
[![Supply Chain: CycloneDX](https://img.shields.io/badge/SBOM-CycloneDX%20v1.5-blue?logo=owasp)](sbom.cyclonedx.json)
[![Secrets: Scanned](https://img.shields.io/badge/Secret%20Scan-detect--secrets-success)](.secrets.baseline)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise AI Data Steward and Schema Intelligence Agent that connects to relational databases (**PostgreSQL**, **MySQL**, **SQLite**) to **generate interactive data dictionaries, enforce AST-based SQL guardrails, audit data quality, bound database concurrency, and ground queries in business rules**.

---

## 🌟 Highlights

- **AST-Based SQLGuard**: Dialect-aware parser powered by `sqlglot`. Strictly enforces single read-only `SELECT` statements, validates tables against an allowlist, blocks destructive DDL/DML, and enforces database-level statement timeouts and row caps (1,000 rows).
- **Least-Privilege Enforcement**: Active runtime catalog inspection identifying superuser or admin credentials without relying on naive username heuristics. Includes interactive UI policy toggles to reject privileged connections.
- **Supply-Chain Security Gates**: Continuous vulnerability auditing (`pip-audit`), secret scanning (`detect-secrets`), reproducible builds (`constraints.txt`), and CycloneDX SBOM release artifacts.
- **Deep Schema Intelligence**: Extracts tables, columns, primary keys, unique constraints, foreign keys, and indexes. Export complete data dictionaries in **JSON** or **Markdown** with one click.
- **Data Quality & Profiling**: Calculates null percentages, distinct counts, candidate primary keys, duplicate row rates, numeric outlier detection (IQR), and computes a composite Data Quality Score (0–100%).
- **Bounded Concurrency & Pool Management**: Bounded SQLAlchemy `QueuePool` with session timeouts, client-side concurrency semaphores, and safe pool disposal on disconnect.

---

## 📋 Supported Environments

| Component | Supported Versions | Notes |
|---|---|---|
| **Python** | 3.10, 3.11, 3.12 | Verified across automated GitHub Actions matrix. |
| **PostgreSQL** | 14, 15, 16 | Tested against `postgres:16-alpine`. Uses `psycopg2-binary`. |
| **MySQL** | 8.0, 8.4 LTS | Tested against `mysql:8.4`. Uses `pymysql` with `utf8mb4`. |
| **SQLite** | 3.x | In-memory and local file testing. |

---

## 🚀 Quick Start

### 1. Clone & Reproducible Install

```bash
git clone https://github.com/KatkamKoushik/neural-nexus-schemasense.git
cd neural-nexus-schemasense

# For standard installation:
pip install -r requirements.txt -r requirements-dev.txt

# For reproducible, constraint-locked pilot deployment:
pip install -r requirements.txt -c constraints.txt
```

### 2. Run Test Suite & Coverage

Verify the security boundaries and functional logic across the 119 unit, adversarial, integration, and UI tests:

```bash
# Fast unit tests (SQLite isolated):
pytest -v --ignore=tests/test_live_integration.py --cov=schemasense

# Full live suite (requires Docker test containers):
docker compose -f docker-compose.test.yml up -d --wait
pytest -v --cov=schemasense --cov-report=term-missing
```

### 3. Supply-Chain & Security Audit Commands

```bash
# 1. Dependency vulnerability audit
pip-audit -r requirements.txt

# 2. Secret scanning against baseline
detect-secrets scan --baseline .secrets.baseline --exclude-files "(\.git|__pycache__|\.venv|venv|\.secrets\.baseline)"

# 3. Generate CycloneDX SBOM
cyclonedx-py requirements constraints.txt -o sbom.cyclonedx.json

# 4. Linting check
ruff check schemasense/ tests/
```

### 4. Configure API Keys

Create `.streamlit/secrets.toml` (already ignored in `.gitignore`):

```toml
# Option A — Single key
GEMINI_API_KEY = "your-gemini-api-key"

# Option B — Multiple keys for automatic quota failover
GEMINI_KEYS = ["key-1", "key-2", "key-3"]
```

Or configure via environment variable:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

### 5. Run Application

```bash
streamlit run app.py
```

---

## 🏗️ Architecture

```
neural-nexus-schemasense/
├── schemasense/                    # Core Python Backend Package
│   ├── __init__.py
│   ├── ai_agent.py                 # Backend-neutral AI agent & session-isolated caching
│   ├── runtime/                    # Inference Runtime Subsystem
│   │   ├── backends/               # Base contract, Gemini, and llama.cpp backends
│   │   ├── parsers.py              # Zero-dependency GGUF metadata parser
│   │   └── registry.py             # Local model discovery and integrity verification
│   ├── connection.py               # Safe URL creation, engine lifecycle, least-privilege inspection, pool disposal
│   ├── sql_guard.py                # AST parser (sqlglot), table allowlists, dialect timeouts & row caps
│   ├── schema_inspector.py         # PKs, FKs, constraints, indexes & data dictionary exports
│   ├── profiler.py                 # Data quality metrics, outlier detection, scoring formula
│   └── rules_engine.py             # PDF business-rule ingestion (pypdf) & prompt grounding
├── tests/                          # 355 Unit, Adversarial, Integration, and AppTest tests
│   ├── conftest.py                 # SQLite test fixtures & mock schemas
│   ├── test_sql_guard.py           # 50 adversarial SQL, dialect, and AST tests
│   ├── test_connection.py          # 12 connection security, pool & concurrency tests
│   ├── test_inspector.py           # Schema inspection, diffing, and export tests
│   ├── test_profiler.py            # Quality metrics and outlier detection tests
│   ├── test_rules_engine.py        # PDF business rule parsing tests
│   ├── test_live_integration.py    # 43 live PostgreSQL 16 & MySQL 8.4 container integration tests
│   └── test_ui_apptest.py          # 4 Streamlit headless AppTest integration tests
├── .github/
│   ├── workflows/
│   │   └── ci.yml                  # Automated CI matrix, security audit, and coverage gates
│   └── dependabot.yml              # Dependabot config for pip & GitHub Actions
├── docs/
│   ├── architecture/
│   │   └── target-architecture.md  # Architectural blueprint & module boundaries
│   ├── security/
│   │   ├── threat-model.md         # Threat model & attack mitigations
│   │   ├── database-permissions.md # DB engine permission setup & dialect limitations
│   │   └── supply-chain.md         # Supply-chain security policy, SBOM & triage SLA
│   ├── deployment/
│   │   └── readiness-checklist.md  # Internal pilot verification checklist
│   └── testing/
│       └── verification-plan.md    # Test strategy, mutation testing evaluation & gates
├── constraints.txt                 # Pinned dependencies for reproducible pilot builds
├── requirements.txt                # Runtime application dependencies
├── requirements-dev.txt            # Development, CI, and audit tooling
├── sbom.cyclonedx.json             # CycloneDX Software Bill of Materials (SBOM)
├── .secrets.baseline               # detect-secrets audited baseline
├── docker-compose.test.yml         # Disposable PostgreSQL 16 & MySQL 8.4 test containers
└── app.py                          # Streamlit UI consuming the schemasense package
```

---

## 🛡️ Security Documentation

- [Threat Model & Attack Mitigations](docs/security/threat-model.md)
- [Database Permissions & Engine Dialect Limitations](docs/security/database-permissions.md)
- [Supply-Chain Security & Vulnerability Triage](docs/security/supply-chain.md)
- [Internal Pilot Readiness Checklist](docs/deployment/readiness-checklist.md)
- [Verification & Quality Gate Plan](docs/testing/verification-plan.md)

---

## 📝 License

MIT — free to use, modify, and distribute.
