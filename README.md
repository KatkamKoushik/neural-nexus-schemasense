# SchemaSense AI — Intelligent Data Dictionary & Enterprise Data Agent

> **Built for Enterprise Relational Databases** | Dual Inference Engines (**Google Gemini** & **llama.cpp**) | **SQLAlchemy** | **Streamlit**

[![SchemaSense CI](https://github.com/KatkamKoushik/neural-nexus-schemasense/actions/workflows/ci.yml/badge.svg)](https://github.com/KatkamKoushik/neural-nexus-schemasense/actions/workflows/ci.yml)
[![Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Security: SQLGuard AST](https://img.shields.io/badge/Security-SQLGuard%20AST-10B981?logo=shield)](docs/security/threat-model.md)
[![Tests: Pytest](https://img.shields.io/badge/Tests-355%20Tests%20%7C%20313%20Passed-brightgreen?logo=pytest)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-89%25-brightgreen)](tests/)
[![Supply Chain: CycloneDX](https://img.shields.io/badge/SBOM-CycloneDX%20v1.5-blue?logo=owasp)](sbom.cyclonedx.json)
[![Secrets: Scanned](https://img.shields.io/badge/Secret%20Scan-detect--secrets-success)](.secrets.baseline)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise AI Data Steward and Schema Intelligence Agent that connects to relational databases (**PostgreSQL**, **MySQL**, **SQLite**) to **generate interactive data dictionaries, enforce AST-based SQL guardrails, audit data quality, bound database concurrency, ground queries in business rules, and execute structured AI inference locally or in the cloud**.

---

## Highlights

- **Dual-Inference AI Engine**: Backend-neutral architecture with explicit provider selection. Supports **Google Gemini** (cloud API with multi-key rotation and rate-limit failover) and **llama.cpp** (local in-process GGUF inference with lazy weight allocation, native grammar-constrained decoding, and zero cloud fallback).
- **AST-Based SQLGuard**: Dialect-aware parser powered by `sqlglot`. Strictly enforces single read-only `SELECT` statements, validates tables against an active allowlist, blocks destructive DDL/DML, and enforces database-level statement timeouts and row caps (1,000 rows).
- **Least-Privilege Enforcement**: Active runtime catalog inspection identifying superuser or admin credentials without relying on naive username heuristics. Includes interactive UI policy toggles to reject privileged connections.
- **Deep Schema Intelligence**: Extracts tables, columns, primary keys, unique constraints, foreign keys, and indexes. Export complete data dictionaries in **JSON** or **Markdown** with one click.
- **Data Quality & Profiling**: Calculates null percentages, distinct counts, candidate primary keys, duplicate row rates, numeric outlier detection (IQR), and computes a composite Data Quality Score (0–100%).
- **Bounded Concurrency & Pool Management**: Bounded SQLAlchemy `QueuePool` with session timeouts, client-side concurrency semaphores, and safe pool disposal on disconnect.
- **Supply-Chain Security Gates**: Continuous vulnerability auditing (`pip-audit`), secret scanning (`detect-secrets`), reproducible builds (`constraints.txt`), and CycloneDX SBOM release artifacts.

---

## Supported Environments

| Component | Supported Versions | Notes |
|---|---|---|
| **Python** | 3.10, 3.11, 3.12 | Verified across automated GitHub Actions matrix. |
| **PostgreSQL** | 14, 15, 16 | Tested against `postgres:16-alpine`. Uses `psycopg2-binary`. |
| **MySQL** | 8.0, 8.4 LTS | Tested against `mysql:8.4`. Uses `pymysql` with `utf8mb4`. |
| **SQLite** | 3.x | In-memory and local file testing. |
| **Cloud AI Engine** | Google Gemini | Uses `google-genai` SDK (`gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash`). |
| **Local AI Engine** | llama.cpp (GGUF) | Optional in-process engine via `llama-cpp-python` (`requirements-local.txt`). |

---

## Quick Start

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/KatkamKoushik/neural-nexus-schemasense.git
cd neural-nexus-schemasense

# Standard installation (cloud Gemini inference + database drivers):
pip install -r requirements.txt -r requirements-dev.txt

# Optional: Add local GGUF inference engine:
pip install -r requirements-local.txt

# For reproducible, constraint-locked deployment:
pip install -r requirements.txt -c constraints.txt
```

### 2. Run Test Suite & Coverage

Verify the security boundaries, backend contracts, and analytical logic across the 355 tests:

```bash
# Fast isolated tests (SQLite in-memory, mocks, and syntax):
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

# 4. Code quality and formatting checks
python -m ruff check app.py schemasense/ tests/
python -m ruff format --check app.py schemasense/ tests/
```

### 4. Configure Inference Providers

#### Option A: Google Gemini (Cloud)
Configure credentials via `.streamlit/secrets.toml` (ignored by Git):

```toml
# Single key:
GEMINI_API_KEY = ""  # Enter your Google Gemini API key here

# Multiple keys for automated round-robin quota failover:
GEMINI_KEYS = ["", ""]  # Add multiple keys to rotate across quotas
```

Or configure via environment variable:
```bash
export GEMINI_API_KEY="<api-key>"
```

#### Option B: llama.cpp (Local GGUF)
1. Install the local inference engine: `pip install -r requirements-local.txt`
2. Launch SchemaSense and select **llama.cpp (Local GGUF)** in the sidebar.
3. Provide the absolute or relative path to your local `.gguf` model file.
4. Inference runs strictly in-process on your local machine; zero prompt data leaves the host.

### 5. Run Application

```bash
streamlit run app.py
```

---

## Architecture

```
neural-nexus-schemasense/
├── schemasense/                    # Core Python Backend Package
│   ├── __init__.py
│   ├── ai_agent.py                 # Backend-neutral AI agent & session-isolated caching
│   ├── runtime/                    # Inference Runtime Subsystem
│   │   ├── backends/               # Base contract, Gemini, and llama.cpp backends
│   │   ├── parsers.py              # Zero-dependency GGUF metadata parser
│   │   ├── registry.py             # Local model discovery and integrity verification
│   │   ├── hardware.py             # Memory and accelerator capability detection
│   │   └── compatibility.py        # Model compatibility and memory estimator
│   ├── connection.py               # Safe URL creation, engine lifecycle, least-privilege inspection, pool disposal
│   ├── sql_guard.py                # AST parser (sqlglot), table allowlists, dialect timeouts & row caps
│   ├── schema_inspector.py         # PKs, FKs, constraints, indexes & data dictionary exports
│   ├── profiler.py                 # Data quality metrics, outlier detection, scoring formula
│   ├── rules_engine.py             # PDF business-rule ingestion (pypdf) & prompt grounding
│   └── benchmark.py                # Synthetic workload & latency benchmark harnesses
├── tests/                          # 355 Unit, Adversarial, Integration, and AppTest tests
│   ├── conftest.py                 # SQLite test fixtures & mock schemas
│   ├── test_backends_base.py       # Neutral backend contract tests
│   ├── test_backends_gemini.py     # Gemini backend & key rotation tests
│   ├── test_backends_llama_cpp.py  # llama.cpp local backend & lazy-loading tests
│   ├── test_backend_selection.py   # Explicit backend selection & isolation tests
│   ├── test_ai_agent.py            # AI agent structured output & fallback tests
│   ├── test_sql_guard.py           # 50 adversarial SQL, dialect, and AST tests
│   ├── test_connection.py          # 12 connection security, pool & concurrency tests
│   ├── test_inspector.py           # Schema inspection, diffing, and export tests
│   ├── test_profiler.py            # Quality metrics and outlier detection tests
│   ├── test_rules_engine.py        # PDF business rule parsing tests
│   ├── test_runtime_parsers.py     # GGUF parser tests
│   ├── test_runtime_registry.py    # Model discovery & SHA256 integrity tests
│   ├── test_runtime_hardware.py    # Hardware detection tests
│   ├── test_runtime_compatibility.py # Runtime compatibility tests
│   ├── test_dependency_hygiene.py  # Dependency hygiene & import boundary tests
│   ├── test_live_integration.py    # Live PostgreSQL 16 & MySQL 8.4 container tests
│   └── test_ui_apptest.py          # Streamlit headless AppTest integration tests
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
├── requirements-local.txt          # Optional local llama.cpp inference engine
├── sbom.cyclonedx.json             # CycloneDX Software Bill of Materials (SBOM)
├── .secrets.baseline               # detect-secrets audited baseline
├── docker-compose.test.yml         # Disposable PostgreSQL 16 & MySQL 8.4 test containers
└── app.py                          # Streamlit UI consuming the schemasense package
```

---

## Security Documentation

- [Threat Model & Attack Mitigations](docs/security/threat-model.md)
- [Database Permissions & Engine Dialect Limitations](docs/security/database-permissions.md)
- [Supply-Chain Security & Vulnerability Triage](docs/security/supply-chain.md)
- [Internal Pilot Readiness Checklist](docs/deployment/readiness-checklist.md)
- [Verification & Quality Gate Plan](docs/testing/verification-plan.md)

---

## License

MIT — free to use, modify, and distribute.
