# SchemaSense AI — Target Architecture

## System Overview

SchemaSense AI is designed around a decoupled, security-first architecture. The presentation layer (Streamlit) interacts exclusively with a standalone, pure-Python domain package (`schemasense`).

```
┌────────────────────────────────────────────────────────┐
│               Streamlit Presentation Layer            │
│  (app.py - State, Glassmorphism UI, Reactive Sessions) │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│               SchemaSense Backend Engine               │
│                                                        │
│  ┌───────────────────────┐  ┌───────────────────────┐  │
│  │   connection.py       │  │    sql_guard.py       │  │
│  │ - Safe URL Creation   │  │ - AST Parsing         │  │
│  │ - Engine Lifecycle    │  │ - Allowlist Checking  │  │
│  │ - Pool Disposal       │  │ - Bounded Execution   │  │
│  └───────────────────────┘  └───────────────────────┘  │
│  ┌───────────────────────┐  ┌───────────────────────┐  │
│  │ schema_inspector.py   │  │     profiler.py       │  │
│  │ - PK / FK / Indexes   │  │ - Statistical Audits  │  │
│  │ - Schema Diffing      │  │ - Outlier Detection   │  │
│  │ - JSON & MD Exports   │  │ - Quality Score (0-100│  │
│  └───────────────────────┘  └───────────────────────┘  │
│  ┌───────────────────────┐                             │
│  │    rules_engine.py    │                             │
│  │ - pypdf Rule Parser   │                             │
│  │ - Prompt Grounding    │                             │
│  └───────────────────────┘                             │
└───────────────────────────┬────────────────────────────┘
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
   ┌───────────────────────┐ ┌───────────────────────┐
   │ Enterprise Relational │ │   Google Gemini API   │
   │ Databases (PG, MySQL) │ │ (Schema Grounded LLM) │
   └───────────────────────┘ └───────────────────────┘
```

## Core Modules & Boundaries

1. **`schemasense.sql_guard`**:
   - Parses incoming SQL statements using `sqlglot`.
   - Enforces single-statement constraints.
   - Enforces read-only AST invariants.
   - Extracts referenced database tables and cross-references against allowlists.
   - Enforces statement timeouts (5s) and row limits (1,000 rows).

2. **`schemasense.schema_inspector`**:
   - Inspects structural metadata (primary keys, foreign keys, unique constraints, indexes).
   - Generates serializable data dictionaries in JSON, Markdown, and XML-delimited prompt contexts.
   - Computes diffs between two schema snapshots.

3. **`schemasense.profiler`**:
   - Computes statistical metrics (null percentage, distinct counts, uniqueness ratios).
   - Detects candidate primary keys.
   - Identifies numeric outliers using the Interquartile Range (IQR) method.
   - Calculates composite Data Quality Scores.

4. **`schemasense.connection`**:
   - Constructs SQLAlchemy URLs without string formatting risks.
   - Handles connection health checks (`pool_pre_ping=True`).
   - Cleans up connection pools on reconnect to prevent resource leaks.

5. **`schemasense.rules_engine`**:
   - Ingests PDF business rules via `pypdf`.
   - Formats extracted rules into XML-tagged prompt sections for safe LLM grounding.
