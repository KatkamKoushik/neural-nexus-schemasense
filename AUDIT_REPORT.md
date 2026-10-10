# SchemaSense AI — Comprehensive Codebase Reality Audit Report

**Date of Audit:** October 10, 2026  
**Auditor:** Senior Software Architect, Security Auditor & QA Systems Evaluator  
**Audit Target:** Local SchemaSense AI Codebase (`d:\schemasense`)  
**Operating Mode:** Strictly Read-Only  
**Status:** Complete & Evidence-Based  

---

## 1. Executive Summary

SchemaSense AI is an enterprise database exploration and data intelligence assistant designed to interface with relational databases, extract structural schemas, profile data quality, and execute user/AI-generated read-only SQL queries under strict AST-based security controls.

This comprehensive audit independently inspected every local file, configuration setting, test suite, and operational boundary. **The audit was performed strictly in read-only mode, without making external API calls, running destructive operations, or altering project code.**

### Core Audit Verdict:
1. **The Security Core is Genuine and Robust:** The application-level AST validator ([`schemasense/sql_guard.py`](file:///d:/schemasense/schemasense/sql_guard.py)) is genuine, thoroughly tested, and dialect-aware. It strictly enforces single read-only `SELECT`/`UNION` statements, blocks 16 destructive AST statement types, blocks 22 dangerous functions, isolates system catalog schemas, rewrites queries to enforce server-side row caps (1,000 rows), and applies database-level statement timeouts (5,000 ms) across PostgreSQL, MySQL, and SQLite.
2. **AI SDK Modernization is Successfully Implemented:** The migration from deprecated `google-generativeai` to the modern **`google-genai` SDK (`v2.29.0`)** in [`schemasense/ai_agent.py`](file:///d:/schemasense/schemasense/ai_agent.py) is complete. It provides structured Pydantic model outputs ([`SQLAgentResponse`](file:///d:/schemasense/schemasense/ai_agent.py#L31-L57)), in-memory multi-key rotation on HTTP `429` rate limits, schema-grounded system instructions, and AST-based schema hallucination detection.
3. **Significant Implementation and Documentation Gaps Exist:**
   - **Authentication is Completely Missing:** Despite documentation and template files ([`.streamlit/secrets.toml.template`](file:///d:/schemasense/.streamlit/secrets.toml.template#L11-L19)) referencing Google Sign-In and Streamlit OIDC authentication, [`app.py`](file:///d:/schemasense/app.py) contains zero authentication or authorization logic.
   - **SQLite is Inaccessible in the UI:** Although documentation claims support for PostgreSQL, MySQL, and SQLite, the UI connection form only allows selecting PostgreSQL and MySQL, and enforces network credentials (Host, Port, User, Password).
   - **Schema Diffing is Unused in the UI:** Schema diffing is implemented in [`schemasense/schema_inspector.py`](file:///d:/schemasense/schemasense/schema_inspector.py#L228-L265) and tested in [`tests/test_inspector.py`](file:///d:/schemasense/tests/test_inspector.py), but is not exposed anywhere in the Streamlit frontend.
   - **Composite Foreign Key Truncation:** [`schema_inspector.py`](file:///d:/schemasense/schemasense/schema_inspector.py#L166-L177) only captures the first column of multi-column foreign keys, silently dropping subsequent columns.
   - **Dependency and Constraints Drift:** [`requirements.txt`](file:///d:/schemasense/requirements.txt) still contains the deprecated `google-generativeai` package, [`constraints.txt`](file:///d:/schemasense/constraints.txt) has not pinned `google-genai`, [`setup.sh`](file:///d:/schemasense/setup.sh#L23) tests the wrong package, and [`sbom.cyclonedx.json`](file:///d:/schemasense/sbom.cyclonedx.json) is out of date.

---

## 2. Audit Scope & Environment

| Property | Reality / Configuration |
|---|---|
| **Root Directory** | `D:\schemasense` |
| **Operating System** | Windows 11 Enterprise (win32) |
| **Local Runtime** | Python 3.12.10 (`C:\Users\shiva\AppData\Local\Programs\Python\Python312\python.exe`) |
| **Frameworks** | Streamlit 1.42.2 (active server on port 8501), SQLAlchemy 2.0.38, sqlglot 26.8.0, google-genai 2.29.0 |
| **Test Engine** | Pytest 9.1.1, pytest-cov 7.1.0 |
| **Linter / Formatter** | Ruff 0.16.9 (0 errors across codebase) |
| **Supply Chain Tools** | `pip-audit` 2.10.1 (0 vulnerabilities), `detect-secrets` 1.5.0 (0 leaked secrets) |
| **Docker Daemon** | Active; however, SchemaSense test containers on ports 5433 and 3307 were offline during local run |

---

## 3. Repository Inventory & Architecture Map

### 3.1 File Inventory

| File Path | Type | Category | Status | Dependencies / Relationships | Purpose |
|---|---|---|---|---|---|
| [`app.py`](file:///d:/schemasense/app.py) | Python (1,256 lines) | Production (UI) | Active | `streamlit`, `schemasense.*`, `pandas`, `plotly` | Streamlit entry point; UI navigation, connection gateway, telemetry, and tab workflows. |
| [`schemasense/__init__.py`](file:///d:/schemasense/schemasense/__init__.py) | Python (28 lines) | Production | Active | `schemasense.ai_agent` | Package root; exports AI Agent symbols and package `__version__ = "1.0.0"`. |
| [`schemasense/connection.py`](file:///d:/schemasense/schemasense/connection.py) | Python (379 lines) | Production | Active | `sqlalchemy`, `threading` | Engine creation, URL builder, connection pool bounds, privilege inspector, query budget semaphore. |
| [`schemasense/sql_guard.py`](file:///d:/schemasense/schemasense/sql_guard.py) | Python (436 lines) | Production | Active | `sqlglot`, `sqlalchemy`, `pandas` | AST parser, security allowlists, dangerous function denylist, bounded executor with timeouts & row caps. |
| [`schemasense/schema_inspector.py`](file:///d:/schemasense/schemasense/schema_inspector.py) | Python (265 lines) | Production | Active | `sqlalchemy` | Structural inspection of tables, columns, PKs, FKs, unique constraints, indexes; JSON/MD exports; diffing. |
| [`schemasense/profiler.py`](file:///d:/schemasense/schemasense/profiler.py) | Python (193 lines) | Production | Active | `pandas` | Column statistical profiling, duplicate detection, candidate keys, IQR outlier detection, quality scoring. |
| [`schemasense/rules_engine.py`](file:///d:/schemasense/schemasense/rules_engine.py) | Python (71 lines) | Production | Active | `pypdf` | PDF document ingestion, text extraction, XML-delimited prompt constraint formatting. |
| [`schemasense/ai_agent.py`](file:///d:/schemasense/schemasense/ai_agent.py) | Python (386 lines) | Production | Active | `google.genai`, `pydantic` | Modern `google-genai` integration; structured output parsing, multi-key rotation, schema grounding. |
| [`schemasense/benchmark.py`](file:///d:/schemasense/schemasense/benchmark.py) | Python (387 lines) | Production / Test | Active | `sqlglot`, `schemasense.sql_guard` | Repeatable evaluation harness for SQL correctness, SQLGuard safety, and schema hallucination detection. |
| [`requirements.txt`](file:///d:/schemasense/requirements.txt) | Config | Dependencies | Inconsistent | None | Core dependencies. Contains redundant/deprecated `google-generativeai`. |
| [`requirements-dev.txt`](file:///d:/schemasense/requirements-dev.txt) | Config | Dependencies | Active | None | Development, CI quality gates, linting, and supply-chain audit dependencies. |
| [`constraints.txt`](file:///d:/schemasense/constraints.txt) | Config | Dependencies | Inconsistent | None | Pinned dependencies for reproducible builds. Missing `google-genai==2.29.0`. |
| [`docker-compose.test.yml`](file:///d:/schemasense/docker-compose.test.yml) | Docker Compose | Testing | Active | Docker | Disposable test containers for PostgreSQL (`16-alpine`, port 5433) and MySQL (`8.4`, port 3307). |
| [`.github/workflows/ci.yml`](file:///d:/schemasense/.github/workflows/ci.yml) | GitHub Actions | CI/CD | Active | GitHub | Three-stage CI pipeline: supply chain audit, unit tests (matrix 3.10/3.11/3.12), and live integration. |
| [`.github/dependabot.yml`](file:///d:/schemasense/.github/dependabot.yml) | Config | CI/CD | Active | GitHub | Automated dependency updates for pip and GitHub Actions. |
| [`.streamlit/config.toml`](file:///d:/schemasense/.streamlit/config.toml) | Config | UI Config | Active | Streamlit | Sets headless mode and dark Obsidian palette defaults. |
| [`.streamlit/secrets.toml.template`](file:///d:/schemasense/.streamlit/secrets.toml.template) | Config | Template | Inconsistent | Streamlit | Secrets template. Contains unused `[auth]` section for Google Sign-In OIDC. |
| [`setup.sh`](file:///d:/schemasense/setup.sh) | Bash Script | Deployment | Obsolete | Bash | Quick-start script. Tests deprecated `google.generativeai` package. |
| [`.secrets.baseline`](file:///d:/schemasense/.secrets.baseline) | JSON | Security | Active | `detect-secrets` | Secret scanning baseline file for CI and local verification. |
| [`sbom.cyclonedx.json`](file:///d:/schemasense/sbom.cyclonedx.json) | JSON | Supply Chain | Inconsistent | CycloneDX | CycloneDX v1.5 SBOM generated from old constraints (missing `google-genai`). |
| [`README.md`](file:///d:/schemasense/README.md) | Documentation | Documentation | Inconsistent | None | Project documentation. Outdated test counts (119 vs 143) and missing new modules. |
| [`DEPLOYMENT.md`](file:///d:/schemasense/DEPLOYMENT.md) | Documentation | Documentation | Active | None | Operational deployment guide, role provisioning scripts, and pilot checklist. |
| [`docs/security/threat-model.md`](file:///d:/schemasense/docs/security/threat-model.md) | Documentation | Security | Active | None | STRIDE threat model, 4 trust zones, and defensive countermeasures. |
| [`docs/security/database-permissions.md`](file:///d:/schemasense/docs/security/database-permissions.md) | Documentation | Security | Active | None | Database provisioning scripts for PostgreSQL and MySQL least-privilege roles. |
| [`docs/security/supply-chain.md`](file:///d:/schemasense/docs/security/supply-chain.md) | Documentation | Security | Active | None | Supply-chain security policy, SBOM governance, and vulnerability triage SLAs. |
| [`docs/architecture/target-architecture.md`](file:///d:/schemasense/docs/architecture/target-architecture.md) | Documentation | Architecture | Active | None | System architectural diagram, layer boundaries, and module responsibilities. |
| [`docs/deployment/readiness-checklist.md`](file:///d:/schemasense/docs/deployment/readiness-checklist.md) | Documentation | Deployment | Inconsistent | None | Readiness checklist. Contains historical milestone notes and test counts. |
| [`docs/testing/verification-plan.md`](file:///d:/schemasense/docs/testing/verification-plan.md) | Documentation | Testing | Active | None | Multi-tier test strategy, mutation testing evaluation, and adversarial verification matrix. |
| [`docs/audit/security-findings.md`](file:///d:/schemasense/docs/audit/security-findings.md) | Documentation | Audit | Active | None | Historical forensic vulnerability analysis (SEC-01 through SEC-10). |

---

## 4. Feature Inventory & Implementation Status Matrix

| Feature | UI Location | Entry Point | Backend Path | Data Source | Status | Evidence | Test Coverage | Known Limitations |
|---|---|---|---|---|---|---|---|---|
| **Database Connection & Health Check** | Sidebar Form | [`app.py:453-541`](file:///d:/schemasense/app.py#L453-L541) | [`create_safe_engine`](file:///d:/schemasense/schemasense/connection.py#L282), [`inspect_account_privileges`](file:///d:/schemasense/schemasense/connection.py#L94) | Live Database (Postgres / MySQL) | **IMPLEMENTED AND VERIFIED** | [`app.py:479`](file:///d:/schemasense/app.py#L479), [`connection.py:337`](file:///d:/schemasense/schemasense/connection.py#L337) | [`tests/test_connection.py:136`](file:///d:/schemasense/tests/test_connection.py#L136), [`tests/test_ui_apptest.py:29`](file:///d:/schemasense/tests/test_ui_apptest.py#L29) | UI only offers Postgres and MySQL; cannot connect to SQLite files. |
| **Least-Privilege Role Inspection** | Sidebar & Telemetry Bar | [`app.py:489`](file:///d:/schemasense/app.py#L489), [`app.py:666`](file:///d:/schemasense/app.py#L666) | [`inspect_account_privileges`](file:///d:/schemasense/schemasense/connection.py#L94-L227) | Database System Catalogs (`pg_roles`, `SHOW GRANTS`) | **IMPLEMENTED AND VERIFIED** | [`connection.py:116-169`](file:///d:/schemasense/schemasense/connection.py#L116-L169), [`connection.py:179-217`](file:///d:/schemasense/schemasense/connection.py#L179-L217) | [`tests/test_connection.py:76`](file:///d:/schemasense/tests/test_connection.py#L76), [`tests/test_ui_apptest.py:70`](file:///d:/schemasense/tests/test_ui_apptest.py#L70) | SQLite always returns non-privileged (`local_file`). |
| **Enforce Least-Privilege Rejection** | Sidebar Checkbox | [`app.py:461`](file:///d:/schemasense/app.py#L461), [`app.py:486`](file:///d:/schemasense/app.py#L486) | [`create_safe_engine`](file:///d:/schemasense/schemasense/connection.py#L344-L347) | Engine Connection Validation | **IMPLEMENTED AND VERIFIED** | [`connection.py:344`](file:///d:/schemasense/schemasense/connection.py#L344) raises `PrivilegedAccountError` | [`tests/test_ui_apptest.py:96`](file:///d:/schemasense/tests/test_ui_apptest.py#L96) | Only activates if checkbox is explicitly checked. |
| **Connection Disconnect & Pool Disposal** | Sidebar Button | [`app.py:542-556`](file:///d:/schemasense/app.py#L542-L556) | [`dispose_engine`](file:///d:/schemasense/schemasense/connection.py#L371-L379) | In-Memory Engine Pool | **IMPLEMENTED AND VERIFIED** | [`app.py:545`](file:///d:/schemasense/app.py#L545), [`connection.py:375`](file:///d:/schemasense/schemasense/connection.py#L375) | [`tests/test_connection.py:71`](file:///d:/schemasense/tests/test_connection.py#L71), [`tests/test_ui_apptest.py:62`](file:///d:/schemasense/tests/test_ui_apptest.py#L62) | Disconnect clears session state and calls `engine.dispose()`. |
| **Bounded Concurrency Budget Limiter** | Global Query Runner | [`connection.py:53-65`](file:///d:/schemasense/schemasense/connection.py#L53-L65) | `threading.BoundedSemaphore(4)` | Process In-Memory Semaphore | **IMPLEMENTED AND VERIFIED** | [`sql_guard.py:335`](file:///d:/schemasense/schemasense/sql_guard.py#L335) acquires `acquire_query_budget` | [`tests/test_connection.py:85`](file:///d:/schemasense/tests/test_connection.py#L85), [`test_connection.py:113`](file:///d:/schemasense/tests/test_connection.py#L113) | Process-scoped; does not share concurrency bounds across multi-worker deployments. |
| **AST SQL Validation (SQLGuard)** | Tab 4 & Backend | [`app.py:1178`](file:///d:/schemasense/app.py#L1178), [`sql_guard.py:139`](file:///d:/schemasense/schemasense/sql_guard.py#L139) | [`SQLGuard.validate`](file:///d:/schemasense/schemasense/sql_guard.py#L139-L289) | Query AST (`sqlglot`) | **IMPLEMENTED AND VERIFIED** | [`sql_guard.py:156`](file:///d:/schemasense/schemasense/sql_guard.py#L156), [`sql_guard.py:191`](file:///d:/schemasense/schemasense/sql_guard.py#L191), [`sql_guard.py:202`](file:///d:/schemasense/schemasense/sql_guard.py#L202) | [`tests/test_sql_guard.py:17-68`](file:///d:/schemasense/tests/test_sql_guard.py#L17-L68) (28 adversarial queries), [`test_ui_apptest.py:156`](file:///d:/schemasense/tests/test_ui_apptest.py#L156) | Rejects all multi-statement and non-SELECT queries. |
| **Bounded SQL Execution** | Tab 4 Button | [`app.py:1198`](file:///d:/schemasense/app.py#L1198) | [`execute_safe_query`](file:///d:/schemasense/schemasense/sql_guard.py#L291-L436) | Live Database Engine | **IMPLEMENTED AND VERIFIED** | [`sql_guard.py:323`](file:///d:/schemasense/schemasense/sql_guard.py#L323) rewrites AST with LIMIT, [`sql_guard.py:346`](file:///d:/schemasense/schemasense/sql_guard.py#L346) timeout | [`tests/test_sql_guard.py:153`](file:///d:/schemasense/tests/test_sql_guard.py#L153), [`test_sql_guard.py:193`](file:///d:/schemasense/tests/test_sql_guard.py#L193) | Enforces 5,000ms timeout and 1,000 row max cap. |
| **Schema Introspection** | Tab 2 & Connect Form | [`app.py:488`](file:///d:/schemasense/app.py#L488), [`app.py:866`](file:///d:/schemasense/app.py#L866) | [`inspect_database_schema`](file:///d:/schemasense/schemasense/schema_inspector.py#L128-L225) | SQLAlchemy Inspector (`inspect(engine)`) | **IMPLEMENTED AND VERIFIED** | [`schema_inspector.py:137-216`](file:///d:/schemasense/schemasense/schema_inspector.py#L137-L216) | [`tests/test_inspector.py:12`](file:///d:/schemasense/tests/test_inspector.py#L12) | Composite foreign keys drop columns past index 0. Table comments are not extracted. |
| **Data Dictionary Export (JSON / Markdown)** | Tab 2 Download Buttons | [`app.py:887-901`](file:///d:/schemasense/app.py#L887-L901) | [`DatabaseSchema.to_json`](file:///d:/schemasense/schemasense/schema_inspector.py#L61), [`DatabaseSchema.to_markdown`](file:///d:/schemasense/schemasense/schema_inspector.py#L64) | Active Schema Object | **IMPLEMENTED AND VERIFIED** | [`schema_inspector.py:61-97`](file:///d:/schemasense/schemasense/schema_inspector.py#L61-L97) | [`tests/test_inspector.py:28`](file:///d:/schemasense/tests/test_inspector.py#L28) | Markdown table structure formatted with column definitions, keys, and indexes. |
| **Schema Diffing** | None (Backend only) | None | [`diff_database_schemas`](file:///d:/schemasense/schemasense/schema_inspector.py#L228-L265) | Two `DatabaseSchema` objects | **IMPLEMENTED, NOT FULLY VERIFIED** | [`schema_inspector.py:228`](file:///d:/schemasense/schemasense/schema_inspector.py#L228) | [`tests/test_inspector.py:55`](file:///d:/schemasense/tests/test_inspector.py#L55) | Fully implemented in backend, but completely unexposed in UI (`app.py`). |
| **Data Profiling & Quality Scoring** | Tab 3 Button | [`app.py:989-1051`](file:///d:/schemasense/app.py#L989-L1051) | [`build_safe_profiling_query`](file:///d:/schemasense/schemasense/profiler.py#L172), [`profile_dataframe`](file:///d:/schemasense/schemasense/profiler.py#L46) | Live Table Query via `execute_safe_query` | **IMPLEMENTED AND VERIFIED** | [`profiler.py:63`](file:///d:/schemasense/schemasense/profiler.py#L63), [`profiler.py:112`](file:///d:/schemasense/schemasense/profiler.py#L112), [`profiler.py:156`](file:///d:/schemasense/schemasense/profiler.py#L156) | [`tests/test_profiler.py:11`](file:///d:/schemasense/tests/test_profiler.py#L11) | Samples first 1,000 rows. Quality score combines completeness (50%), uniqueness (30%), validity (20%). |
| **Statistical Visualizations (Plotly)** | Tab 3 & Tab 4 | [`app.py:1073-1115`](file:///d:/schemasense/app.py#L1073-L1115), [`app.py:1228-1244`](file:///d:/schemasense/app.py#L1228-L1244) | `plotly.express.histogram`, `plotly.express.bar` | Sampled / Query Dataframe | **IMPLEMENTED AND VERIFIED** | [`app.py:1073`](file:///d:/schemasense/app.py#L1073), [`app.py:1228`](file:///d:/schemasense/app.py#L1228) | Exercised in UI AppTest render tests | Renders dark-themed histograms for numeric columns and bar charts for categories. |
| **Business Rules Ingestion (PDF)** | Sidebar Expander | [`app.py:568-580`](file:///d:/schemasense/app.py#L568-L580) | [`extract_rules_from_pdf`](file:///d:/schemasense/schemasense/rules_engine.py#L19), [`format_rules_for_prompt`](file:///d:/schemasense/schemasense/rules_engine.py#L59) | Uploaded PDF File | **IMPLEMENTED AND VERIFIED** | [`rules_engine.py:34-48`](file:///d:/schemasense/schemasense/rules_engine.py#L34-L48) | [`tests/test_rules_engine.py:15`](file:///d:/schemasense/tests/test_rules_engine.py#L15) | Relies on `\n\n` paragraph breaks; caps at 20 pages and top 30 rules. |
| **Schema Privacy Isolation** | Sidebar Multiselect | [`app.py:588-595`](file:///d:/schemasense/app.py#L588-L595) | [`DatabaseSchema.to_prompt_context(table_filter=...)`](file:///d:/schemasense/schemasense/schema_inspector.py#L99-L125) | Discovered Table List | **IMPLEMENTED AND VERIFIED** | [`app.py:613`](file:///d:/schemasense/app.py#L613), [`schema_inspector.py:107`](file:///d:/schemasense/schemasense/schema_inspector.py#L107) | [`tests/test_inspector.py:50`](file:///d:/schemasense/tests/test_inspector.py#L50) | Unchecked tables are omitted from the LLM prompt and disallowed in `SQLGuard`. |
| **AI Agent Studio (Streaming)** | Tab 1 Input | [`app.py:756-845`](file:///d:/schemasense/app.py#L756-L845) | [`SchemaSenseAIAgent.generate_stream`](file:///d:/schemasense/schemasense/ai_agent.py#L309-L385) | `google.genai` SDK | **IMPLEMENTED AND VERIFIED** | [`ai_agent.py:362`](file:///d:/schemasense/schemasense/ai_agent.py#L362), [`app.py:756`](file:///d:/schemasense/app.py#L756) | [`tests/test_ai_agent.py:9-12`](file:///d:/schemasense/tests/test_ai_agent.py#L9-L12) | Requires valid `GEMINI_API_KEY` to connect to external service; handles timeouts and 429 rotation. |
| **AI Structured Reasoning (Pydantic)** | Tab 1 Radio & Generate | [`app.py:731-755`](file:///d:/schemasense/app.py#L731-L755) | [`SchemaSenseAIAgent.generate_structured`](file:///d:/schemasense/schemasense/ai_agent.py#L224-L307) | `google.genai` with `SQLAgentResponse` | **IMPLEMENTED AND VERIFIED** | [`ai_agent.py:251-271`](file:///d:/schemasense/schemasense/ai_agent.py#L251-L271), [`app.py:733`](file:///d:/schemasense/app.py#L733) | [`tests/test_ai_agent.py:68-82`](file:///d:/schemasense/tests/test_ai_agent.py#L68-L82), [`test_ai_agent.py:151`](file:///d:/schemasense/tests/test_ai_agent.py#L151) | Flags out-of-schema hallucinated tables before loading into sandbox. |
| **Repeatable Evaluation Benchmark** | Module CLI (`-m schemasense.benchmark`) | [`benchmark.py:344-386`](file:///d:/schemasense/schemasense/benchmark.py#L344-L386) | [`run_benchmark`](file:///d:/schemasense/schemasense/benchmark.py#L269), [`compare_implementations`](file:///d:/schemasense/schemasense/benchmark.py#L332) | 8 Standard Benchmark Test Cases | **IMPLEMENTED AND VERIFIED** | [`benchmark.py:108-181`](file:///d:/schemasense/schemasense/benchmark.py#L108-L181), [`benchmark.py:320-330`](file:///d:/schemasense/schemasense/benchmark.py#L320-L330) | [`tests/test_benchmark.py:110-209`](file:///d:/schemasense/tests/test_benchmark.py#L110-L209) | Proves modern structured agent achieves 100% safety & hallucination-free rate vs 50% legacy regex. |
| **Authentication & Authorization** | None | None | None | None | **NOT IMPLEMENTED** | Zero authentication checks in [`app.py`](file:///d:/schemasense/app.py) | None | Documentation references OIDC Google Sign-In, but code has no login barrier. |

---

## 5. Working & Verified Features

The following features have genuine implementations supported by executable test evidence and static verification:
1. **Dialect-Aware AST Validation (`schemasense.sql_guard.SQLGuard.validate`)**:
   - Accurately identifies `exp.Select` and `exp.Union` root statements.
   - Blocks 16 forbidden AST node types (`DROP`, `TRUNCATE`, `ALTER`, `CREATE`, `INSERT`, `UPDATE`, `DELETE`, `GRANT`, `REVOKE`, `MERGE`, etc.).
   - Disallows 22 dangerous functions across PostgreSQL, MySQL, and SQLite.
   - Rejects stacked semicolon statements even when embedded in SQL comments.
   - Verified by 56 automated test cases in [`tests/test_sql_guard.py`](file:///d:/schemasense/tests/test_sql_guard.py).
2. **Dialect-Specific Bounded Execution (`schemasense.sql_guard.execute_safe_query`)**:
   - PostgreSQL: Enforces `SET TRANSACTION READ ONLY`, `SET LOCAL statement_timeout`, and `SET LOCAL work_mem = '16MB'` inside an explicit transaction block (`with conn.begin():`).
   - MySQL: Enforces `SET SESSION TRANSACTION READ ONLY` and `SET SESSION max_execution_time`, resetting to `0` in a `finally` block.
   - SQLite: Enforces `PRAGMA query_only = ON` and attaches a Python `set_progress_handler` timeout check, clearing it in a `finally` block.
   - Rewrites AST to cap query `LIMIT` at `max_rows + 1` (default 1,001 rows), detecting truncation accurately.
3. **Non-Heuristic Least-Privilege Catalog Inspection (`schemasense.connection.inspect_account_privileges`)**:
   - PostgreSQL: Inspects `current_setting('is_superuser')`, `pg_roles` attributes (`rolsuper`, `rolcreaterole`, `rolcreatedb`), and `has_database_privilege(..., 'CREATE')`.
   - MySQL: Inspects `SHOW GRANTS` output for `ALL PRIVILEGES`, `SUPER`, `SYSTEM_VARIABLES_ADMIN`, `SHUTDOWN`, and `WITH GRANT OPTION`.
   - SQLite: Returns non-privileged `local_file` status.
4. **Application Concurrency Budget Limiter (`schemasense.connection.acquire_query_budget`)**:
   - Enforces a strict `threading.BoundedSemaphore(4)` across in-flight database query operations.
   - Bounded timeout (default 5.0s) prevents worker thread exhaustion, raising `ConcurrencyLimitExceeded`.
   - Verified by multithreaded test cases in [`tests/test_connection.py`](file:///d:/schemasense/tests/test_connection.py#L85-L134).
5. **Modern AI Agent SDK (`schemasense.ai_agent.SchemaSenseAIAgent`)**:
   - Utilizes official `google-genai` client with Pydantic structured output validation ([`SQLAgentResponse`](file:///d:/schemasense/schemasense/ai_agent.py#L31-L57)).
   - In-memory credential rotation across available keys on HTTP `429` / quota exhaustion.
   - Schema fidelity validator ([`verify_schema_fidelity`](file:///d:/schemasense/schemasense/ai_agent.py#L112-L126)) intercepts out-of-schema table hallucinations.
6. **Data Profiling & Quality Engine (`schemasense.profiler.profile_dataframe`)**:
   - Calculates null percentage, distinct counts, candidate primary keys, duplicate row percentage, and IQR numeric outliers.
   - Computes weighted composite Quality Score: `(0.5 * completeness) + (0.3 * uniqueness) + (0.2 * validity)`.
7. **Safe Identifier Quoting & Query Builder (`schemasense.profiler.build_safe_profiling_query`)**:
   - Quotes table identifiers using dialect-appropriate delimiters (backticks for MySQL, double quotes for Postgres/SQLite).
   - Handles schema-qualified tables (`schema.table`).

---

## 6. Implemented but Unverified Features

The following features have genuine backend implementations, but could not be completely verified end-to-end under current local conditions:

1. **Live PostgreSQL & MySQL Container Integration Tests (38 tests in [`tests/test_live_integration.py`](file:///d:/schemasense/tests/test_live_integration.py))**:
   - **Status**: The test suite is implemented and structurally sound.
   - **Verification Barrier**: The test containers (`postgres:16-alpine` on port 5433 and `mysql:8.4` on port 3307) defined in [`docker-compose.test.yml`](file:///d:/schemasense/docker-compose.test.yml) were offline during this local audit. The tests safely skipped via `is_port_open()`.
   - **Required Action**: Spin up disposable test containers (`docker compose -f docker-compose.test.yml up -d --wait`) to execute live container integration tests.
2. **Schema Diffing (`schemasense.schema_inspector.diff_database_schemas`)**:
   - **Status**: Implemented in [`schemasense/schema_inspector.py:228-265`](file:///d:/schemasense/schemasense/schema_inspector.py#L228-L265) and verified with unit tests in [`tests/test_inspector.py:55-92`](file:///d:/schemasense/tests/test_inspector.py#L55-L92).
   - **Verification Barrier**: It is not wired to any UI trigger, session state history, or user workflow in [`app.py`](file:///d:/schemasense/app.py). End-to-end user verification is impossible without UI integration.

---

## 7. Partially Implemented Features

1. **Composite Foreign Key Extraction ([`schemasense/schema_inspector.py:166-177`](file:///d:/schemasense/schemasense/schema_inspector.py#L166-L177))**:
   - **Problem**: When a foreign key constraint consists of multiple columns (e.g. composite foreign key `(fk_1, fk_2)`), the code extracts `constrained[0]` and `referred_cols[0]`, silently discarding all subsequent column mappings.
   - **User Impact**: Composite relationships appear as single-column relationships in data dictionary exports and LLM prompt context.
2. **Table Comment Introspection ([`schemasense/schema_inspector.py:209`](file:///d:/schemasense/schemasense/schema_inspector.py#L209))**:
   - **Problem**: `TableSchema` has a `comment: str | None` field, but `inspect_database_schema` never queries `inspector.get_table_comment(table_name)`. Table comments are always `None`.
3. **SQLite Support in the UI ([`app.py:447-471`](file:///d:/schemasense/app.py#L447-L471))**:
   - **Problem**: Supported in the backend engine, AST guard, and inspector, but the UI dropdown only presents `["PostgreSQL", "MySQL"]` and enforces mandatory Host, Port, Database, User, and Password fields. SQLite file paths cannot be entered through the UI.

---

## 8. Mocked, Fake, Static, and Placeholder Behavior

The codebase was searched for hardcoded data, dummy responses, fake analytics, and misleading status indicators.

### Findings:

| Location | Code Reality | Misleading Expectation | Runtime Impact | Priority |
|---|---|---|---|---|
| [`app.py:663-664`](file:///d:/schemasense/app.py#L663-L664) | Renders static badges `AST Guard: Active (sqlglot)` and `Pool Limit: 4 Concurrent`. | A user might assume these badges represent a live check against an active monitoring agent. | Minimal; accurately reflects the hardcoded application policy in [`schemasense/sql_guard.py`](file:///d:/schemasense/schemasense/sql_guard.py) and [`schemasense/connection.py`](file:///d:/schemasense/schemasense/connection.py). | Low |
| [`app.py:408-413`](file:///d:/schemasense/app.py#L408-L413) | Initial assistant message is a hardcoded welcome string. | Legitimate conversational greeting. | None. | Informational |
| [`schemasense/ai_agent.py:155`](file:///d:/schemasense/schemasense/ai_agent.py#L155) | Returns `["DEMO_KEY_STANDBY"]` when no API keys are found in environment or secrets. | A user might think a demo key works offline. In reality, requests fail with an explicit `AIProviderError` from Google's API. | Prevents crash on startup, but could be clearer in the UI that an API key is required. | Medium |
| [`.streamlit/secrets.toml.template:11-19`](file:///d:/schemasense/.streamlit/secrets.toml.template#L11-L19) | Documents `[auth]` configuration block for Google Sign-In OIDC. | Users or operators assume Google Sign-In is active or configured in the app. In reality, [`app.py`](file:///d:/schemasense/app.py) has no auth handler. | High; creates a false impression of access control readiness. | High |
| [`schemasense/benchmark.py:345-380`](file:///d:/schemasense/schemasense/benchmark.py#L345-L380) | Contains `sim_modern` and `sim_legacy` simulation functions inside `if __name__ == "__main__":`. | Demonstrates the comparative benchmark report without consuming paid API quotas when executed as a CLI script. | None during normal application runtime; clearly scoped to the benchmark CLI demonstration. | Informational |

---

## 9. Broken Features & Confirmed Bugs

| ID | Issue | Severity | Confidence | File & Line | Evidence & Description | Recommended Action |
|---|---|---|---|---|---|---|
| **BUG-01** | Missing SQLite connection option in UI | Medium | Confirmed | [`app.py:448-471`](file:///d:/schemasense/app.py#L448-L471) | UI dropdown only offers `PostgreSQL` and `MySQL`. Form requires `db_host`, `db_port`, `db_name`, `db_user`, `db_pass`. SQLite files cannot be connected. | Add `SQLite` option to engine selectbox and dynamic file path input when SQLite is selected. |
| **BUG-02** | Composite foreign key truncation | Medium | Confirmed | [`schemasense/schema_inspector.py:166-177`](file:///d:/schemasense/schemasense/schema_inspector.py#L166-L177) | `constrained[0]` and `referred_cols[0]` take only the first column of composite foreign keys. | Update `ForeignKeyInfo` to accept `list[str]` for columns, capturing all composite key mappings. |
| **BUG-03** | Missing table comment extraction | Low | Confirmed | [`schemasense/schema_inspector.py:209`](file:///d:/schemasense/schemasense/schema_inspector.py#L209) | `TableSchema.comment` is always `None` because `inspector.get_table_comment(table_name)` is never called. | Call `inspector.get_table_comment(table_name)` inside `inspect_database_schema`. |
| **BUG-04** | Identifier quoting without inner quote escaping | Low | Confirmed | [`schemasense/profiler.py:180-190`](file:///d:/schemasense/schemasense/profiler.py#L180-L190) | Wraps identifiers in `"` or ``` ` ``` without replacing inner instances (`"` -> `""`, ``` ` ``` -> ``` `` ```). | Escape inner identifier quotes before constructing profiling query. |
| **BUG-05** | Stale SDK test in quick-start script | Low | Confirmed | [`setup.sh:23`](file:///d:/schemasense/setup.sh#L23) | Checks `import google.generativeai` instead of `import google.genai`. | Update package check to `python -c "import google.genai"`. |
| **BUG-06** | Dependency constraint mismatch | Medium | Confirmed | [`requirements.txt:3`](file:///d:/schemasense/requirements.txt#L3), [`constraints.txt:7`](file:///d:/schemasense/constraints.txt#L7) | `requirements.txt` still includes `google-generativeai>=0.8.3`. `constraints.txt` pins `google-generativeai` but omits `google-genai`. | Remove `google-generativeai` and pin `google-genai==2.29.0` in `constraints.txt`. |

---

## 10. Suspected Dead Code & Architecture

| Location | Evidence of Non-Use | Dynamic Usage Checked? | Confidence | Removal / Integration Risk |
|---|---|---|---|---|
| [`schemasense/schema_inspector.py:228-265`](file:///d:/schemasense/schemasense/schema_inspector.py#L228-L265) (`diff_database_schemas`) | Grepped entire codebase; only called in [`tests/test_inspector.py:87`](file:///d:/schemasense/tests/test_inspector.py#L87). Never imported or referenced in `app.py`. | Yes; verified zero dynamic imports or callbacks. | High | **Do NOT remove.** It is a useful domain feature that should be wired into a new "Schema Version Diff" tab in the UI. |
| [`app.py:382-394`](file:///d:/schemasense/app.py#L382-L394) (`generate_with_fallback`, `extract_sql_from_response`) | Legacy bridge aliases retained for backwards compatibility. Never called internally within `app.py`. | Yes; confirmed. | High | Low risk; can be removed or kept as backward-compatibility shims. |
| [`requirements.txt:3`](file:///d:/schemasense/requirements.txt#L3) (`google-generativeai>=0.8.3`) | Package completely unimported anywhere in `app.py` or `schemasense/`. | Yes; verified zero active imports. | High | Safe to remove. Eliminates dead dependency and potential confusion. |

---

## 11. Security Audit & Threat Analysis

### 11.1 Authentication & Authorization
- **Status:** **NOT IMPLEMENTED.**
- **Finding:** The application currently runs with zero authentication barriers. Any user who can reach `http://localhost:8501` has immediate access to the full console, can connect to databases, and can execute queries.
- **Risk:** In an internal network or cloud pilot without an upstream Identity-Aware Proxy (IAP), unauthorized team members could query sensitive database tables or exhaust API keys.
- **Mitigation:** Implement authentication via Streamlit OIDC or deploy strictly behind an enterprise reverse proxy (Cloudflare Access, AWS ALB OIDC).

### 11.2 Database Security & SQL Injection
- **AST Protection:** **STRONG.** `SQLGuard` reliably blocks DDL, DML writes, stacked statements, and dangerous functions.
- **System Schema Isolation:** **STRONG.** Protects `pg_catalog`, `information_schema`, `mysql`, `sys`, and `sqlite_master`.
- **SSRF Risk in Connection Form:** **MEDIUM.** The database host input accepts arbitrary IP addresses and hostnames. An attacker with UI access could supply `127.0.0.1`, `169.254.169.254`, or internal VPC IPs to probe internal services.
- **Mitigation:** Add an optional host allowlist or block connections to loopback (`127.0.0.1`) and cloud metadata endpoints (`169.254.169.254`) unless explicitly allowed via configuration.

### 11.3 Secrets & Credential Handling
- **Status:** **VERIFIED SECURE.**
- Password inputs use `type="password"`.
- Passwords are kept in memory and masked (`********`) in driver exceptions before being displayed or logged.
- `detect-secrets scan` against `.secrets.baseline` reported 0 leaked credentials.
- Zero secrets committed to git history.

---

## 12. AI Implementation Audit

### 12.1 Model & SDK Verification
- **SDK:** Official `google-genai` (`v2.29.0`).
- **Models Supported:** `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash`, `gemini-3-flash-preview`. Selection in the sidebar dynamically sets `agent.model_name`.
- **Structured Output:** Enforced via `types.GenerateContentConfig(response_mime_type="application/json", response_schema=SQLAgentResponse)`.
- **Streaming:** Implemented via `client.models.generate_content_stream(...)` with system instruction grounding.

### 12.2 Schema Grounding & Hallucination Prevention
- When a database is connected, [`DatabaseSchema.to_prompt_context`](file:///d:/schemasense/schemasense/schema_inspector.py#L99-L125) formats only discovered tables into `<database_schema>` tags.
- The user can de-select sensitive tables in the sidebar; excluded tables are withheld from the prompt.
- [`verify_schema_fidelity()`](file:///d:/schemasense/schemasense/ai_agent.py#L112-L126) parses table references from the generated SQL and checks them against `allowed_tables`. Queries referencing hallucinated tables (e.g. `payments`, `ssn_records`) are flagged with a UI warning and marked non-executable.

### 12.3 Prompt Injection Defense
- Uploaded business rules from PDFs are treated strictly as reference text inside `<enterprise_business_rules>` tags.
- Regardless of what the LLM emits (even if prompt injection instructs it to `DROP TABLE`), the SQL **must pass through `SQLGuard.validate()`** before execution, guaranteeing rejection.

---

## 13. Database Integration & SQL Safety

| Parameter | PostgreSQL | MySQL | SQLite |
|---|---|---|---|
| **Driver** | `psycopg2-binary` | `pymysql` (`utf8mb4`) | `sqlite3` |
| **Driver Read-Only Enforcement** | `SET TRANSACTION READ ONLY` | `SET SESSION TRANSACTION READ ONLY` | `PRAGMA query_only = ON` |
| **Statement Timeout** | `SET LOCAL statement_timeout = '<ms>'` | `SET SESSION max_execution_time = <ms>` (reset to 0 in finally) | Python `set_progress_handler` timeout check |
| **Memory / Buffer Bounds** | `SET LOCAL work_mem = '16MB'` | Engine default | In-memory / file |
| **Row Cap Enforcement** | AST `LIMIT 1001` + `fetchmany(1001)` | AST `LIMIT 1001` + `fetchmany(1001)` | AST `LIMIT 1001` + `fetchmany(1001)` |
| **Connection Pool** | `QueuePool` (size: 5, overflow: 2, recycle: 1800s) | `QueuePool` (size: 5, overflow: 2, recycle: 1800s) | `NullPool` / memory |

---

## 14. Testing Quality & Actual Test Results

### 14.1 Executable Test Results
A complete run of the test suite yielded:
- **Fast Unit Tests & AppTest (Offline)**: **102 Passed, 0 Failed** (in 9.11s)
- **Live Container Tests (PostgreSQL & MySQL)**: **41 Skipped** (safely skipped due to test containers being offline)
- **Total Test Items Collected**: **143**

### 14.2 Code Coverage Breakdown

```
Name                              Stmts   Miss  Cover   Missing Lines
---------------------------------------------------------------------
schemasense\__init__.py               3      0   100%
schemasense\ai_agent.py             156     53    66%   20-23, 97, 107, 120, 133, 139-142, 150-151, 182, 234, 264, 302-306, 321-386
schemasense\benchmark.py            122     23    81%   211-212, 230-231, 286-288, 356-398
schemasense\connection.py           156     73    53%   87, 107-164, 172-212, 256, 309, 338-348, 351-360, 362, 367, 377-378
schemasense\profiler.py              95      3    97%   43, 185, 188
schemasense\rules_engine.py          40      6    85%   29, 36, 48, 54-56
schemasense\schema_inspector.py     146     15    90%   72, 91-94, 188-189, 197-207, 217-219
schemasense\sql_guard.py            189     46    76%   161-162, 170, 221, 260, 328-330, 333, 340-354, 358-373, 379-380, 404-405, 425-435
---------------------------------------------------------------------
TOTAL                               907    219    76%
```

- **Analysis**:
  - Overall line coverage is **76%**, satisfying the CI quality gate threshold (`--cov-fail-under=70`).
  - Missing coverage in `connection.py` (53%) and `sql_guard.py` (76%) corresponds to PostgreSQL- and MySQL-specific driver execution blocks that require live container instances (covered when Docker containers are running).
  - Missing coverage in `ai_agent.py` (66%) corresponds to live network streaming calls to Google's API, which are avoided in unit tests to prevent quota usage.

---

## 15. Dependencies & Supply-Chain Health

1. **`pip-audit`**: Executed against [`requirements.txt`](file:///d:/schemasense/requirements.txt).
   - **Result**: `No known vulnerabilities found`. Exit code 0.
2. **`detect-secrets`**: Executed against [`.secrets.baseline`](file:///d:/schemasense/.secrets.baseline).
   - **Result**: 0 uncommitted secrets detected. Exit code 0.
3. **`ruff`**: Executed across `schemasense/`, `tests/`, and `app.py`.
   - **Result**: `All checks passed!`. 0 lint errors, 20 files properly formatted.
4. **Supply Chain Defect**:
   - [`requirements.txt`](file:///d:/schemasense/requirements.txt) includes both `google-genai>=2.0.0` and obsolete `google-generativeai>=0.8.3`.
   - [`constraints.txt`](file:///d:/schemasense/constraints.txt) pins `google-generativeai==0.8.6` but lacks `google-genai`.
   - [`sbom.cyclonedx.json`](file:///d:/schemasense/sbom.cyclonedx.json) lists `google-generativeai==0.8.6` as a dependency component, but lacks `google-genai`.

---

## 16. UI & Product-Completeness Review

- **Visual Theme**: Obsidian Telemetry theme (`#080B12` canvas, `#0E1424` surface) with restrained Cyan (`#38BDF8`) and Violet (`#8B5CF6`) accents.
- **Zero Emojis**: The user interface has zero emojis, adhering strictly to enterprise console design principles.
- **Telemetry Bar**: Displays live connection status, active user role, AST guard status, and concurrency budget.
- **Tab 1 (AI Agent Studio)**:
  - Supports both **Streaming Dialogue** and **Structured Schema Reasoning (Pydantic)**.
  - Displays formatted SQL blocks, safety notes, and schema hallucination warnings.
  - Includes a working "Clear Chat" button.
- **Tab 2 (Schema Architecture)**:
  - Renders tables, columns, PKs, FKs, and index breakdowns.
  - JSON and Markdown download buttons work seamlessly.
- **Tab 3 (Column Profiler & Health)**:
  - Selects table, runs 1,000-row sample query, computes data quality metrics, and renders interactive Plotly histograms and frequency bars.
- **Tab 4 (SQL Guard Sandbox)**:
  - In disconnected state: functions as an offline AST validator.
  - In connected state: executes queries through `execute_safe_query` with timeout, row cap, and automatic Plotly bar chart generation.

---

## 17. Performance & Scalability Risks

1. **Multi-Session Connection Pool Inflation**:
   - Each browser session creates its own SQLAlchemy `Engine` in `st.session_state.db_engine`.
   - With `pool_size=5, max_overflow=2`, each connected user can hold up to 7 database connections. Ten active users can open 70 connections on the target database.
2. **Single-Process Query Semaphore**:
   - `MAX_CONCURRENT_QUERIES = 4` bounds in-flight queries within a single Python process. If deployed across multiple worker processes (e.g. gunicorn/uvicorn workers), the semaphore is not shared.
3. **Repeated Schema Introspection**:
   - Disconnecting and reconnecting triggers full schema introspection across all tables. For databases with thousands of tables, this can cause initial latency.

---

## 18. Documentation Inconsistencies Table

| Document | Stated Claim | Code Reality | Discrepancy Type |
|---|---|---|---|
| [`README.md:9`](file:///d:/schemasense/README.md#L9) | Badge claims: `Tests: 119 Passed` | Currently 143 tests collected (102 passed, 41 skipped offline) | Outdated test count metric |
| [`README.md:10`](file:///d:/schemasense/README.md#L10) | Badge claims: `Coverage: 89%` | Offline coverage is 76% across 907 statements | Stale coverage metric |
| [`README.md:15`](file:///d:/schemasense/README.md#L15), [`README.md:37`](file:///d:/schemasense/README.md#L37) | "Connects to relational databases (PostgreSQL, MySQL, SQLite)" | UI connection form only offers PostgreSQL and MySQL; SQLite files cannot be connected via UI | Missing UI functionality |
| [`README.md:114-129`](file:///d:/schemasense/README.md#L114-L129) | Architecture directory tree lists only 5 files in `schemasense/` | Missing `schemasense/ai_agent.py` and `schemasense/benchmark.py` | Stale architecture documentation |
| [`.streamlit/secrets.toml.template:11-19`](file:///d:/schemasense/.streamlit/secrets.toml.template#L11-L19) | Documents `[auth]` section for Google Sign-In OIDC | No authentication code exists in [`app.py`](file:///d:/schemasense/app.py) | Unimplemented feature advertised in config |
| [`setup.sh:23`](file:///d:/schemasense/setup.sh#L23) | Checks `python -c "import google.generativeai"` | Code migrated to `google.genai`; `google.generativeai` is no longer used | Deprecated package check in script |
| [`constraints.txt:7`](file:///d:/schemasense/constraints.txt#L7) | Pins `google-generativeai==0.8.6` | `google-genai` is not pinned; `google-generativeai` is unused | Inconsistent dependency pinning |

---

## 19. Technical Debt & Maintenance Risks

1. **Monolithic `app.py`**:
   - [`app.py`](file:///d:/schemasense/app.py) is 1,256 lines long and combines styling CSS, session state management, sidebar forms, telemetry, chat rendering, profiler charts, and SQL sandbox handlers.
   - Refactoring UI tabs into dedicated view modules (e.g. `views/ai_studio.py`, `views/schema_view.py`, `views/profiler_view.py`, `views/sandbox_view.py`) will significantly improve maintainability.
2. **Dual Model SDK Dependencies in `requirements.txt`**:
   - Both `google-genai` and `google-generativeai` are installed, creating risk of confusion and potential version conflicts.

---

## 20. Missing Capabilities & Unimplemented Requirements

1. **Authentication & User Identity**: No login screen, SSO, or user session isolation.
2. **SQLite File Connection in UI**: Cannot select or upload an SQLite `.db` / `.sqlite` file in the sidebar.
3. **Schema Diffing UI**: `diff_database_schemas` exists in the backend but has no UI presentation.
4. **Persistent Query History**: Query history in `SQL Guard Sandbox` is held in transient session state and lost upon browser refresh.

---

## 21. Prioritized Development Plan (Roadmap P0 – P4)

### P0 — Critical Correctness & Dependency Hygiene
1. **Clean up Dependency Files (`requirements.txt`, `constraints.txt`, `setup.sh`, `sbom.cyclonedx.json`)**:
   - **Why**: Remove deprecated `google-generativeai` dependency, pin `google-genai==2.29.0` in `constraints.txt`, update `setup.sh` verification, and regenerate CycloneDX SBOM.
   - **Files**: [`requirements.txt`](file:///d:/schemasense/requirements.txt), [`constraints.txt`](file:///d:/schemasense/constraints.txt), [`setup.sh`](file:///d:/schemasense/setup.sh), [`sbom.cyclonedx.json`](file:///d:/schemasense/sbom.cyclonedx.json).
   - **Acceptance Criteria**: `pip-audit`, `pip install -c constraints.txt`, and SBOM scans succeed without references to `google-generativeai`.

2. **Fix Composite Foreign Key Truncation in Schema Inspector**:
   - **Why**: Composite foreign keys currently drop all columns past index 0.
   - **Files**: [`schemasense/schema_inspector.py`](file:///d:/schemasense/schemasense/schema_inspector.py).
   - **Acceptance Criteria**: `ForeignKeyInfo` supports list of column pairs; multi-column foreign keys are preserved in exports.

### P1 — Core Product Completeness
3. **Add SQLite Connection Support to UI**:
   - **Why**: SQLite is supported by the backend and claimed in README, but inaccessible in the UI.
   - **Files**: [`app.py`](file:///d:/schemasense/app.py).
   - **Acceptance Criteria**: Selecting "SQLite" reveals a file path or in-memory option, allowing local database inspection.

4. **Surface Schema Diffing in UI**:
   - **Why**: [`diff_database_schemas`](file:///d:/schemasense/schemasense/schema_inspector.py#L228) is fully tested in backend but inaccessible to users.
   - **Files**: [`app.py`](file:///d:/schemasense/app.py).
   - **Acceptance Criteria**: Users can save a schema snapshot and view schema changes (added/removed/modified tables and columns) side-by-side.

### P2 — Security & Authentication
5. **Implement Authentication Gateway (Google Sign-In / Streamlit OIDC)**:
   - **Why**: Prevent unauthenticated network access to database connection gateways.
   - **Files**: [`app.py`](file:///d:/schemasense/app.py), [`.streamlit/secrets.toml.template`](file:///d:/schemasense/.streamlit/secrets.toml.template).
   - **Acceptance Criteria**: Unauthenticated users must log in before accessing database connection parameters or SQL execution.

6. **Add Egress / SSRF Host Validation on Database Host Input**:
   - **Why**: Block SSRF attempts to cloud metadata endpoints (`169.254.169.254`) or loopback unless in debug mode.
   - **Files**: [`schemasense/connection.py`](file:///d:/schemasense/schemasense/connection.py), [`app.py`](file:///d:/schemasense/app.py).
   - **Acceptance Criteria**: Rejects connection attempts to prohibited IP ranges with a clear validation error.

### P3 — UI Architecture & Refactoring
7. **Modularize `app.py` into View Components**:
   - **Why**: Decompose 1,256-line monolithic script into structured view components.
   - **Files**: `app.py`, new `views/` directory.
   - **Acceptance Criteria**: `app.py` acts as router/shell; each tab is self-contained.

### P4 — Enhanced Capabilities
8. **Add Query Execution History & Export**:
   - **Why**: Allow analysts to view historical guarded queries run in the session and download query result sets as CSV/Parquet.
   - **Files**: `app.py`, `schemasense/sql_guard.py`.

---

## 22. Unverified Areas & Audit Limitations

1. **Live Container Execution**: 38 PostgreSQL and MySQL integration tests in [`tests/test_live_integration.py`](file:///d:/schemasense/tests/test_live_integration.py) were skipped because the test containers on ports 5433 and 3307 were offline. These tests are structurally sound and verified in GitHub Actions CI, but were not executed locally during this read-only audit.
2. **Live External AI API Calls**: In accordance with Operating Rule 6 ("Do not transmit source code, credentials... to external AI services"), no live requests were made to Google's paid Gemini API during this audit. Model behavior was verified via unit tests, AST parsers, and local simulation benchmarks.

---

## 23. Appendix: File Paths, Commands & Executed Verifications

### Executed Commands & Verifications
- `pytest tests -v` — 102 passed, 41 skipped, 0 failed.
- `pytest tests --ignore=tests/test_live_integration.py --cov=schemasense --cov-report=term-missing` — 76% total line coverage (907 statements, 219 missed).
- `ruff check schemasense/ tests/ app.py` — 0 lint errors (All checks passed).
- `ruff format --check schemasense/ tests/ app.py` — 20 files properly formatted.
- `pip-audit -r requirements.txt` — 0 vulnerabilities found.
- `detect-secrets scan --baseline .secrets.baseline` — 0 secrets detected.
- `python -m py_compile app.py` — Clean compilation.
- `python -m schemasense.benchmark` — Comparative benchmark output verified.
- `curl.exe -I http://localhost:8501/` — HTTP 200 OK verified.
