# Verification Plan & Adversarial Test Suite Documentation

## 1. Test Strategy & Defense-in-Depth

The verification strategy guarantees that SchemaSense enforces security boundaries, data integrity, and operational reliability across database engines. 

Testing is split across three distinct tiers:
1. **Isolated Unit & Guardrail Tests (`tests/test_sql_guard.py`, `tests/test_inspector.py`, `tests/test_profiler.py`, `tests/test_rules_engine.py`, `tests/test_connection.py`)**: Fast, pure-Python / in-memory SQLite tests validating AST parsing, boundary constraints, statistical algorithms, PDF parsing, password masking, and prompt generation.
2. **Live Integration & Adversarial Verification Suite (`tests/test_live_integration.py`)**: Disposable, containerized environments testing against real database engines (`postgres:16-alpine`, `mysql:8.4`, and `sqlite3`).
3. **Headless Streamlit UI Workflow Tests (`tests/test_ui_apptest.py`)**: Programmatic UI tests using Streamlit's native `AppTest` framework to simulate user input, form submissions, least-privilege warnings, and admin credential rejection without browser binaries.

---

## 2. Verified Engine Versions Matrix

| Database Engine | Pinned Docker Image / Version | Driver / Connector | Test Port | Verification Status |
|---|---|---|---|---|
| **PostgreSQL** | `postgres:16-alpine` | `psycopg2-binary >= 2.9.0` | `5433` (mapped from 5432) | **VERIFIED (Live Container)** |
| **MySQL** | `mysql:8.4` (LTS) | `pymysql >= 1.1.0` | `3307` (mapped from 3306) | **VERIFIED (Live Container)** |
| **SQLite** | In-memory & file-backed (`Python 3.12.10`) | `sqlite3` DBAPI | N/A (Local) | **VERIFIED (Local Engine)** |

---

## 3. Adversarial Verification Scenarios Tested

### 1. Write Prevention (Application + Database Engine Layers)
- **Application AST**: `sqlglot` AST parsing intercepts `INSERT`, `UPDATE`, `DELETE`, `DROP`, `TRUNCATE`, `ALTER`, and `CREATE` before execution.
- **Database Role Level**: Direct queries submitted via dedicated `schemasense_readonly` accounts fail at the engine level with `permission denied` (PostgreSQL) or `command denied` (MySQL), proving resilience even if application parsing were bypassed.
- **Engine Read-Only Enforcement**: Verified `SET TRANSACTION READ ONLY` on PostgreSQL and `PRAGMA query_only = ON` on SQLite.

### 2. Multi-Statement & Stacked Injection Attacks
- Evaluated stacked attacks (`SELECT 1; DROP TABLE customers;`, `-- \n DROP TABLE orders;`, and `/* safe */ ; DROP TABLE customers;`).
- In all engines, AST validation enforces exactly one top-level statement and rejects command batching.

### 3. Unauthorized Table & System Schema Boundaries
- Tested queries against unauthorized table `secret_passwords`. Both application allowlisting and database-level grants reject access.
- Tested system catalog queries (`SELECT * FROM pg_catalog.pg_tables`, `SELECT * FROM mysql.user`, `sqlite_master`). Rejected by `SYSTEM_SCHEMAS` protection and allowlists.
- Tested schema-qualified access (`hr_schema.employees`). Permitted when explicitly included in `allowed_tables`, blocked when omitted.

### 4. Dangerous Function Execution
- Blocked administrative and filesystem exfiltration functions:
  - PostgreSQL: `pg_sleep`, `pg_read_file`, `pg_ls_dir`, `pg_terminate_backend`, `dblink`.
  - MySQL: `sleep`, `load_file`, `sys_eval`.
  - SQLite: `load_extension`, `writefile`.

### 5. Statement Timeouts, Cancellation & Connection Pool Reuse
- **PostgreSQL**: Heavy cartesian joins cancelled within 200ms via `SET LOCAL statement_timeout`. Handled as `QueryTimeoutError`.
- **MySQL**: Expensive hash-sorted join cancelled within 200ms via `SET SESSION max_execution_time`. Cleanly reset to 0 in `finally` block.
- **SQLite**: Infinite recursive CTE interrupted via `set_progress_handler`. Reset to `None` in `finally` block.
- **Connection Pool State**: Verified that subsequent queries on the same connection pool execute normally after a timeout or query execution failure, with zero hung connections or state leakage.

### 6. Row Caps & Server-Side Bounding
- Evaluated 2,500-row table (`analytics_log`).
- Verified AST rewriting adds `LIMIT (max_rows + 1)`.
- Verified client-side `fetchmany(max_rows + 1)` truncates at `max_rows` and sets `is_truncated = True`.
- Verified queries with user-supplied small `LIMIT` return without truncation warnings (`is_truncated = False`).
- Verified exact row boundary edge case (`LIMIT 5` on 5-row table correctly reports `is_truncated = False`).

### 7. Dialect Syntax & Edge Cases
- Recursive CTEs (`WITH RECURSIVE cnt ...`) verified on PostgreSQL and MySQL.
- Dialect-specific quoted identifiers (`"customers"` on PostgreSQL, `` `customers` `` on MySQL) verified.
- Dialect-specific search syntax (`ILIKE` on PostgreSQL) verified.
- Malformed syntax (`SELECT FROM WHERE;`) cleanly intercepted as `SQLSecurityViolation`.

---

## 4. Coverage Thresholds & Code Coverage Analysis

Coverage is measured via `pytest-cov` across the `schemasense` package:

| Module | Statements | Missing Lines | Coverage % |
|---|---|---|---|
| `schemasense/__init__.py` | 1 | 0 | **100%** |
| `schemasense/profiler.py` | 95 | 3 | **97%** |
| `schemasense/schema_inspector.py` | 146 | 8 | **95%** |
| `schemasense/sql_guard.py` | 189 | 22 | **88%** |
| `schemasense/rules_engine.py` | 40 | 6 | **85%** |
| `schemasense/connection.py` | 156 | 28 | **82%** |
| **TOTAL** | **627** | **67** | **89%** |

### Quality Gate Thresholds:
- **Fast Unit Tests Stage (Isolated SQLite, No Containers)**: Baseline 75% -> Quality Gate Enforced: `--cov-fail-under=70`.
- **Full Live Integration Stage (Postgres 16 + MySQL 8.4)**: Baseline 89% -> Quality Gate Enforced: `--cov-fail-under=85`.

---

## 5. Mutation Testing Evaluation

During Milestone 5, mutation testing was formally evaluated using `mutmut` (v3.8.0) to assess test suite assertion sensitivity:

### Evaluation Findings:
1. **Platform Limitations**: `mutmut` native execution on Windows is unsupported (tracked in upstream issue [#397](https://github.com/boxed/mutmut/issues/397)), requiring WSL or a Linux runner (`ubuntu-latest`).
2. **Computational Complexity & Execution Time**: Mutating `schemasense/sql_guard.py` alone produces over 140 AST mutants. Executing the test suite for each mutant introduces significant overhead (minutes per module).
3. **Quality Recommendation**: Mutation testing is valuable for validating that security assertions cannot pass trivially. However, it should NOT be run as a blocking synchronous gate on every pull request due to runner queuing and token limits.
4. **Implementation Plan**: In Milestone 6+, mutation testing will run as a scheduled weekly GitHub Actions workflow targeting `schemasense/sql_guard.py` and `schemasense/connection.py`.

---

## 6. Reproducible Test Execution

### 1. Launch Disposable Containers
```bash
docker compose -f docker-compose.test.yml up -d --wait
```

### 2. Run Test Suites
```bash
# Fast unit tests only (no docker required):
pytest -v --ignore=tests/test_live_integration.py --cov=schemasense

# Full suite including live PostgreSQL & MySQL verification:
pytest -v --cov=schemasense --cov-report=term-missing

# Run linting check:
ruff check schemasense/ tests/
```

### 3. Teardown Disposable Containers
```bash
docker compose -f docker-compose.test.yml down -v
```

---

## 7. Current Test Status
- **SQLGuard & AST Tests (`test_sql_guard.py`)**: 50 Passed
- **Connection & Resource Tests (`test_connection.py`)**: 12 Passed
- **Schema Inspector Tests (`test_inspector.py`)**: 3 Passed
- **Data Profiler Tests (`test_profiler.py`)**: 3 Passed
- **Rules Engine Tests (`test_rules_engine.py`)**: 4 Passed
- **Live Integration Tests (`test_live_integration.py`)**: 43 Passed
- **Streamlit AppTest UI Tests (`test_ui_apptest.py`)**: 4 Passed
- **Total Tests**: **119 Passed / 0 Failed (100% Pass Rate)**
- **Ruff Lint Errors**: **0 Errors**
