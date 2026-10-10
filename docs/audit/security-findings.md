# Security Findings & Residual Risk Register

This document tracks all verified security findings, forensic remediations, and residual risks for SchemaSense AI.

## Verified Risk Register

| Finding ID | Severity | Category | Description | Status |
|---|---|---|---|---|
| **SEC-01** | CRITICAL (P0) | SQL Security | Regex-based query filter bypass (multi-statement, CTE writes, functions) | **RESOLVED** (`schemasense.sql_guard` AST parser) |
| **SEC-02** | HIGH (P0) | Reliability / DoS | Unbounded query execution and result sets without timeouts or caps | **RESOLVED** (5s timeout & 1,000 row cap) |
| **SEC-03** | HIGH (P1) | Supply Chain | Runtime `subprocess.check_call` to install `pymysql` | **RESOLVED** (Removed runtime package managers) |
| **SEC-04** | MEDIUM (P1) | AI Security | Schema metadata injection & unconstrained prompt exposure | **RESOLVED** (Selective table filter & XML delimiter tagging) |
| **SEC-05** | MEDIUM (P1) | Concurrency | Thread-unsafe global `genai.configure()` state across sessions | **RESOLVED** (Per-session key isolation; removed top-level call) |
| **SEC-06** | MEDIUM (P2) | DB Engine | SQLAlchemy connection pool leakage on re-connection | **RESOLVED** (`dispose_engine` helper) |
| **SEC-07** | HIGH (P0) | SQL Security | Table allowlist schema spoofing bypass (`pg_catalog.customers`) | **RESOLVED** (Protected system schemas + exact schema checks) |
| **SEC-08** | HIGH (P0) | SQL Security | Cross-dialect dangerous functions (`load_file`, `load_extension`, `pg_ls_dir`) | **RESOLVED** (Expanded `DISALLOWED_FUNCTIONS` across dialects) |
| **SEC-09** | HIGH (P1) | Defense in Depth | Unscoped timeouts & missing DB-level read-only mode | **RESOLVED** (PRAGMA query_only, SET TRANSACTION READ ONLY, timeout resets) |
| **SEC-10** | MEDIUM (P2) | Privacy | Full schema dumping to external LLM without table selection | **RESOLVED** (Sidebar table privacy multi-select) |

---

## Forensic Vulnerability Analysis (Milestone 2 Updates)

### SEC-07: Table Allowlist Schema Spoofing Bypass
- **Vulnerability:** In earlier versions, `bare_table in norm_allowed` was checked on any extracted table. If `customers` was allowed in the user's database, an attacker could query `SELECT * FROM pg_catalog.customers` or `SELECT * FROM secret_schema.customers`. Because the bare table was `customers`, the check evaluated to `True`.
- **Exploitation Impact:** Allowed reading internal system tables, catalog tables, or tables in other database schemas if they shared a name with an allowed table.
- **Remediation:** 
  1. Defined `SYSTEM_SCHEMAS = {"pg_catalog", "information_schema", "mysql", "performance_schema", "sys", "sqlite_master", "sqlite_schema"}`. Queries referencing system schemas are rejected unless explicitly authorized.
  2. If a schema is specified (`schema.table`), it must strictly match an allowed schema-qualified table or the standard user schema (`public` / `main`). Foreign schemas are rejected immediately.
- **Verification:** Regression test `test_schema_spoofing_prevented` and adversarial test suite in `tests/test_sql_guard.py`.

### SEC-08: Dialect-Specific Dangerous Functions
- **Vulnerability:** MySQL file reads (`LOAD_FILE('/etc/passwd')`), SQLite dynamic library loading (`load_extension('evil.so')`), and PostgreSQL directory listings (`pg_ls_dir('/tmp')`) were not in the initial blocklist.
- **Exploitation Impact:** Arbitrary file reads on MySQL hosts, remote code execution via SQLite extensions, and host inspection via PostgreSQL.
- **Remediation:** Expanded `DISALLOWED_FUNCTIONS` to include `load_file`, `sys_eval`, `sys_exec`, `load_extension`, `readfile`, `writefile`, `pg_ls_dir`, `pg_stat_file`, `dblink`, `dblink_exec`.
- **Verification:** Parameterized regression tests in `tests/test_sql_guard.py`.

### SEC-09: Unscoped Timeouts & Database-Level Read-Only Mode
- **Vulnerability:**
  - In PostgreSQL, `SET LOCAL statement_timeout` outside an explicit transaction block is ineffective in autocommit mode.
  - In MySQL, `SET SESSION max_execution_time` altered the session without resetting, leaking the timeout across pooled connections.
  - In SQLite, statement timeouts were unenforced, allowing infinite recursion in CTEs (`WITH RECURSIVE bomb ...`).
- **Remediation:**
  - PostgreSQL: Executes within `with conn.begin():` with `SET TRANSACTION READ ONLY` and `SET LOCAL statement_timeout`. Both reset automatically when the transaction completes.
  - MySQL: Executes `SET SESSION TRANSACTION READ ONLY` and `SET SESSION max_execution_time = {timeout_ms}`, and resets `max_execution_time = 0` in a `finally` block.
  - SQLite: Executes `PRAGMA query_only = ON` to enforce database-level read-only mode, and attaches a progress handler (`set_progress_handler`) that interrupts queries exceeding the timeout threshold. Resets progress handler in a `finally` block.
- **Verification:** `test_execute_safe_query_timeout` with infinite recursive CTEs.

### SEC-10: Sensitive Metadata Disclosure & API Key Isolation
- **Vulnerability:** Connecting to an enterprise database with hundreds of tables dumped all table names and columns directly into the LLM system prompt. Global `genai.configure()` mutates process-wide configuration.
- **Remediation:**
  1. Added a sidebar table multi-select under **🛡️ Schema Privacy & Isolation**, allowing users to selectively choose which tables are shared with Google Gemini.
  2. Passed `table_filter` into `to_prompt_context()` so unselected tables are never transmitted to external providers.
  3. Added an optional password-masked `Gemini API Key` input in the sidebar with per-session state isolation.
  4. Removed module top-level process-wide `genai.configure()` invocations.

---

## Residual Risks & Remaining Limitations

1. **Database Role Permissions (Defense in Depth Boundary):**
   - *Limitation:* SchemaSense enforces application-level AST validation and session-level read-only commands (`SET TRANSACTION READ ONLY`, `PRAGMA query_only = ON`). However, database administrators **must always configure a read-only database user account** at the database server level (`GRANT SELECT ON ...`). Application-level guardrails must supplement, not replace, principle of least privilege at the database layer.
2. **PostgreSQL / MySQL Integration Tests on Live Instances:**
   - *Limitation:* The current test suite runs 64 unit and adversarial tests against SQLite in-memory fixtures with full dialect AST verification. Live containerized PostgreSQL and MySQL integration tests require Docker daemon or cloud test databases.
3. **Complex CTE Aliases Shadowing Built-In Tables:**
   - *Limitation:* If a user writes an extremely complex multi-level nested CTE that re-aliases an unauthorized table name multiple times, AST traversal verifies that every referenced external table matches the allowlist. Recursive CTEs without termination are caught by the 5-second timeout.
