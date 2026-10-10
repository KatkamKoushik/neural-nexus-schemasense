# Database Role Permissions & Least-Privilege Guide

SchemaSense operates on a defense-in-depth security model. Application-level AST validation (`sql_guard.py`) is the first layer of defense, but **must never be the sole defense**. 

Every live database connection in SchemaSense should use a strictly scoped, read-only database service account with zero write privileges.

---

## 1. PostgreSQL Least-Privilege Configuration

### Role Provisioning Script

Run as superuser or database owner on `PostgreSQL 14+`:

```sql
-- 1. Create the dedicated read-only role with a secure password
CREATE ROLE schemasense_readonly WITH LOGIN PASSWORD 'STRONG_RANDOMLY_GENERATED_PASSWORD';

-- 2. Grant connection rights to the target database
GRANT CONNECT ON DATABASE production_db TO schemasense_readonly;

-- 3. Grant schema usage (required to lookup objects in schema)
GRANT USAGE ON SCHEMA public TO schemasense_readonly;

-- 4. Grant SELECT on all existing tables in the schema
GRANT SELECT ON ALL TABLES IN SCHEMA public TO schemasense_readonly;

-- 5. Revoke access to sensitive or secret tables explicitly
REVOKE SELECT ON TABLE secret_passwords FROM schemasense_readonly;
REVOKE SELECT ON TABLE customer_credentials FROM schemasense_readonly;
REVOKE SELECT ON TABLE payment_tokens FROM schemasense_readonly;

-- 6. Ensure newly created tables in public also inherit read-only SELECT
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT SELECT ON TABLES TO schemasense_readonly;

-- 7. Explicitly prevent temporary table creation (prevents temp table disk attacks)
REVOKE CREATE ON SCHEMA public FROM schemasense_readonly;
```

### PostgreSQL Safeguard Behavior
- **Attempted writes (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`)**: Immediately rejected with `ERROR: permission denied for table <table_name>`.
- **Attempted access to sensitive revoked tables**: Immediately rejected with `ERROR: permission denied for table <table_name>`.
- **Session statement timeout**: Enforced at execution time via `SET LOCAL statement_timeout = '<ms>'` within an explicit transaction.

---

## 2. MySQL / MariaDB Least-Privilege Configuration

### User Provisioning Script

Run as administrative user on `MySQL 8.0+` / `MySQL 8.4 LTS`:

```sql
-- 1. Create dedicated user
CREATE USER IF NOT EXISTS 'schemasense_readonly'@'%' IDENTIFIED BY 'STRONG_RANDOMLY_GENERATED_PASSWORD';

-- 2. Clear any existing privileges
REVOKE ALL PRIVILEGES, GRANT OPTION FROM 'schemasense_readonly'@'%';

-- 3. Grant SELECT strictly on explicitly approved tables
-- CRITICAL: Avoid database-wide wildcard grants ('GRANT SELECT ON db.*') if sensitive tables exist,
-- because MySQL does not permit table-level REVOKE overrides on database-level wildcard grants.
GRANT SELECT ON production_db.customers TO 'schemasense_readonly'@'%';
GRANT SELECT ON production_db.orders TO 'schemasense_readonly'@'%';
GRANT SELECT ON production_db.products TO 'schemasense_readonly'@'%';
GRANT SELECT ON production_db.analytics_log TO 'schemasense_readonly'@'%';

-- 4. Apply changes
FLUSH PRIVILEGES;
```

### MySQL Safeguard Behavior
- **Attempted writes (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`)**: Immediately rejected with `ERROR 1142 (42000): INSERT command denied to user 'schemasense_readonly'@'...' for table '...'`.
- **Attempted access to ungranted tables**: Rejected with `ERROR 1142 (42000): SELECT command denied to user ...`.
- **Session statement timeout**: Enforced via `SET SESSION max_execution_time = <ms>`, with guaranteed reset to `0` in a `finally` block.

---

## 3. SQLite Least-Privilege Configuration

Because SQLite is an embedded engine without an integrated role-based authentication model, security is enforced via:

1. **File System Permissions**: Mount the SQLite database file as read-only (`chmod 444 db.sqlite`).
2. **Read-Only Connection URI**: Open the SQLite file using URI mode `file:db.sqlite?mode=ro`.
3. **Engine-Level Query-Only Mode**: SchemaSense executes `PRAGMA query_only = ON;` upon acquiring the connection, ensuring that any write attempt fails immediately with `attempt to write a readonly database`.
4. **Progress Handler Interrupt**: Statement timeouts are enforced using Python's SQLite `set_progress_handler` callback, interrupting runaway recursive CTEs or Cartesian joins without hanging the thread.

---

## Summary of Defense Layers

| Attack Vector | Layer 1: AST SQLGuard (`sqlglot`) | Layer 2: Database Role Privileges | Layer 3: Connection / Session Control |
|---|---|---|---|
| Destructive DDL (`DROP`, `TRUNCATE`) | Blocked before execution | Blocked (`permission denied`) | Read-only transaction |
| DML Writes (`INSERT`, `UPDATE`, `DELETE`) | Blocked before execution | Blocked (`permission denied`) | `SET TRANSACTION READ ONLY` / `PRAGMA query_only` |
| Multi-statement / Stacked SQL | Blocked (single-statement check) | N/A | Driver parameter isolation |
| Sensitive Table Access | Table allowlist filtering | Table grant exclusion | Active schema segregation |
| Runaway / Infinite Queries | AST Server-Side `LIMIT` clamping | Resource limits | Session `statement_timeout` & `set_progress_handler` |
| Disallowed DB Functions (`pg_read_file`, `load_file`) | AST function denylist | Restricted role lacks superuser rights | N/A |

---

## 5. Engine Limitations & Resource Exhaustion Realities

> [!WARNING]
> **Application-level checks alone do not prevent all expensive queries or database resource exhaustion.**
> While AST parsing prevents malicious statements (writes, drops, stacked queries, unauthorized tables), legitimate `SELECT` queries on permitted tables can still be computationally expensive or resource-intensive.

### Engine-Specific Technical Limitations:

1. **PostgreSQL Parameter Scopes (`SUSET` vs `USERSET`)**:
   - `temp_file_limit` is a `SUSET` (superuser-only) parameter. Attempting to execute `SET LOCAL temp_file_limit = '64MB'` under a least-privileged role causes `InsufficientPrivilege` and aborts the transaction block.
   - **Operational Mitigation**: DBAs must configure disk bounds at role creation:
     ```sql
     ALTER ROLE schemasense_readonly SET temp_file_limit = '64MB';
     ```
   - In contrast, `work_mem` is `USERSET` and is dynamically bounded by SchemaSense per session (`SET LOCAL work_mem = '16MB'`).

2. **MySQL Privilege Hierarchy**:
   - MySQL does not support hierarchical permission revokes. If a user is granted `GRANT SELECT ON production_db.*`, issuing `REVOKE SELECT ON production_db.secret_table` fails or has no effect.
   - **Operational Mitigation**: Never grant wildcard database privileges. Explicitly grant `SELECT` on each permitted business table individually.
   - MySQL's `MAX_EXECUTION_TIME` hint applies to read queries, but complex sorts requiring disk-based temporary tables can still create I/O pressure before the timeout fires.

3. **SQLite Embedded Scope**:
   - SQLite lacks role-based access control, network authentication, and user quotas.
   - Security relies completely on process-level filesystem permissions (`chmod 444`), connection URI modes (`mode=ro`), and the application's AST SQLGuard.

4. **Inherent Complexity of Analytical SQL**:
   - Even with `LIMIT 1000` AST clamping, an unindexed `JOIN` or massive aggregation (`SELECT c.*, o.* FROM customers c CROSS JOIN orders o`) must be computed by the database engine *before* the LIMIT can be applied to the result set.
   - Application-layer guardrails mitigate this via strict execution timeouts (`timeout_ms=5000`) and concurrency semaphores (`MAX_CONCURRENT_QUERIES=4`), but **database-level resource quotas, query budget limits, and read replicas are required for production multi-tenant environments**.

