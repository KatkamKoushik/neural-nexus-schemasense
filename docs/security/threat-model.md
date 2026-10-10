# SchemaSense Threat Model & Security Architecture

## 1. System Overview & Scope

SchemaSense is an enterprise database exploration and data intelligence assistant. It interfaces with PostgreSQL, MySQL, and SQLite databases to extract schema structures, profile data quality, and execute user/AI-generated read-only analytical queries.

### Architecture Trust Zones

```
[ External User / Web Client ]
              │ (HTTP / TLS)
              ▼
╔════════════════════════════════════════════════════════════════╗
║  ZONE 1: Presentation & Session (Streamlit runtime)           ║
║  - Credential masking (type="password")                        ║
║  - Transient st.session_state (zero disk credential logging)  ║
║  - Concurrency budget limiter (MAX_CONCURRENT_QUERIES = 4)     ║
╚════════════════════════════════════════════════════════════════╝
              │ Python In-Process Calls
              ▼
╔════════════════════════════════════════════════════════════════╗
║  ZONE 2: Application Security Guardrails (sql_guard.py)       ║
║  - AST-level validation (sqlglot)                             ║
║  - Strict single-statement enforcement (no stacked queries)   ║
║  - Disallowed administrative / filesystem function blocking    ║
║  - Protected system catalog blocking (pg_catalog, mysql, etc.)║
║  - Table allowlisting & schema-spoofing prevention            ║
║  - Server-side LIMIT clamping (max 5,000 rows)                ║
╚════════════════════════════════════════════════════════════════╝
              │ SQLAlchemy Connection Pool (Bounded QueuePool)
              ▼
╔════════════════════════════════════════════════════════════════╗
║  ZONE 3: Session & Driver Controls                            ║
║  - Scoped statement timeouts (PostgreSQL / MySQL / SQLite)     ║
║  - Session read-only modes (SET TRANSACTION READ ONLY)        ║
║  - Memory bounds (SET LOCAL work_mem = '16MB')                 ║
║  - Guaranteed cleanup in finally blocks                       ║
╚════════════════════════════════════════════════════════════════╝
              │ TCP / Network Socket
              ▼
╔════════════════════════════════════════════════════════════════╗
║  ZONE 4: Database Engine Role Privileges                      ║
║  - Least-privilege role (schemasense_readonly)                 ║
║  - Engine-level denial of INSERT, UPDATE, DELETE, DDL         ║
║  - Revocation of secret / sensitive tables at catalog level    ║
╚════════════════════════════════════════════════════════════════╝
```

---

## 2. Threat Actors & Asset Inventory

### Threat Actors
1. **Malicious or Compromised Analyst**: Attempts destructive DDL (`DROP`), unauthorized DML (`DELETE`, `UPDATE`), or privilege escalation (`GRANT`).
2. **Adversarial External Input (Prompt Injection)**: Untrusted business documents (PDFs) or prompt injection vectors attempting to force the AI model into generating destructive SQL.
3. **Denial-of-Service (DoS) Attacker**: Submits Cartesian cross-joins, unindexed expressions, or runaway recursive CTEs to exhaust database memory, CPU, or temporary disk.
4. **Credential Sniffer**: Attempts to extract database passwords or API keys from logs, tracebacks, or web UI state.

### Protected Assets
- **Database Data Integrity**: Absolute protection against writes, schema tampering, table truncation, or drops.
- **Sensitive / Audit Data**: Protection of unapproved tables (e.g. `secret_passwords`, credentials, PCI tokens).
- **Database Compute & Memory Stability**: Protection against server exhaustion via unbounded queries or concurrency surges.
- **Application & Service Credentials**: Zero leakage of database passwords or Gemini API keys.

---

## 3. STRIDE Threat Analysis & Defense Matrix

| Threat Category | Potential Attack Vector | Layer 1: Application AST Guardrail | Layer 2: Driver / Session Controls | Layer 3: Database Role Privileges |
|---|---|---|---|---|
| **Spoofing** | Attacker spoofs schema name (`pg_catalog.customers` or `secret_schema.tbl`) | Rejected by `SYSTEM_SCHEMAS` and table allowlist matching | Isolated connection pool | Schema USAGE permissions revoked |
| **Tampering** | Destructive DDL (`DROP`, `TRUNCATE`, `ALTER`) or DML writes (`INSERT`, `UPDATE`, `DELETE`) | Blocked by AST node check (`FORBIDDEN_AST_NODES`) | `SET TRANSACTION READ ONLY` / `PRAGMA query_only = ON` | Engine blocks: `permission denied for table` |
| **Repudiation** | Submitting batch or stacked statements (`SELECT 1; DROP TABLE users;`) | Blocked: AST requires exactly one root expression | DBAPI parameterization prevents driver stacking | Database user lacks DDL rights |
| **Information Disclosure** | Reading local files (`pg_read_file`, `load_file`), directory listing (`pg_ls_dir`), or secret tables | Blocked: `DISALLOWED_FUNCTIONS` denylist + table allowlist | Connection string password masking in error strings | Specific table `REVOKE SELECT` at database level |
| **Denial of Service** | Cartesian product joins, deep recursive CTEs, or unconstrained table dumps | Server-side AST LIMIT rewrite (`LIMIT max_rows + 1`) | Scoped `statement_timeout` & `set_progress_handler` with reset | Bounded `work_mem` and role-level `temp_file_limit` |
| **Elevation of Privilege** | User connects with superuser account (`postgres`, `root`) | Non-heuristic catalog privilege inspection detects and warns | `enforce_least_privilege` flag rejects connection | Administrative tasks prohibited from SchemaSense |

---

## 4. Deployment Assumptions & Operational Requirements

1. **Dedicated Database Service Accounts**: SchemaSense **must** connect using a dedicated read-only role (`schemasense_readonly`). Deployments must not provide root/superuser credentials.
2. **Network Isolation & VPC Peering**: The database server should reside within a private subnet or VPC, accessible only to the SchemaSense host via firewall rules or TLS tunnels.
3. **Transient Memory Processing**: Data extracted during queries is held in Streamlit process memory for session visualization and is not persisted to disk.
4. **LLM Provider Redaction**: Schema privacy controls allow operators to de-select sensitive tables in the sidebar so their schema and metadata are never sent to external LLMs.

---

## 5. Known Limitations & Residual Risks

1. **Subquery Execution Cost**: While client-side memory is protected by `fetchmany` and server-side results are bounded by AST `LIMIT`, an unindexed `ORDER BY` across billions of rows will consume database CPU until the statement timeout aborts it.
2. **SQLite Multi-User Limitations**: SQLite lacks user-level privilege separation; protection relies on read-only URI mount (`mode=ro`), `PRAGMA query_only = ON`, and progress handler interrupts.
3. **Database-Specific Function Evolution**: New procedural or administrative extensions added to database engines may not appear in the static `DISALLOWED_FUNCTIONS` list; this is mitigated by database role-level privilege revocation.
