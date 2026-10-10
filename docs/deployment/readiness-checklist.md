# Deployment-Readiness Checklist & Prioritized Residual-Risk Report

## 1. Controlled Deployment Readiness Gate

- **Deployment Classification**: **APPROVED FOR CONTROLLED INTERNAL PILOT / DATA ANALYST ACCESS**
- **Public / Unauthenticated SaaS Status**: **NOT CERTIFIED** (Requires reverse proxy, enterprise SSO, and network microsegmentation).
- **Automated Verification Score**: **119 / 119 Tests Passing (100% Pass Rate)** across fast unit tests, adversarial SQL injection suite, live PostgreSQL (`16-alpine`), live MySQL (`8.4` LTS), and Streamlit headless AppTest UI workflows.
- **Code Coverage**: **89% Line Coverage** across `schemasense/` (627 statements, 67 missed) with strict CI quality thresholds.
- **Ruff Code Quality Status**: **0 Lint Errors** across all packages and test suites.
- **Dependency & Supply-Chain Scan**: **0 Known Vulnerabilities** (`pip-audit -r requirements.txt`).
- **Secret Scanning Status**: **0 Leaked Credentials** in working tree (`detect-secrets` verified with `.secrets.baseline`).
- **Software Bill of Materials (SBOM)**: CycloneDX JSON (`sbom.cyclonedx.json`) generated from reproducible `constraints.txt`.

---

## 2. Pre-Deployment Operational Checklist

### A. Database Provisioning & Security Controls
- [x] **Dedicated Read-Only Account**: Connected service account is `schemasense_readonly` (not `postgres`, `root`, or database owner).
- [x] **Verified Least-Privilege Report**: Application verifies `current_setting('is_superuser') = 'off'` (PostgreSQL) and absence of `SUPER` / `ALL PRIVILEGES` (MySQL).
- [x] **Interactive UI Policy Enforcement**: Sidebar UI provides "Enforce Least-Privilege Role" toggle that hard-rejects administrative connections with `PrivilegedAccountError`.
- [x] **Catalog & Sensitive Table Revocation**: Highly sensitive tables (`secret_passwords`, credential tables) explicitly excluded from `schemasense_readonly` grants.
- [x] **Role-Level Resource Limits**: PostgreSQL role configured with `ALTER ROLE schemasense_readonly SET temp_file_limit = '64MB';` to bound disk usage.
- [x] **Statement Timeouts Enabled**: Queries automatically capped at 5,000ms (clamped between 100ms and 30,000ms max).

### B. Application Resource Bounding
- [x] **Connection Pool Sizing**: Bounded to `pool_size = 5`, `max_overflow = 2`, `pool_timeout = 10s`, and `pool_recycle = 1800s`.
- [x] **Concurrency Budget Limiter**: Global query execution throttled via semaphore (`MAX_CONCURRENT_QUERIES = 4`) to prevent worker thread saturation.
- [x] **Row-Cap Clamping**: AST rewrites queries to inject `LIMIT 1001`; client-side `fetchmany` truncates results to 1,000 rows.
- [x] **Memory & Buffer Protection**: Session sets `work_mem = '16MB'` on PostgreSQL to constrain sort buffers.

### C. Secret & Session Hygiene
- [x] **Zero Hardcoded Secrets**: Zero credentials or API keys committed to working tree.
- [x] **Audited Secret Baseline**: Pre-commit and CI scans enforce `.secrets.baseline`.
- [x] **Input Masking**: Password fields and API keys set to `type="password"`.
- [x] **Password Stripping from Errors**: Driver exceptions scrub user passwords before displaying in logs or UI.
- [x] **Disconnection Lifecycle**: Disconnect button releases active engine via `dispose_engine()` and clears session state.

### D. Automated CI Guardrails
- [x] **Security & Supply-Chain Stage**: Automated `pip-audit`, `detect-secrets`, OSV-Scanner, and CycloneDX SBOM artifact generation.
- [x] **Fast Unit & Lint Stage**: Python 3.10, 3.11, 3.12 matrix testing Ruff, byte-compilation, and isolated SQLite tests with `--cov-fail-under=70`.
- [x] **Live Container Stage**: GitHub Actions service containers running `postgres:16-alpine` and `mysql:8.4` for automated adversarial verification.
- [x] **Reproducible Installation**: CI and pilot builds pinned via `constraints.txt`.

---

## 3. Prioritized Residual-Risk Registry

| Risk ID | Threat Scenario | Residual Severity | Likelihood | Technical Defense in Place | Recommended Operational Action |
|---|---|---|---|---|---|
| **RSK-01** | Superuser Credential Provisioning | **High** | Low (Internal) | Automatic catalog privilege detection (`inspect_account_privileges`) displays persistent security warnings in UI, and optional UI enforcement rejects them outright. | Enforce database infrastructure policy requiring DBAs to reject root/admin provisioning requests. |
| **RSK-02** | CPU Exhaustion via Complex Non-Sargable Joins | **Medium** | Medium | Bounded statement timeouts (default 5,000ms; max 30s) automatically abort execution before runaway degradation. | Monitor database CPU alarms and provision read replicas for high-traffic analytical workloads. |
| **RSK-03** | Temporary Disk Usage Spikes on Large Aggregations | **Low** | Medium | Session `work_mem` clamped to 16MB. Role-level `temp_file_limit` terminates queries exceeding 64MB disk. | Verify database host has sufficient ephemeral disk storage mounted for temporary query buffers. |
| **RSK-04** | Prompt Injection via Ingested Business PDFs | **Low** | Low | Extracted PDF business rules are treated strictly as reference text inside delimited XML blocks. All generated SQL passes through AST SQLGuard before execution. | Do not ingest untrusted or public PDF documents without administrative review. |
| **RSK-05** | Direct SQLite File-Level Write Risk | **Low** | Low | Enforced via `PRAGMA query_only = ON` and read-only URI connection mode (`mode=ro`). | Set operating system file permissions to read-only (`chmod 444`) on the SQLite file. |
| **RSK-06** | Upstream SDK Deprecation (`google-generativeai`) | **Low** | Medium | SDK is deprecated upstream in favor of `google-genai`. Code is isolated behind `generate_with_fallback`. | Upgrade to `google-genai` in Milestone 6. |

---

## 4. Controlled Deployment Architecture Recommendation

For pilot deployment within an internal corporate network:
1. **Host Environment**: Deploy SchemaSense container or virtual machine inside a private VPC.
2. **Access Control**: Put Streamlit behind an enterprise identity-aware proxy (e.g., Cloudflare Access, AWS ALB with OIDC, or VPN).
3. **Database Target**: Connect exclusively to a read-only database replica using the `schemasense_readonly` account.
4. **LLM Connection**: Configure `GEMINI_API_KEY` via container environment variables or AWS Secrets Manager / GCP Secret Manager.
