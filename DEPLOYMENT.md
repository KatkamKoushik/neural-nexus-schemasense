# SchemaSense — Controlled Deployment Guide

This document describes the operational requirements, security guardrails, and deployment procedure for SchemaSense.

---

## 1. Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **Database Connectivity**: Access to target PostgreSQL, MySQL, or SQLite instance
- **AI Engine Key**: Google Gemini API key (`GEMINI_API_KEY`)

---

## 2. Security Prerequisites & Database Role Provisioning

SchemaSense operates on a defense-in-depth model. **Never connect using a root, superuser, or database owner account.**

1. **PostgreSQL**: Provision a dedicated read-only role:
   ```sql
   CREATE ROLE schemasense_readonly WITH LOGIN PASSWORD 'STRONG_PASSWORD';
   GRANT CONNECT ON DATABASE production_db TO schemasense_readonly;
   GRANT USAGE ON SCHEMA public TO schemasense_readonly;
   GRANT SELECT ON ALL TABLES IN SCHEMA public TO schemasense_readonly;
   ALTER ROLE schemasense_readonly SET temp_file_limit = '64MB';
   REVOKE CREATE ON SCHEMA public FROM schemasense_readonly;
   ```

2. **MySQL**: Provision a dedicated read-only user on approved tables:
   ```sql
   CREATE USER 'schemasense_readonly'@'%' IDENTIFIED BY 'STRONG_PASSWORD';
   GRANT SELECT ON production_db.customers TO 'schemasense_readonly'@'%';
   GRANT SELECT ON production_db.orders TO 'schemasense_readonly'@'%';
   FLUSH PRIVILEGES;
   ```

*See [docs/security/database-permissions.md](docs/security/database-permissions.md) for complete provisioning scripts.*

---

## 3. Installation & Verification

```bash
# 1. Clone repository
git clone https://github.com/KatkamKoushik/neural-nexus-schemasense.git
cd neural-nexus-schemasense

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# 3. Install pinned dependencies (reproducible with constraints)
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt -c constraints.txt

# (Optional) Install local offline inference engine (llama-cpp-python)
# CPU: pip install -r requirements-local.txt
# CUDA: CMAKE_ARGS="-DGGML_CUDA=on" pip install -r requirements-local.txt

# 4. Verify installation via automated test suite and coverage
pytest -v --ignore=tests/test_live_integration.py --cov=schemasense

# 5. Run supply-chain security checks
pip-audit -r requirements.txt
detect-secrets scan --baseline .secrets.baseline --exclude-files "(\.git|__pycache__|\.venv|venv|\.secrets\.baseline)"
cyclonedx-py requirements constraints.txt -o sbom.cyclonedx.json
```

---

## 4. Running the Application

### Local / Controlled Pilot
```bash
export GEMINI_API_KEY="your-api-key"
streamlit run app.py --server.port 8501
```

### Production / Containerized Pilot
- Deploy inside a private VPC subnet.
- Connect exclusively to read-only database replicas.
- Put Streamlit behind an enterprise Identity-Aware Proxy (e.g. Cloudflare Access, AWS ALB OIDC).

---

## 5. Security & Governance References
- [Threat Model & STRIDE Analysis](docs/security/threat-model.md)
- [Database Permissions & Least Privilege Guide](docs/security/database-permissions.md)
- [Supply-Chain Security & Vulnerability Triage Policy](docs/security/supply-chain.md)
- [Verification Plan & Adversarial Test Matrix](docs/testing/verification-plan.md)
- [Deployment Readiness Checklist & Risk Registry](docs/deployment/readiness-checklist.md)