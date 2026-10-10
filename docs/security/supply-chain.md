# SchemaSense — Dependency & Supply-Chain Security Policy

## 1. Overview & Strategy

SchemaSense employs defense-in-depth across the open-source software supply chain. While modern web applications rely on open-source packages for functionality (SQL parsing, database drivers, analytics, UI rendering), unmanaged dependencies introduce risks such as known Common Vulnerabilities and Exposures (CVEs), malicious package takeovers, transitive dependency drift, and credentials leakage.

Our supply-chain security architecture combines:
1. **Automated vulnerability scanning**: `pip-audit` and `osv-scanner` in CI.
2. **Reproducible builds**: Layered `requirements.txt`, `requirements-dev.txt`, and `constraints.txt`.
3. **Automated patch tracking**: Dependabot for automated dependency and GitHub Actions version monitoring.
4. **Machine-readable software bill of materials (SBOM)**: CycloneDX JSON generation for release transparency.
5. **Pre-commit and CI secret scanning**: `detect-secrets` with checked-in `.secrets.baseline`.

---

## 2. Dependency Architecture & Reproducibility

To balance stability with maintainability without overpinning every library, SchemaSense uses a three-tier dependency model:

| Artifact | Purpose | Example Entry |
|---|---|---|
| `requirements.txt` | Runtime application dependencies with tested lower/upper bounds. | `sqlalchemy>=2.0.0` |
| `requirements-dev.txt` | Development, linting, testing, and CI tooling. | `pytest-cov>=5.0.0`, `ruff>=0.5.0` |
| `constraints.txt` | Exact tested versions for reproducible deployment and release builds. | `sqlalchemy==2.0.38` |

### Reproducible Installation Procedure
For standard development:
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

For deterministic pilot deployments and CI runs:
```bash
pip install -r requirements.txt -c constraints.txt
```

---

## 3. Vulnerability Scanners Evaluation: pip-audit vs OSV-Scanner

SchemaSense evaluated two automated vulnerability scanners:

| Capability | `pip-audit` | Google `osv-scanner` | SchemaSense Integration |
|---|---|---|---|
| **Ecosystem Focus** | Python / PyPI dedicated. | Polyglot (Go, Rust, Python, npm, etc.). | Both evaluated; `pip-audit` runs natively in Python CI jobs, `osv-scanner` acts as complementary multi-format scanner. |
| **Data Source** | PyPA Advisory Database & OSV.dev. | Open Source Vulnerabilities (OSV.dev). | Both query curated OSV advisories. |
| **Input Format** | `requirements.txt`, active environment. | Lockfiles, SBOMs (`sbom.cyclonedx.json`), git commits. | `pip-audit` scans requirements; `osv-scanner` scans generated SBOM and lockfiles. |
| **Speed & Portability** | Native Python CLI via `pip`. | Standalone Go binary or GitHub Action. | Zero external non-Python dependencies for local runs with `pip-audit`. |

### Execution Commands:
- **pip-audit**:
  ```bash
  pip-audit -r requirements.txt
  ```
- **OSV-Scanner (via GitHub Actions or CLI)**:
  ```bash
  osv-scanner --sbom=sbom.cyclonedx.json
  ```

---

## 4. Software Bill of Materials (SBOM)

SchemaSense standardizes on the **CycloneDX** industry specification (OWASP Foundation) for software bill of materials generation.

### Release SBOM Generation Command:
```bash
cyclonedx-py requirements constraints.txt -o sbom.cyclonedx.json
```
The output file `sbom.cyclonedx.json` contains:
- Complete component names and versions.
- Package URLs (PURLs) conforming to `pkg:pypi/<package>@<version>`.
- Cryptographic package references and distribution endpoints.

---

## 5. Secret Scanning Policy

### Automated Secret Scanning
Committed database passwords, API keys, or private URLs pose severe operational risks.
- We utilize `detect-secrets` configured with `.secrets.baseline`.
- Baseline verification command:
  ```bash
  detect-secrets scan --baseline .secrets.baseline --exclude-files "(\.git|__pycache__|\.venv|venv|\.secrets\.baseline)"
  ```
- Any newly committed secret immediately fails local checks and CI quality gates.

### Historical Repository Audit Finding
During the Milestone 5 Phase 1 security audit, historical Git commit `6c520fe4` was identified as having temporarily committed `.streamlit/secrets.toml` with Gemini API keys and database credentials, subsequently removed in commit `b0a516a5`.
- **Active status**: Current working tree and release files contain zero active credentials.
- **Pilot boundary requirement**: Any credentials that existed in git history must be revoked and rotated. The internal pilot database instances must use freshly generated credentials never shared outside the internal environment.

---

## 6. Vulnerability Triage & Escalation SLA

When Dependabot, `pip-audit`, or `osv-scanner` identifies a vulnerability, engineering follows this triage matrix:

| Severity (CVSS v3) | Remediation SLA | Action Required |
|---|---|---|
| **Critical (9.0 - 10.0)** | 24 Hours | Immediate patch release; upgrade package or apply temporary hotfix constraint. |
| **High (7.0 - 8.9)** | 72 Hours | Prioritized bump; verify test suite and deploy updated build. |
| **Medium (4.0 - 6.9)** | 14 Days | Scheduled update in next regular sprint cycle. |
| **Low (0.1 - 3.9)** | 30 Days | Evaluated during routine weekly Dependabot updates. |

### False-Positive / Unreachable Code Triage
If an upstream advisory affects a package feature not imported or invoked by SchemaSense:
1. Document the unreachability analysis in an issue or PR.
2. If safe, prefer upgrading the dependency rather than ignoring the advisory.
3. If upgrading causes breaking changes, create a documented pip-audit ignore rule with an expiration date and ticket reference.

---

## 7. Upstream Ecosystem Advisory: google-generativeai

- `google-generativeai` has reached end-of-support upstream in favor of the newer `google-genai` SDK.
- SchemaSense isolates LLM integration behind `generate_with_fallback()` in `app.py`.
- **Roadmap Item**: Migration to `google-genai` is scheduled for Milestone 6 (Model Orchestration & Upstream SDK Modernization).

---

## 8. Critical Supply-Chain Disclaimer

> [!WARNING]
> **Vulnerability scanners and SBOMs do not guarantee application security.**
> - Scanners only detect *publicly disclosed, cataloged CVEs* in published versions.
> - Zero-day vulnerabilities, malicious code published before cataloging, and repository account takeovers are invisible to signature-based scanners.
> - Most critical database and application vulnerabilities (SQL injection, privilege escalation, authorization bypasses, and denial-of-service) stem from **application logic and system architecture**, not third-party dependency CVEs.
> - Supply-chain scanning is a necessary baseline control, not a substitute for architectural defense-in-depth, AST guardrails, and least-privilege database account enforcement.
