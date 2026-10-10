"""Database connection lifecycle, least-privilege inspection, and safe URL creation."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

logger = logging.getLogger("schemasense.connection")


class ConnectionConfigError(ValueError):
    """Raised when connection parameters are missing or invalid."""


class DriverMissingError(ImportError):
    """Raised when a required DBAPI driver is not installed."""


class DatabaseConnectionError(RuntimeError):
    """Raised when connection to database server fails."""


class PrivilegedAccountError(PermissionError):
    """Raised when an operation requires a least-privilege account but a privileged account is detected."""


class ConcurrencyLimitExceeded(RuntimeError):
    """Raised when active database query concurrency budget is exhausted."""


SUPPORTED_DIALECTS = {
    "PostgreSQL": "postgresql+psycopg2",
    "MySQL": "mysql+pymysql",
    "SQLite": "sqlite",
}

# ─────────────────────────────────────────────────────────────
# RESOURCE BOUNDING: APPLICATION QUERY CONCURRENCY
# ─────────────────────────────────────────────────────────────

MAX_CONCURRENT_QUERIES = 4
QUERY_CONCURRENCY_TIMEOUT_SEC = 5.0
_QUERY_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_QUERIES)


@contextmanager
def acquire_query_budget(timeout_sec: float = QUERY_CONCURRENCY_TIMEOUT_SEC):
    """Context manager enforcing bounded database query concurrency across application threads."""
    acquired = _QUERY_SEMAPHORE.acquire(timeout=timeout_sec)
    if not acquired:
        raise ConcurrencyLimitExceeded(
            f"Query concurrency limit of {MAX_CONCURRENT_QUERIES} active executions reached. "
            "Please wait for in-flight queries to finish."
        )
    try:
        yield
    finally:
        _QUERY_SEMAPHORE.release()


# ─────────────────────────────────────────────────────────────
# LEAST-PRIVILEGE ACCOUNT INSPECTION
# ─────────────────────────────────────────────────────────────


@dataclass
class AccountPrivilegeReport:
    """Report detailing database-level permissions and least-privilege adherence."""

    is_privileged: bool
    reasons: list[str] = field(default_factory=list)
    current_user: str = "unknown"
    db_type: str = ""

    @property
    def summary(self) -> str:
        if not self.is_privileged:
            return (
                f"Verified: Connected with least-privilege role '{self.current_user}'."
            )
        return (
            f"Privileged database account detected for '{self.current_user}': "
            + "; ".join(self.reasons)
            + ". Use a restricted read-only account (e.g. 'schemasense_readonly')."
        )


def inspect_account_privileges(
    engine: Engine, db_type: str = "PostgreSQL"
) -> AccountPrivilegeReport:
    """
    Inspects database engine catalogs and grants to detect superuser or excessive administrative privileges.
    Does NOT rely on username heuristics (e.g., 'admin' or 'root' string matching).
    """
    clean_type = db_type.strip()
    reasons: list[str] = []
    current_user = "unknown"

    with engine.connect() as conn:
        if clean_type.lower() in ("postgresql", "postgres"):
            try:
                current_user = (
                    conn.execute(text("SELECT CURRENT_USER")).scalar() or "unknown"
                )
            except (SQLAlchemyError, DBAPIError):
                pass

            # 1. Check PostgreSQL runtime setting
            try:
                is_super = (
                    conn.execute(
                        text("SELECT current_setting('is_superuser')")
                    ).scalar()
                    == "on"
                )
                if is_super:
                    reasons.append(
                        "PostgreSQL superuser status is active (current_setting('is_superuser') = 'on')"
                    )
            except (SQLAlchemyError, DBAPIError):
                pass

            # 2. Check pg_roles attributes
            try:
                role_row = conn.execute(
                    text(
                        "SELECT rolsuper, rolcreaterole, rolcreatedb FROM pg_roles WHERE rolname = CURRENT_USER"
                    )
                ).fetchone()
                if role_row:
                    if role_row[0]:
                        reasons.append("Superuser role flag (rolsuper = true)")
                    if role_row[1]:
                        reasons.append(
                            "Role creation permission (rolcreaterole = true)"
                        )
                    if role_row[2]:
                        reasons.append(
                            "Database creation permission (rolcreatedb = true)"
                        )
            except (SQLAlchemyError, DBAPIError):
                pass

            # 3. Check database-level CREATE privilege
            try:
                can_create = conn.execute(
                    text(
                        "SELECT has_database_privilege(CURRENT_USER, current_database(), 'CREATE')"
                    )
                ).scalar()
                if can_create:
                    reasons.append(
                        "Database modification permission (CREATE on current database)"
                    )
            except (SQLAlchemyError, DBAPIError):
                pass

            return AccountPrivilegeReport(
                is_privileged=len(reasons) > 0,
                reasons=list(dict.fromkeys(reasons)),
                current_user=str(current_user),
                db_type="PostgreSQL",
            )

        elif clean_type.lower() == "mysql":
            try:
                current_user = (
                    conn.execute(text("SELECT CURRENT_USER()")).scalar() or "unknown"
                )
            except (SQLAlchemyError, DBAPIError):
                pass

            try:
                grants = [
                    row[0] for row in conn.execute(text("SHOW GRANTS")).fetchall()
                ]
                for grant in grants:
                    g_upper = grant.upper()
                    if (
                        "ALL PRIVILEGES ON *.*" in g_upper
                        or "ALL PRIVILEGES ON `*`.*" in g_upper
                    ):
                        reasons.append(
                            "Global ALL PRIVILEGES on entire MySQL server (*.*)"
                        )
                    elif "ALL PRIVILEGES" in g_upper:
                        reasons.append("Database-wide ALL PRIVILEGES")
                    if "SUPER" in g_upper:
                        reasons.append("SUPER administrative privilege")
                    if "SYSTEM_VARIABLES_ADMIN" in g_upper:
                        reasons.append("SYSTEM_VARIABLES_ADMIN privilege")
                    if "SHUTDOWN" in g_upper:
                        reasons.append("Server SHUTDOWN privilege")
                    if "WITH GRANT OPTION" in g_upper:
                        reasons.append("Privilege delegation (WITH GRANT OPTION)")
                    if (
                        "INSERT" in g_upper
                        or "UPDATE" in g_upper
                        or "DELETE" in g_upper
                        or "DROP" in g_upper
                    ) and "ON *.*" in g_upper:
                        reasons.append("Global write privileges on all databases (*.*)")
            except (SQLAlchemyError, DBAPIError) as e:
                logger.warning(f"Failed to inspect MySQL grants: {e}")

            return AccountPrivilegeReport(
                is_privileged=len(reasons) > 0,
                reasons=list(dict.fromkeys(reasons)),
                current_user=str(current_user),
                db_type="MySQL",
            )

        else:
            # SQLite / embedded
            return AccountPrivilegeReport(
                is_privileged=False,
                reasons=[],
                current_user="local_file",
                db_type="SQLite",
            )


# ─────────────────────────────────────────────────────────────
# SAFE CONNECTION & POOL MANAGEMENT
# ─────────────────────────────────────────────────────────────


def build_safe_url(
    db_type: str,
    host: str,
    port: int | str,
    db_name: str,
    user: str,
    password: str,
) -> URL:
    """Build a safe SQLAlchemy URL object without raw string concatenation."""
    if db_type not in SUPPORTED_DIALECTS:
        raise ConnectionConfigError(
            f"Unsupported database type '{db_type}'. Supported: {', '.join(SUPPORTED_DIALECTS.keys())}"
        )

    if not all([host, db_name, user, password]):
        raise ConnectionConfigError(
            "Host, database name, username, and password are all required."
        )

    try:
        clean_port = int(str(port).strip())
        if clean_port <= 0 or clean_port > 65535:
            raise ValueError()
    except (ValueError, TypeError):
        raise ConnectionConfigError(
            f"Port must be an integer between 1 and 65535 (got '{port}')"
        )

    driver = SUPPORTED_DIALECTS[db_type]
    clean_host = str(host).strip()
    clean_db = str(db_name).strip()
    clean_user = str(user).strip()

    query_params = {}
    if db_type == "MySQL":
        query_params["charset"] = "utf8mb4"

    return URL.create(
        drivername=driver,
        username=clean_user,
        password=str(password),
        host=clean_host,
        port=clean_port,
        database=clean_db,
        query=query_params,
    )


def create_safe_engine(
    db_type: str,
    host: str,
    port: int | str,
    db_name: str,
    user: str,
    password: str,
    connect_timeout: int = 10,
    enforce_least_privilege: bool = False,
) -> Engine:
    """
    Creates a configured SQLAlchemy Engine with bounded connection pool and connection validation.

    Args:
        db_type: Database type ('PostgreSQL', 'MySQL', 'SQLite')
        host: Host IP or hostname
        port: Port number
        db_name: Database name
        user: Username
        password: Password
        connect_timeout: Network connect timeout in seconds
        enforce_least_privilege: If True, rejects privileged accounts with PrivilegedAccountError
    """
    url = build_safe_url(db_type, host, port, db_name, user, password)

    connect_args = {"connect_timeout": connect_timeout}
    if db_type == "MySQL":
        connect_args.update(
            {
                "read_timeout": connect_timeout * 3,
                "write_timeout": connect_timeout * 3,
            }
        )

    # Bounded pool parameters for networked databases
    pool_kwargs = {
        "pool_pre_ping": True,
        "pool_recycle": 1800,
    }
    if db_type in ("PostgreSQL", "MySQL"):
        pool_kwargs.update(
            {
                "pool_size": 5,
                "max_overflow": 2,
                "pool_timeout": 10,
            }
        )

    try:
        engine = create_engine(
            url,
            connect_args=connect_args,
            **pool_kwargs,
        )
        # Test connection immediately
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        # Inspect and attach privilege report
        report = inspect_account_privileges(engine, db_type=db_type)
        engine._schemasense_privilege_report = report

        if enforce_least_privilege and report.is_privileged:
            dispose_engine(engine)
            raise PrivilegedAccountError(f"Connection rejected: {report.summary}")

        return engine

    except ImportError as ie:
        err = str(ie).lower()
        if "pymysql" in err:
            raise DriverMissingError(
                "MySQL driver (pymysql) is missing. Install via: pip install pymysql"
            ) from ie
        if "psycopg2" in err:
            raise DriverMissingError(
                "PostgreSQL driver (psycopg2) is missing. Install via: pip install psycopg2-binary"
            ) from ie
        raise
    except PrivilegedAccountError:
        raise
    except SQLAlchemyError as se:
        # Avoid leaking full passwords in error messages
        err_msg = str(se)
        if str(password) in err_msg and password:
            err_msg = err_msg.replace(str(password), "********")
        raise DatabaseConnectionError(f"Database connection failed: {err_msg}") from se


def dispose_engine(engine: Engine | None) -> None:
    """Safely disposes an existing SQLAlchemy engine to close all pool sockets."""
    if engine is not None:
        try:
            engine.dispose()
            logger.info("Successfully disposed database engine connection pool.")
        except (SQLAlchemyError, RuntimeError) as e:
            logger.warning(f"Error disposing database engine: {e}")
