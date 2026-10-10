"""SQLGuard — AST-based SQL parser, security validator, and bounded executor."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd
import sqlglot
from sqlalchemy import Engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlglot import exp
from sqlglot.errors import ParseError

from schemasense.connection import ConcurrencyLimitExceeded, acquire_query_budget

logger = logging.getLogger("schemasense.sql_guard")


class SQLSecurityViolation(ValueError):
    """Raised when an SQL query violates security guardrails."""


class QueryTimeoutError(TimeoutError):
    """Raised when an SQL query exceeds the enforced execution timeout."""


class QueryExecutionError(RuntimeError):
    """Raised when query execution fails in the database engine."""


__all__ = [
    "ConcurrencyLimitExceeded",
    "QueryExecutionError",
    "QueryTimeoutError",
    "SQLGuard",
    "SQLSecurityViolation",
    "ValidationResult",
    "execute_safe_query",
]


@dataclass
class ValidationResult:
    is_valid: bool
    error_message: str | None = None
    extracted_tables: list[str] = field(default_factory=list)
    normalized_sql: str | None = None
    ast_root: exp.Expression | None = None


# Explicitly forbidden AST expression classes across all dialects
FORBIDDEN_AST_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.TruncateTable,
    exp.Merge,
    exp.Grant,
    exp.Revoke,
    exp.Command,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Into,
    exp.Copy,
)

# High-risk functions across PostgreSQL, MySQL, and SQLite
DISALLOWED_FUNCTIONS = {
    # PostgreSQL file access, signaling, and administrative routines
    "pg_read_file",
    "pg_read_binary_file",
    "pg_write_file",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "pg_ls_dir",
    "pg_stat_file",
    "pg_rotate_logfile",
    "pg_reload_conf",
    "lo_export",
    "lo_import",
    "lo_unlink",
    "lo_creat",
    "lo_open",
    "lo_write",
    "dblink",
    "dblink_exec",
    "dblink_connect",
    # MySQL execution and file routines
    "load_file",
    "sys_eval",
    "sys_exec",
    "sys_get",
    "sys_set",
    # SQLite dynamic extensions and CLI file utilities
    "load_extension",
    "readfile",
    "writefile",
    # Denial of service sleep/benchmark routines
    "benchmark",
    "sleep",
    "pg_sleep",
}

# Protected database system catalog schemas
SYSTEM_SCHEMAS = {
    "pg_catalog",
    "information_schema",
    "mysql",
    "performance_schema",
    "sys",
    "sqlite_master",
    "sqlite_schema",
}

DIALECT_MAPPING = {
    "postgresql": "postgres",
    "postgres": "postgres",
    "mysql": "mysql",
    "sqlite": "sqlite",
}


class SQLGuard:
    """Security validator enforcing AST single-statement read-only rules."""

    @classmethod
    def normalize_dialect(cls, dialect: str) -> str:
        clean = (dialect or "").strip().lower()
        return DIALECT_MAPPING.get(clean, "postgres")

    @classmethod
    def validate(
        cls,
        query: str,
        dialect: str = "postgres",
        allowed_tables: Sequence[str] | set[str] | None = None,
    ) -> ValidationResult:
        """Parses and validates that query is a single read-only SELECT against allowed tables."""
        if not query or not query.strip():
            return ValidationResult(
                is_valid=False, error_message="Empty query provided."
            )

        clean_sql = query.strip()
        read_dialect = cls.normalize_dialect(dialect)

        # 1. Parse AST
        try:
            parsed_expressions = sqlglot.parse(clean_sql, read=read_dialect)
        except ParseError as pe:
            return ValidationResult(
                is_valid=False, error_message=f"SQL syntax error: {pe}"
            )
        except (ValueError, TypeError, KeyError) as e:
            return ValidationResult(
                is_valid=False, error_message=f"Failed to parse SQL: {e}"
            )

        # Filter out empty statements (e.g. trailing semicolons)
        statements = [expr for expr in parsed_expressions if expr is not None]

        if not statements:
            return ValidationResult(
                is_valid=False, error_message="No valid SQL statements found."
            )

        # 2. Enforce strictly single-statement execution
        if len(statements) > 1:
            return ValidationResult(
                is_valid=False,
                error_message="Multi-statement execution is strictly prohibited. Exactly one query is permitted.",
            )

        root = statements[0]

        # 3. Verify root expression is a SELECT or UNION
        if not isinstance(root, (exp.Select, exp.Union)):
            return ValidationResult(
                is_valid=False,
                error_message=f"Only read-only SELECT queries are permitted (received {root.__class__.__name__}).",
            )

        # 4. Check entire AST tree for forbidden expressions (including CTEs and subqueries)
        for forbidden_type in FORBIDDEN_AST_NODES:
            found = root.find(forbidden_type)
            if found:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Dangerous SQL construct '{forbidden_type.__name__}' is strictly forbidden.",
                )

        # 5. Check for disallowed administrative or file system functions
        all_funcs = list(root.find_all(exp.Func)) + list(root.find_all(exp.Anonymous))
        for func in all_funcs:
            func_name = (getattr(func, "name", None) or str(func.this)).lower()
            if func_name in DISALLOWED_FUNCTIONS:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Disallowed administrative function '{func_name}' is not permitted.",
                )

        # 6. Extract CTE definitions so their aliases are not mistaken for external tables
        cte_aliases = set()
        for cte in root.find_all(exp.CTE):
            alias = cte.alias_or_name
            if alias:
                cte_aliases.add(alias.lower())

        # 7. Extract real database table references
        extracted_tables = []
        for table_node in root.find_all(exp.Table):
            t_name = table_node.name
            if not t_name:
                continue
            # If the table reference is a CTE alias, skip allowlist check
            if t_name.lower() in cte_aliases:
                continue

            schema_part = table_node.db
            full_name = f"{schema_part}.{t_name}" if schema_part else t_name
            extracted_tables.append(full_name)

            # Block unauthorized system catalog schemas
            if schema_part:
                schema_lower = schema_part.lower()
                if schema_lower in SYSTEM_SCHEMAS:
                    # Only allow if explicitly allowed in allowed_tables
                    allowed_full = (
                        {t.lower() for t in allowed_tables} if allowed_tables else set()
                    )
                    if full_name.lower() not in allowed_full:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"Access to protected system schema '{schema_part}' is strictly forbidden.",
                            extracted_tables=extracted_tables,
                        )

        # 8. Check against allowed tables if an allowlist is provided
        if allowed_tables is not None:
            allowed_with_schema = {
                t.strip().lower() for t in allowed_tables if "." in t
            }
            allowed_bare = {t.strip().lower().split(".")[-1] for t in allowed_tables}

            for ext_table in extracted_tables:
                ext_clean = ext_table.lower()
                bare_table = ext_clean.split(".")[-1]

                if "." in ext_clean:
                    schema_name = ext_clean.split(".")[0]
                    # If schema is explicitly given, must match an allowed schema.table or default user schema
                    if ext_clean in allowed_with_schema:
                        continue
                    if schema_name in ("public", "main") and bare_table in allowed_bare:
                        continue
                    return ValidationResult(
                        is_valid=False,
                        error_message=(
                            f"Unauthorized schema-qualified table reference '{ext_table}'. "
                            f"Allowed tables: {sorted(set(allowed_tables))}"
                        ),
                        extracted_tables=extracted_tables,
                    )
                else:
                    # Bare table name provided: must be in allowed bare table names
                    if bare_table not in allowed_bare:
                        return ValidationResult(
                            is_valid=False,
                            error_message=(
                                f"Unauthorized table reference '{ext_table}'. "
                                f"Allowed tables: {sorted(set(allowed_tables))}"
                            ),
                            extracted_tables=extracted_tables,
                        )

        return ValidationResult(
            is_valid=True,
            extracted_tables=extracted_tables,
            normalized_sql=root.sql(dialect=read_dialect),
            ast_root=root,
        )


def execute_safe_query(
    engine: Engine,
    query: str,
    allowed_tables: Sequence[str] | set[str] | None = None,
    dialect: str = "postgres",
    timeout_ms: int = 5000,
    max_rows: int = 1000,
) -> tuple[pd.DataFrame, bool]:
    """
    Validates and executes an SQL query safely with timeouts, db-level read-only mode, and row limits.

    Returns:
        tuple[pd.DataFrame, bool]: (DataFrame of results, is_truncated)
    """
    validation = SQLGuard.validate(
        query, dialect=dialect, allowed_tables=allowed_tables
    )
    if not validation.is_valid:
        raise SQLSecurityViolation(validation.error_message)

    read_dialect = SQLGuard.normalize_dialect(dialect)
    root = validation.ast_root

    # Bounded resource clamping
    clamped_max_rows = min(max(1, int(max_rows)), 5000)
    clamped_timeout_ms = min(max(100, int(timeout_ms)), 30000)

    # Server-Side Bounding: Ensure query AST has LIMIT capped at clamped_max_rows + 1
    fetch_limit = clamped_max_rows + 1
    if root is not None and isinstance(root, exp.Select):
        existing_limit = root.find(exp.Limit)
        if existing_limit is None:
            root.limit(fetch_limit, copy=False)
        else:
            try:
                lit_val = int(existing_limit.expression.this)
                if lit_val > fetch_limit:
                    existing_limit.expression.set("this", str(fetch_limit))
            except (AttributeError, ValueError):
                pass
        exec_sql = root.sql(dialect=read_dialect)
    else:
        exec_sql = query.strip().rstrip(";")

    with acquire_query_budget():
        try:
            with engine.connect() as conn:
                # 1. PostgreSQL: Enforce Transaction Read-Only, Scoped Timeout, & Temp Disk/Memory Limits
                if read_dialect == "postgres":
                    with conn.begin():
                        try:
                            conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                        except (SQLAlchemyError, OperationalError):
                            pass
                        conn.exec_driver_sql(
                            f"SET LOCAL statement_timeout = '{int(clamped_timeout_ms)}'"
                        )
                        try:
                            conn.exec_driver_sql("SET LOCAL work_mem = '16MB'")
                        except (SQLAlchemyError, OperationalError):
                            pass
                        result = conn.execute(text(exec_sql))
                        rows = result.fetchmany(clamped_max_rows + 1)
                        cols = list(result.keys())

                # 2. MySQL: Enforce Read-Only Mode & Scoped max_execution_time with Reset
                elif read_dialect == "mysql":
                    try:
                        conn.exec_driver_sql("SET SESSION TRANSACTION READ ONLY")
                    except (SQLAlchemyError, OperationalError):
                        pass
                    conn.exec_driver_sql(
                        f"SET SESSION max_execution_time = {int(clamped_timeout_ms)}"
                    )
                    try:
                        result = conn.execute(text(exec_sql))
                        rows = result.fetchmany(clamped_max_rows + 1)
                        cols = list(result.keys())
                    finally:
                        try:
                            conn.exec_driver_sql("SET SESSION max_execution_time = 0")
                        except (SQLAlchemyError, OperationalError):
                            pass

                # 3. SQLite: Enforce query_only PRAGMA & Progress Handler Timeout
                else:
                    try:
                        conn.exec_driver_sql("PRAGMA query_only = ON")
                    except (SQLAlchemyError, OperationalError):
                        pass

                    raw_conn = getattr(
                        conn.connection, "dbapi_connection", conn.connection
                    )
                    start_time = time.time()
                    timeout_sec = clamped_timeout_ms / 1000.0

                    def _progress_check() -> int:
                        if time.time() - start_time > timeout_sec:
                            return 1  # Non-zero interrupts the SQLite VM
                        return 0

                    if hasattr(raw_conn, "set_progress_handler"):
                        raw_conn.set_progress_handler(_progress_check, 1000)

                    try:
                        result = conn.execute(text(exec_sql))
                        rows = result.fetchmany(clamped_max_rows + 1)
                        cols = list(result.keys())
                    finally:
                        if hasattr(raw_conn, "set_progress_handler"):
                            try:
                                raw_conn.set_progress_handler(None, 0)
                            except (AttributeError, RuntimeError):
                                pass

                is_truncated = len(rows) > clamped_max_rows
                if is_truncated:
                    rows = rows[:clamped_max_rows]

                df = pd.DataFrame(rows, columns=cols)
                return df, is_truncated

        except OperationalError as oe:
            err_str = str(oe).lower()
            if (
                "timeout" in err_str
                or "canceling statement" in err_str
                or "max_execution_time" in err_str
                or "interrupted" in err_str
            ):
                raise QueryTimeoutError(
                    f"Query exceeded the {clamped_timeout_ms}ms timeout limit and was terminated."
                ) from oe
            if "readonly database" in err_str or "read-only" in err_str:
                raise SQLSecurityViolation(
                    f"Database-level read-only protection blocked execution: {oe}"
                ) from oe
            if "temp_file_limit" in err_str or "temporary file size" in err_str:
                raise QueryExecutionError(
                    "Query aborted: temporary disk usage exceeded limit (64MB)."
                ) from oe
            raise QueryExecutionError(f"Database error: {oe}") from oe
        except SQLAlchemyError as se:
            raise QueryExecutionError(f"SQL execution failed: {se}") from se
