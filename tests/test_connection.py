"""Unit tests for connection string safety, engine disposal, privilege inspection, and concurrency bounding."""

import threading
import time

import pytest
from sqlalchemy import create_engine

from schemasense.connection import (
    MAX_CONCURRENT_QUERIES,
    ConcurrencyLimitExceeded,
    ConnectionConfigError,
    DatabaseConnectionError,
    acquire_query_budget,
    build_safe_url,
    create_safe_engine,
    dispose_engine,
    inspect_account_privileges,
)


def test_build_safe_url_postgresql():
    url = build_safe_url(
        db_type="PostgreSQL",
        host="db.enterprise.com",
        port="5432",
        db_name="olist",
        user="admin",
        password="secret#password@123",
    )
    assert url.drivername == "postgresql+psycopg2"
    assert url.host == "db.enterprise.com"
    assert url.port == 5432
    assert url.database == "olist"
    assert url.username == "admin"
    assert url.password == "secret#password@123"


def test_build_safe_url_mysql():
    url = build_safe_url(
        db_type="MySQL",
        host="127.0.0.1",
        port=3306,
        db_name="ecommerce",
        user="root",
        password="root_password",
    )
    assert url.drivername == "mysql+pymysql"
    assert url.query.get("charset") == "utf8mb4"


def test_build_safe_url_invalid_port():
    with pytest.raises(ConnectionConfigError):
        build_safe_url("PostgreSQL", "localhost", "not-a-port", "db", "user", "pass")


def test_build_safe_url_missing_fields():
    with pytest.raises(ConnectionConfigError):
        build_safe_url("PostgreSQL", "", 5432, "db", "user", "pass")


def test_build_safe_url_unsupported_engine():
    with pytest.raises(ConnectionConfigError):
        build_safe_url("OracleDB", "localhost", 1521, "db", "user", "pass")


def test_dispose_engine_handles_none():
    dispose_engine(None)  # Should not raise any exception


def test_dispose_engine_disposes_active_engine():
    engine = create_engine("sqlite:///:memory:")
    dispose_engine(engine)  # Should cleanly dispose


def test_sqlite_privilege_inspection():
    engine = create_engine("sqlite:///:memory:")
    report = inspect_account_privileges(engine, db_type="SQLite")
    assert report.is_privileged is False
    assert report.db_type == "SQLite"
    assert "least-privilege" in report.summary
    dispose_engine(engine)


def test_concurrency_budget_enforcement():
    """Verifies that acquire_query_budget strictly limits concurrent query slots."""
    slots_acquired = []

    def _worker():
        try:
            with acquire_query_budget(timeout_sec=0.2):
                slots_acquired.append(True)
                time.sleep(0.3)
        except ConcurrencyLimitExceeded:
            slots_acquired.append(False)

    # Spawn MAX_CONCURRENT_QUERIES + 2 threads
    threads = [
        threading.Thread(target=_worker) for _ in range(MAX_CONCURRENT_QUERIES + 2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly MAX_CONCURRENT_QUERIES should succeed, excess should fail
    successful = [s for s in slots_acquired if s is True]
    failed = [s for s in slots_acquired if s is False]
    assert len(successful) == MAX_CONCURRENT_QUERIES
    assert len(failed) == 2


def test_concurrency_budget_exception_safety():
    """Verifies that an unhandled exception inside acquire_query_budget releases the slot."""
    with pytest.raises(ZeroDivisionError), acquire_query_budget(timeout_sec=0.5):
        _ = 1 / 0

    # Next attempt must succeed immediately without deadlock
    acquired = False
    with acquire_query_budget(timeout_sec=0.5):
        acquired = True
    assert acquired is True


def test_concurrency_recovery_after_timeout():
    """Verifies that after in-flight queries release slots, new queries succeed immediately."""
    with acquire_query_budget(timeout_sec=0.5):
        pass  # released immediately

    acquired = False
    with acquire_query_budget(timeout_sec=0.5):
        acquired = True
    assert acquired is True


def test_create_safe_engine_password_masking():
    """Verifies that passwords are never leaked in cleartext within exception strings."""
    secret = "TopSecretPassword123!"
    with pytest.raises(DatabaseConnectionError) as exc_info:
        create_safe_engine(
            db_type="PostgreSQL",
            host="127.0.0.1",
            port=65432,  # Non-existent port
            db_name="fake_db",
            user="some_user",
            password=secret,
            connect_timeout=1,
        )
    err_str = str(exc_info.value)
    assert secret not in err_str, (
        "Password must NOT be leaked in cleartext error message"
    )
