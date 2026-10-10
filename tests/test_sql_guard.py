"""Comprehensive adversarial and functional tests for SQLGuard."""

import pytest

from schemasense.sql_guard import (
    QueryTimeoutError,
    SQLGuard,
    SQLSecurityViolation,
    execute_safe_query,
)

# ─────────────────────────────────────────────────────────────
# 1. ADVERSARIAL SQL TESTS — MUST BE REJECTED
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "query,reason",
    [
        ("DROP TABLE customers;", "Destructive DDL"),
        ("TRUNCATE TABLE customers;", "Destructive DDL"),
        ("ALTER TABLE customers ADD COLUMN malicious TEXT;", "Schema modification"),
        ("CREATE TABLE backdoor (id INT);", "Table creation"),
        ("INSERT INTO customers (id, name) VALUES (999, 'Hacker');", "DML write"),
        ("UPDATE customers SET name = 'Pwned' WHERE id = 1;", "DML write"),
        ("DELETE FROM customers WHERE id = 1;", "DML write"),
        ("GRANT ALL PRIVILEGES ON DATABASE mydb TO hacker;", "Privilege escalation"),
        (
            "REVOKE ALL PRIVILEGES ON DATABASE mydb FROM admin;",
            "Privilege modification",
        ),
        ("SELECT 1; DROP TABLE customers;", "Multi-statement injection"),
        ("SELECT 1; SELECT 2;", "Multiple statements"),
        ("SELECT * FROM customers; --\nDROP TABLE orders;", "Comment multi-statement"),
        (
            "WITH del AS (DELETE FROM orders RETURNING *) SELECT * FROM del;",
            "CTE with nested DELETE",
        ),
        (
            "WITH ins AS (INSERT INTO customers VALUES (1, 'a', 'b', 'c') RETURNING *) SELECT * FROM ins;",
            "CTE with nested INSERT",
        ),
        ("SELECT pg_read_file('/etc/passwd');", "Disallowed function pg_read_file"),
        (
            "SELECT pg_terminate_backend(1234);",
            "Disallowed function pg_terminate_backend",
        ),
        ("SELECT pg_sleep(10);", "Disallowed function pg_sleep"),
        ("SELECT load_file('/etc/passwd');", "Disallowed function load_file"),
        ("SELECT load_extension('evil.so');", "Disallowed function load_extension"),
        ("SELECT pg_ls_dir('/tmp');", "Disallowed function pg_ls_dir"),
        ("SELECT * FROM pg_catalog.customers;", "Protected system schema pg_catalog"),
        (
            "SELECT * FROM information_schema.tables;",
            "Protected system schema information_schema",
        ),
        ("SELECT * FROM secret_schema.customers;", "Unauthorized schema qualifier"),
        ("SELECT * FROM secret_passwords;", "Disallowed table reference"),
        (
            "SELECT c.name FROM (SELECT * FROM secret_passwords) s JOIN customers c ON 1=1;",
            "Disallowed table inside subquery",
        ),
        ("", "Empty query"),
        ("   ", "Whitespace query"),
        ("SELECT * FROM", "SQL Syntax error"),
    ],
)
def test_adversarial_queries_rejected(query, reason):
    allowed_tables = {"customers", "orders"}
    result = SQLGuard.validate(query, dialect="postgres", allowed_tables=allowed_tables)
    assert not result.is_valid, f"Expected query to be rejected ({reason}): {query}"
    assert result.error_message is not None


# ─────────────────────────────────────────────────────────────
# 2. LEGITIMATE QUERIES — MUST BE PERMITTED
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM customers;",
        "SELECT id, name, email FROM customers WHERE city = 'Delhi';",
        "SELECT c.name, o.amount FROM customers c JOIN orders o ON c.id = o.customer_id;",
        "SELECT city, count(*) as cnt FROM customers GROUP BY city HAVING cnt > 1;",
        "SELECT * FROM customers ORDER BY id DESC LIMIT 10;",
        "SELECT * FROM customers WHERE id IN (SELECT customer_id FROM orders WHERE amount > 100);",
        "WITH high_orders AS (SELECT * FROM orders WHERE amount > 500) SELECT * FROM high_orders;",
        'SELECT * FROM "customers" WHERE "id" = 1;',
        "SELECT * FROM customers -- inline comment\nWHERE id = 2;",
        "SELECT * FROM customers /* block comment */ WHERE id = 3;",
        "SELECT c.name, (SELECT max(amount) FROM orders WHERE customer_id = c.id) as max_amt FROM customers c;",
        "SELECT 1 as num UNION SELECT 2 as num;",
    ],
)
def test_legitimate_queries_accepted(query):
    allowed_tables = {"customers", "orders"}
    result = SQLGuard.validate(query, dialect="postgres", allowed_tables=allowed_tables)
    assert result.is_valid, (
        f"Expected query to be valid: {query}, got error: {result.error_message}"
    )


# ─────────────────────────────────────────────────────────────
# 3. TABLE ALLOWLIST AND CTE ALIAS RESOLUTION
# ─────────────────────────────────────────────────────────────


def test_cte_aliases_not_mistaken_for_unauthorized_tables():
    query = """
    WITH regional_summary AS (
        SELECT city, count(*) as user_count FROM customers GROUP BY city
    )
    SELECT * FROM regional_summary WHERE user_count > 0;
    """
    allowed_tables = {"customers"}
    result = SQLGuard.validate(query, dialect="postgres", allowed_tables=allowed_tables)
    assert result.is_valid, f"Failed on CTE alias resolution: {result.error_message}"
    assert "customers" in [t.lower() for t in result.extracted_tables]


def test_schema_qualified_table_names():
    query = "SELECT * FROM public.customers;"
    allowed_tables = {"customers"}
    result = SQLGuard.validate(query, dialect="postgres", allowed_tables=allowed_tables)
    assert result.is_valid, (
        f"Expected schema qualification to match allowlist: {result.error_message}"
    )


def test_schema_spoofing_prevented():
    allowed_tables = {"customers"}
    # Even though bare table is 'customers', system and foreign schemas must be rejected
    res_sys = SQLGuard.validate(
        "SELECT * FROM pg_catalog.customers", allowed_tables=allowed_tables
    )
    assert not res_sys.is_valid
    assert "system schema" in res_sys.error_message.lower()

    res_foreign = SQLGuard.validate(
        "SELECT * FROM hr_schema.customers", allowed_tables=allowed_tables
    )
    assert not res_foreign.is_valid
    assert "unauthorized schema-qualified" in res_foreign.error_message.lower()


# ─────────────────────────────────────────────────────────────
# 4. BOUNDED EXECUTION, TIMEOUTS & TRUNCATION TESTS
# ─────────────────────────────────────────────────────────────


def test_execute_safe_query_success(sqlite_engine):
    allowed_tables = {"customers", "orders"}
    df, is_truncated = execute_safe_query(
        sqlite_engine,
        "SELECT id, name FROM customers ORDER BY id ASC LIMIT 5",
        allowed_tables=allowed_tables,
        dialect="sqlite",
        max_rows=10,
    )
    assert len(df) == 5
    assert not is_truncated
    assert list(df.columns) == ["id", "name"]


def test_execute_safe_query_truncation_detection(sqlite_engine):
    allowed_tables = {"customers"}
    # The fixture seeds 25 customers; we set max_rows=10
    df, is_truncated = execute_safe_query(
        sqlite_engine,
        "SELECT * FROM customers",
        allowed_tables=allowed_tables,
        dialect="sqlite",
        max_rows=10,
    )
    assert len(df) == 10
    assert is_truncated is True


def test_execute_safe_query_blocks_unauthorized_table(sqlite_engine):
    allowed_tables = {"customers"}
    with pytest.raises(SQLSecurityViolation) as exc_info:
        execute_safe_query(
            sqlite_engine,
            "SELECT * FROM secret_passwords",
            allowed_tables=allowed_tables,
            dialect="sqlite",
        )
    assert "Unauthorized table" in str(exc_info.value)


def test_execute_safe_query_timeout(sqlite_engine):
    allowed_tables = {"customers"}
    # Infinite recursive CTE must be interrupted by statement timeout
    with pytest.raises(QueryTimeoutError):
        execute_safe_query(
            sqlite_engine,
            "WITH RECURSIVE bomb(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM bomb) SELECT count(*) FROM bomb",
            allowed_tables=allowed_tables,
            dialect="sqlite",
            timeout_ms=250,
        )


def test_sql_guard_comment_stacked_attack():
    # Attacker tries to hide second statement after comment
    res = SQLGuard.validate(
        "SELECT * FROM customers /* safe */ ; DROP TABLE customers;",
        allowed_tables={"customers"},
    )
    assert res.is_valid is False
    assert "Multi-statement execution is strictly prohibited" in str(res.error_message)


def test_execute_safe_query_exact_row_boundary(sqlite_engine):
    """Verifies that returning exactly max_rows is NOT flagged as truncated."""
    allowed_tables = {"customers"}
    df, is_truncated = execute_safe_query(
        sqlite_engine,
        "SELECT * FROM customers LIMIT 5",
        allowed_tables=allowed_tables,
        dialect="sqlite",
        max_rows=5,
    )
    assert len(df) == 5
    assert is_truncated is False


def test_sql_guard_disallows_system_catalog_in_subquery():
    res = SQLGuard.validate(
        "SELECT * FROM customers WHERE id IN (SELECT count(*) FROM sqlite_master);",
        dialect="sqlite",
        allowed_tables={"customers"},
    )
    assert res.is_valid is False
    assert (
        "sqlite_master" in str(res.error_message).lower()
        or "unauthorized" in str(res.error_message).lower()
    )
