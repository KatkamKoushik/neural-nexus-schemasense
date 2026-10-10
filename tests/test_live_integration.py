"""Live integration and adversarial verification tests against disposable PostgreSQL and MySQL instances.

Covers:
1. Live schema inspection across engines.
2. Read-only query execution, JOINs, aggregations.
3. Server-side row bounding and client-side truncation detection.
4. Dialect-specific statement timeouts, cancellation, and connection pool reuse.
5. Independent database-role security (privilege boundaries at the DB engine).
6. Adversarial injection: writes, stacked queries, disallowed functions, system catalogs.
7. Dialect syntax: recursive CTEs, quoted identifiers, and malformed queries.
"""

import socket

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError

from schemasense.connection import (
    PrivilegedAccountError,
    build_safe_url,
    create_safe_engine,
    dispose_engine,
    inspect_account_privileges,
)
from schemasense.schema_inspector import inspect_database_schema
from schemasense.sql_guard import (
    QueryExecutionError,
    QueryTimeoutError,
    SQLSecurityViolation,
    execute_safe_query,
)


def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Checks if a network port is reachable."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


# ─────────────────────────────────────────────────────────────
# POSTGRESQL FIXTURES & SETUP
# ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def postgres_admin_engine():
    if not is_port_open("127.0.0.1", 5433):
        pytest.skip("Live PostgreSQL test container (port 5433) is not available.")

    url = build_safe_url(
        db_type="PostgreSQL",
        host="127.0.0.1",
        port=5433,
        db_name="schemasense_test",
        user="test_admin",
        password="test_admin_pass",
    )
    engine = create_engine(url, pool_pre_ping=True)

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS orders CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS customers CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS secret_passwords CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS analytics_log CASCADE;"))
        conn.execute(text("DROP SCHEMA IF EXISTS hr_schema CASCADE;"))

        # Create tables
        conn.execute(
            text("""
            CREATE TABLE customers (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) UNIQUE,
                city VARCHAR(50)
            );
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE orders (
                order_id SERIAL PRIMARY KEY,
                customer_id INT REFERENCES customers(id),
                amount NUMERIC(10,2),
                order_date DATE
            );
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE secret_passwords (
                id SERIAL PRIMARY KEY,
                hash VARCHAR(255) NOT NULL
            );
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE analytics_log (
                event_id SERIAL PRIMARY KEY,
                event_type VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        )

        # Create secondary schema
        conn.execute(text("CREATE SCHEMA hr_schema;"))
        conn.execute(
            text("""
            CREATE TABLE hr_schema.employees (
                emp_id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                salary NUMERIC(10,2)
            );
        """)
        )

        # Seed data
        for i in range(1, 26):
            conn.execute(
                text(
                    "INSERT INTO customers (name, email, city) VALUES (:name, :email, :city)"
                ),
                {
                    "name": f"Customer {i}",
                    "email": f"cust{i}@test.com",
                    "city": "Delhi" if i % 2 == 0 else "Bangalore",
                },
            )
            conn.execute(
                text(
                    "INSERT INTO orders (customer_id, amount, order_date) VALUES (:cid, :amt, '2026-01-01')"
                ),
                {"cid": i, "amt": i * 75.0},
            )

        # Seed 2,500 analytics rows to test bounded execution & row caps
        conn.execute(
            text("""
            INSERT INTO analytics_log (event_type)
            SELECT 'click' FROM generate_series(1, 2500);
        """)
        )

        # Seed hr_schema
        conn.execute(
            text("""
            INSERT INTO hr_schema.employees (name, salary)
            VALUES ('Alice HR', 85000), ('Bob HR', 92000);
        """)
        )

        # Create restricted read-only database user
        conn.execute(
            text("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'schemasense_readonly') THEN
                    CREATE ROLE schemasense_readonly WITH LOGIN PASSWORD 'readonly_pass';
                END IF;
            END
            $$;
        """)
        )
        conn.execute(
            text("GRANT CONNECT ON DATABASE schemasense_test TO schemasense_readonly;")
        )
        conn.execute(text("GRANT USAGE ON SCHEMA public TO schemasense_readonly;"))
        conn.execute(text("GRANT USAGE ON SCHEMA hr_schema TO schemasense_readonly;"))
        conn.execute(
            text("GRANT SELECT ON ALL TABLES IN SCHEMA public TO schemasense_readonly;")
        )
        conn.execute(
            text(
                "GRANT SELECT ON ALL TABLES IN SCHEMA hr_schema TO schemasense_readonly;"
            )
        )
        conn.execute(
            text("REVOKE SELECT ON TABLE secret_passwords FROM schemasense_readonly;")
        )

    yield engine
    dispose_engine(engine)


@pytest.fixture(scope="module")
def postgres_readonly_engine(postgres_admin_engine):
    engine = create_safe_engine(
        db_type="PostgreSQL",
        host="127.0.0.1",
        port=5433,
        db_name="schemasense_test",
        user="schemasense_readonly",
        password="readonly_pass",
    )
    yield engine
    dispose_engine(engine)


# ─────────────────────────────────────────────────────────────
# MYSQL FIXTURES & SETUP
# ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mysql_admin_engine():
    if not is_port_open("127.0.0.1", 3307):
        pytest.skip("Live MySQL test container (port 3307) is not available.")

    url = build_safe_url(
        db_type="MySQL",
        host="127.0.0.1",
        port=3307,
        db_name="schemasense_test",
        user="test_admin",
        password="test_admin_pass",
    )
    engine = create_engine(url, pool_pre_ping=True)

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS orders;"))
        conn.execute(text("DROP TABLE IF EXISTS customers;"))
        conn.execute(text("DROP TABLE IF EXISTS secret_passwords;"))
        conn.execute(text("DROP TABLE IF EXISTS analytics_log;"))

        conn.execute(
            text("""
            CREATE TABLE customers (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) UNIQUE,
                city VARCHAR(50)
            );
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE orders (
                order_id INT AUTO_INCREMENT PRIMARY KEY,
                customer_id INT,
                amount DECIMAL(10,2),
                order_date DATE,
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            );
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE secret_passwords (
                id INT AUTO_INCREMENT PRIMARY KEY,
                hash VARCHAR(255) NOT NULL
            );
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE analytics_log (
                event_id INT AUTO_INCREMENT PRIMARY KEY,
                event_type VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        )

        # Seed data
        for i in range(1, 26):
            conn.execute(
                text(
                    "INSERT INTO customers (name, email, city) VALUES (:name, :email, :city)"
                ),
                {
                    "name": f"Customer {i}",
                    "email": f"cust{i}@test.com",
                    "city": "Delhi" if i % 2 == 0 else "Bangalore",
                },
            )
            conn.execute(
                text(
                    "INSERT INTO orders (customer_id, amount, order_date) VALUES (:cid, :amt, '2026-01-01')"
                ),
                {"cid": i, "amt": i * 75.0},
            )

        # Seed rows for analytics_log
        for _ in range(50):
            conn.execute(
                text("""
                INSERT INTO analytics_log (event_type) VALUES 
                ('click'), ('view'), ('cart'), ('checkout'), ('login'),
                ('click'), ('view'), ('cart'), ('checkout'), ('login'),
                ('click'), ('view'), ('cart'), ('checkout'), ('login'),
                ('click'), ('view'), ('cart'), ('checkout'), ('login'),
                ('click'), ('view'), ('cart'), ('checkout'), ('login')
            """)
            )

        conn.commit()

    # Configure restricted read-only user on MySQL root
    root_url = build_safe_url(
        db_type="MySQL",
        host="127.0.0.1",
        port=3307,
        db_name="schemasense_test",
        user="root",
        password="root_test_pass",
    )
    root_engine = create_engine(root_url, pool_pre_ping=True)
    with root_engine.connect() as conn:
        conn.execute(
            text(
                "CREATE USER IF NOT EXISTS 'schemasense_readonly'@'%' IDENTIFIED BY 'readonly_pass';"
            )
        )
        conn.execute(
            text("REVOKE ALL PRIVILEGES, GRANT OPTION FROM 'schemasense_readonly'@'%';")
        )
        conn.execute(
            text(
                "GRANT SELECT ON schemasense_test.customers TO 'schemasense_readonly'@'%';"
            )
        )
        conn.execute(
            text(
                "GRANT SELECT ON schemasense_test.orders TO 'schemasense_readonly'@'%';"
            )
        )
        conn.execute(
            text(
                "GRANT SELECT ON schemasense_test.analytics_log TO 'schemasense_readonly'@'%';"
            )
        )
        conn.execute(text("FLUSH PRIVILEGES;"))
        conn.commit()
    dispose_engine(root_engine)

    yield engine
    dispose_engine(engine)


@pytest.fixture(scope="module")
def mysql_readonly_engine(mysql_admin_engine):
    engine = create_safe_engine(
        db_type="MySQL",
        host="127.0.0.1",
        port=3307,
        db_name="schemasense_test",
        user="schemasense_readonly",
        password="readonly_pass",
    )
    yield engine
    dispose_engine(engine)


# ─────────────────────────────────────────────────────────────
# POSTGRESQL LIVE INTEGRATION & ADVERSARIAL TESTS
# ─────────────────────────────────────────────────────────────


def test_live_postgres_schema_inspection(postgres_admin_engine):
    schema = inspect_database_schema(postgres_admin_engine)
    assert "customers" in schema.tables
    assert "orders" in schema.tables
    assert "analytics_log" in schema.tables
    cust = schema.tables["customers"]
    assert "id" in cust.primary_key
    assert any(c.name == "email" for c in cust.columns)
    # Verify live constraint extraction
    orders_tbl = schema.tables["orders"]
    assert any(fk.referred_table == "customers" for fk in orders_tbl.foreign_keys)


def test_live_postgres_safe_query_execution(postgres_readonly_engine):
    allowed_tables = {"customers", "orders"}
    df, is_truncated = execute_safe_query(
        engine=postgres_readonly_engine,
        query="SELECT c.name, o.amount FROM customers c JOIN orders o ON c.id = o.customer_id ORDER BY o.amount DESC LIMIT 5",
        allowed_tables=allowed_tables,
        dialect="postgres",
        max_rows=10,
    )
    assert len(df) == 5
    assert not is_truncated
    assert "name" in df.columns
    assert "amount" in df.columns


def test_live_postgres_row_cap_truncation(postgres_readonly_engine):
    allowed_tables = {"analytics_log"}
    df, is_truncated = execute_safe_query(
        engine=postgres_readonly_engine,
        query="SELECT * FROM analytics_log",
        allowed_tables=allowed_tables,
        dialect="postgres",
        max_rows=500,
    )
    assert len(df) == 500
    assert is_truncated is True


def test_live_postgres_statement_timeout(postgres_readonly_engine):
    allowed_tables = {"customers"}
    with pytest.raises(QueryTimeoutError):
        execute_safe_query(
            engine=postgres_readonly_engine,
            query="SELECT count(*) FROM customers a, customers b, customers c, customers d, customers e, customers f",
            allowed_tables=allowed_tables,
            dialect="postgres",
            timeout_ms=200,
        )


def test_live_postgres_connection_pool_reuse_after_timeout(postgres_readonly_engine):
    """Verifies that connection pool remains completely healthy and resets timeout after an aborted query."""
    allowed_tables = {"customers"}
    # Force a timeout
    try:
        execute_safe_query(
            engine=postgres_readonly_engine,
            query="SELECT count(*) FROM customers a, customers b, customers c, customers d, customers e, customers f",
            allowed_tables=allowed_tables,
            dialect="postgres",
            timeout_ms=150,
        )
    except QueryTimeoutError:
        pass

    # Next query on same engine must succeed cleanly
    df, is_truncated = execute_safe_query(
        engine=postgres_readonly_engine,
        query="SELECT name FROM customers ORDER BY id LIMIT 3",
        allowed_tables=allowed_tables,
        dialect="postgres",
        timeout_ms=5000,
    )
    assert len(df) == 3
    assert not is_truncated


def test_live_postgres_blocked_write_operations(postgres_readonly_engine):
    allowed_tables = {"customers"}
    for write_sql in [
        "INSERT INTO customers (name, email) VALUES ('Hacker', 'h@h.com')",
        "UPDATE customers SET name = 'Owned' WHERE id = 1",
        "DELETE FROM customers WHERE id = 1",
        "DROP TABLE customers",
    ]:
        with pytest.raises(SQLSecurityViolation):
            execute_safe_query(
                engine=postgres_readonly_engine,
                query=write_sql,
                allowed_tables=allowed_tables,
                dialect="postgres",
            )


def test_live_postgres_stacked_statements_blocked(postgres_readonly_engine):
    allowed_tables = {"customers"}
    with pytest.raises(SQLSecurityViolation):
        execute_safe_query(
            engine=postgres_readonly_engine,
            query="SELECT 1; DROP TABLE customers;",
            allowed_tables=allowed_tables,
            dialect="postgres",
        )


def test_live_postgres_disallowed_functions_blocked(postgres_readonly_engine):
    allowed_tables = {"customers"}
    for dangerous_sql in [
        "SELECT pg_sleep(5)",
        "SELECT pg_read_file('pg_hba.conf')",
        "SELECT pg_ls_dir('.')",
    ]:
        with pytest.raises(SQLSecurityViolation):
            execute_safe_query(
                engine=postgres_readonly_engine,
                query=dangerous_sql,
                allowed_tables=allowed_tables,
                dialect="postgres",
            )


def test_live_postgres_system_catalogs_blocked(postgres_readonly_engine):
    allowed_tables = {"customers"}
    with pytest.raises(SQLSecurityViolation):
        execute_safe_query(
            engine=postgres_readonly_engine,
            query="SELECT * FROM pg_catalog.pg_tables",
            allowed_tables=allowed_tables,
            dialect="postgres",
        )


def test_live_postgres_recursive_cte(postgres_readonly_engine):
    df, is_truncated = execute_safe_query(
        engine=postgres_readonly_engine,
        query="WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 5) SELECT x FROM cnt",
        allowed_tables=None,
        dialect="postgres",
    )
    assert len(df) == 5
    assert not is_truncated


def test_live_postgres_quoted_identifiers(postgres_readonly_engine):
    allowed_tables = {"customers"}
    df, _ = execute_safe_query(
        engine=postgres_readonly_engine,
        query='SELECT "name", "city" FROM "customers" WHERE "id" = 1',
        allowed_tables=allowed_tables,
        dialect="postgres",
    )
    assert len(df) == 1
    assert "name" in df.columns


def test_live_postgres_db_role_blocks_write_independently(postgres_readonly_engine):
    """Verifies that the database role itself blocks writes even if application validation were bypassed."""
    with (
        pytest.raises(DBAPIError) as exc_info,
        postgres_readonly_engine.connect() as conn,
    ):
        conn.execute(
            text("INSERT INTO customers (name, email) VALUES ('Bypassed', 'b@b.com')")
        )
    assert "permission denied" in str(exc_info.value).lower()


def test_live_postgres_db_role_blocks_unauthorized_table(postgres_readonly_engine):
    """Verifies that the database user cannot read secret_passwords at the DB permissions level."""
    with (
        pytest.raises(DBAPIError) as exc_info,
        postgres_readonly_engine.connect() as conn,
    ):
        conn.execute(text("SELECT * FROM secret_passwords"))
    assert "permission denied" in str(exc_info.value).lower()


def test_live_postgres_schema_qualified_table(postgres_readonly_engine):
    """Verifies schema-qualified table behavior in PostgreSQL."""
    # When authorized
    df, is_truncated = execute_safe_query(
        engine=postgres_readonly_engine,
        query="SELECT name, salary FROM hr_schema.employees ORDER BY salary DESC",
        allowed_tables={"hr_schema.employees"},
        dialect="postgres",
    )
    assert len(df) == 2
    assert not is_truncated
    assert "salary" in df.columns

    # When not in allowed_tables
    with pytest.raises(SQLSecurityViolation):
        execute_safe_query(
            engine=postgres_readonly_engine,
            query="SELECT name, salary FROM hr_schema.employees",
            allowed_tables={"customers"},
            dialect="postgres",
        )


def test_live_postgres_dialect_ilike(postgres_readonly_engine):
    """Verifies PostgreSQL-specific ILIKE case-insensitive pattern matching."""
    df, _ = execute_safe_query(
        engine=postgres_readonly_engine,
        query="SELECT name FROM customers WHERE name ILIKE 'customer%' ORDER BY id LIMIT 3",
        allowed_tables={"customers"},
        dialect="postgres",
    )
    assert len(df) == 3


def test_live_postgres_malformed_sql(postgres_readonly_engine):
    """Verifies malformed SQL syntax is cleanly rejected before reaching execution."""
    with pytest.raises(SQLSecurityViolation):
        execute_safe_query(
            engine=postgres_readonly_engine,
            query="SELECT FROM WHERE;",
            allowed_tables={"customers"},
            dialect="postgres",
        )


def test_live_postgres_pool_reuse_after_execution_failure(postgres_readonly_engine):
    """Verifies pool health after query execution failure (e.g., non-existent column)."""
    with pytest.raises(QueryExecutionError):
        execute_safe_query(
            engine=postgres_readonly_engine,
            query="SELECT nonexistent_column FROM customers",
            allowed_tables={"customers"},
            dialect="postgres",
        )

    # Next query on same engine must succeed cleanly
    df, _ = execute_safe_query(
        engine=postgres_readonly_engine,
        query="SELECT name FROM customers LIMIT 2",
        allowed_tables={"customers"},
        dialect="postgres",
    )
    assert len(df) == 2


# ─────────────────────────────────────────────────────────────
# MYSQL LIVE INTEGRATION & ADVERSARIAL TESTS
# ─────────────────────────────────────────────────────────────


def test_live_mysql_schema_inspection(mysql_admin_engine):
    schema = inspect_database_schema(mysql_admin_engine)
    assert "customers" in schema.tables
    assert "orders" in schema.tables
    cust = schema.tables["customers"]
    assert "id" in cust.primary_key


def test_live_mysql_safe_query_execution(mysql_readonly_engine):
    allowed_tables = {"customers", "orders"}
    df, is_truncated = execute_safe_query(
        engine=mysql_readonly_engine,
        query="SELECT c.name, o.amount FROM customers c JOIN orders o ON c.id = o.customer_id ORDER BY o.amount DESC LIMIT 5",
        allowed_tables=allowed_tables,
        dialect="mysql",
        max_rows=10,
    )
    assert len(df) == 5
    assert not is_truncated
    assert "name" in df.columns


def test_live_mysql_row_cap_truncation(mysql_readonly_engine):
    allowed_tables = {"analytics_log"}
    df, is_truncated = execute_safe_query(
        engine=mysql_readonly_engine,
        query="SELECT * FROM analytics_log",
        allowed_tables=allowed_tables,
        dialect="mysql",
        max_rows=300,
    )
    assert len(df) == 300
    assert is_truncated is True


def test_live_mysql_statement_timeout(mysql_readonly_engine):
    allowed_tables = {"customers"}
    with pytest.raises(QueryTimeoutError):
        execute_safe_query(
            engine=mysql_readonly_engine,
            query="SELECT a.name, b.name FROM customers a JOIN customers b ON a.name != b.name JOIN customers c ON b.name != c.name JOIN customers d ON c.name != d.name JOIN customers e ON d.name != e.name ORDER BY md5(concat(a.name, b.name, c.name, d.name, e.name))",
            allowed_tables=allowed_tables,
            dialect="mysql",
            timeout_ms=200,
        )


def test_live_mysql_connection_pool_reuse_after_timeout(mysql_readonly_engine):
    """Verifies that MySQL session state resets cleanly and subsequent queries succeed."""
    allowed_tables = {"customers"}
    try:
        execute_safe_query(
            engine=mysql_readonly_engine,
            query="SELECT a.name, b.name FROM customers a JOIN customers b ON a.name != b.name JOIN customers c ON b.name != c.name JOIN customers d ON c.name != d.name JOIN customers e ON d.name != e.name ORDER BY md5(concat(a.name, b.name, c.name, d.name, e.name))",
            allowed_tables=allowed_tables,
            dialect="mysql",
            timeout_ms=150,
        )
    except QueryTimeoutError:
        pass

    # Ensure subsequent query works without timeout
    df, is_truncated = execute_safe_query(
        engine=mysql_readonly_engine,
        query="SELECT name FROM customers ORDER BY id LIMIT 3",
        allowed_tables=allowed_tables,
        dialect="mysql",
        timeout_ms=5000,
    )
    assert len(df) == 3
    assert not is_truncated


def test_live_mysql_blocked_write_operations(mysql_readonly_engine):
    allowed_tables = {"customers"}
    for write_sql in [
        "INSERT INTO customers (name, email) VALUES ('Hacker', 'h@h.com')",
        "UPDATE customers SET name = 'Owned' WHERE id = 1",
        "DELETE FROM customers WHERE id = 1",
        "DROP TABLE customers",
    ]:
        with pytest.raises(SQLSecurityViolation):
            execute_safe_query(
                engine=mysql_readonly_engine,
                query=write_sql,
                allowed_tables=allowed_tables,
                dialect="mysql",
            )


def test_live_mysql_stacked_statements_blocked(mysql_readonly_engine):
    allowed_tables = {"customers"}
    with pytest.raises(SQLSecurityViolation):
        execute_safe_query(
            engine=mysql_readonly_engine,
            query="SELECT 1; DROP TABLE customers;",
            allowed_tables=allowed_tables,
            dialect="mysql",
        )


def test_live_mysql_disallowed_functions_blocked(mysql_readonly_engine):
    allowed_tables = {"customers"}
    for dangerous_sql in [
        "SELECT sleep(5)",
        "SELECT load_file('/etc/passwd')",
        "SELECT sys_eval('id')",
    ]:
        with pytest.raises(SQLSecurityViolation):
            execute_safe_query(
                engine=mysql_readonly_engine,
                query=dangerous_sql,
                allowed_tables=allowed_tables,
                dialect="mysql",
            )


def test_live_mysql_system_schemas_blocked(mysql_readonly_engine):
    allowed_tables = {"customers"}
    with pytest.raises(SQLSecurityViolation):
        execute_safe_query(
            engine=mysql_readonly_engine,
            query="SELECT * FROM mysql.user",
            allowed_tables=allowed_tables,
            dialect="mysql",
        )


def test_live_mysql_recursive_cte(mysql_readonly_engine):
    df, is_truncated = execute_safe_query(
        engine=mysql_readonly_engine,
        query="WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 5) SELECT x FROM cnt",
        allowed_tables=None,
        dialect="mysql",
    )
    assert len(df) == 5
    assert not is_truncated


def test_live_mysql_quoted_identifiers(mysql_readonly_engine):
    allowed_tables = {"customers"}
    df, _ = execute_safe_query(
        engine=mysql_readonly_engine,
        query="SELECT `name`, `city` FROM `customers` WHERE `id` = 1",
        allowed_tables=allowed_tables,
        dialect="mysql",
    )
    assert len(df) == 1
    assert "name" in df.columns


def test_live_mysql_db_role_blocks_write_independently(mysql_readonly_engine):
    """Verifies that MySQL DB user privileges reject write operations."""
    with pytest.raises(DBAPIError) as exc_info, mysql_readonly_engine.connect() as conn:
        conn.execute(
            text("INSERT INTO customers (name, email) VALUES ('Bypassed', 'b@b.com')")
        )
    assert (
        "command denied" in str(exc_info.value).lower()
        or "access denied" in str(exc_info.value).lower()
    )


def test_live_mysql_db_role_blocks_unauthorized_table(mysql_readonly_engine):
    """Verifies that MySQL user cannot read secret_passwords at the DB permissions level."""
    with pytest.raises(DBAPIError) as exc_info, mysql_readonly_engine.connect() as conn:
        conn.execute(text("SELECT * FROM secret_passwords"))
    assert (
        "command denied" in str(exc_info.value).lower()
        or "select command denied" in str(exc_info.value).lower()
    )


def test_live_mysql_schema_inspection_foreign_keys(mysql_admin_engine):
    """Verifies that MySQL foreign keys are accurately parsed by schema_inspector."""
    schema = inspect_database_schema(mysql_admin_engine)
    orders_tbl = schema.tables["orders"]
    assert any(fk.referred_table == "customers" for fk in orders_tbl.foreign_keys)


def test_live_mysql_malformed_sql(mysql_readonly_engine):
    """Verifies malformed SQL syntax is rejected on MySQL dialect."""
    with pytest.raises(SQLSecurityViolation):
        execute_safe_query(
            engine=mysql_readonly_engine,
            query="SELECT FROM WHERE;",
            allowed_tables={"customers"},
            dialect="mysql",
        )


def test_live_mysql_pool_reuse_after_execution_failure(mysql_readonly_engine):
    """Verifies MySQL session and connection pool health after query failure."""
    with pytest.raises(QueryExecutionError):
        execute_safe_query(
            engine=mysql_readonly_engine,
            query="SELECT nonexistent_column FROM customers",
            allowed_tables={"customers"},
            dialect="mysql",
        )

    # Subsequent query succeeds without issue
    df, _ = execute_safe_query(
        engine=mysql_readonly_engine,
        query="SELECT name FROM customers LIMIT 2",
        allowed_tables={"customers"},
        dialect="mysql",
    )
    assert len(df) == 2


# ─────────────────────────────────────────────────────────────
# SQLITE LIVE INTEGRATION & ADVERSARIAL TESTS
# ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sqlite_test_engine(tmp_path_factory):
    db_file = tmp_path_factory.mktemp("sqlite_live") / "live_test.db"
    url = f"sqlite:///{db_file}"
    engine = create_engine(url)

    with engine.connect() as conn:
        conn.execute(
            text("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                city TEXT
            );
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE analytics_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event TEXT
            );
        """)
        )
        for i in range(1, 26):
            conn.execute(
                text("INSERT INTO customers (name, city) VALUES (:name, :city)"),
                {
                    "name": f"Customer {i}",
                    "city": "Delhi" if i % 2 == 0 else "Bangalore",
                },
            )
        for _ in range(500):
            conn.execute(
                text(
                    "INSERT INTO analytics_log (event) VALUES ('click'), ('view'), ('cart')"
                )
            )
        conn.commit()

    yield engine
    dispose_engine(engine)


def test_live_sqlite_schema_inspection(sqlite_test_engine):
    schema = inspect_database_schema(sqlite_test_engine)
    assert "customers" in schema.tables
    assert "analytics_log" in schema.tables


def test_live_sqlite_safe_execution(sqlite_test_engine):
    df, is_truncated = execute_safe_query(
        engine=sqlite_test_engine,
        query="SELECT name, city FROM customers ORDER BY id LIMIT 5",
        allowed_tables={"customers"},
        dialect="sqlite",
        max_rows=10,
    )
    assert len(df) == 5
    assert not is_truncated


def test_live_sqlite_row_cap_truncation(sqlite_test_engine):
    df, is_truncated = execute_safe_query(
        engine=sqlite_test_engine,
        query="SELECT * FROM analytics_log",
        allowed_tables={"analytics_log"},
        dialect="sqlite",
        max_rows=250,
    )
    assert len(df) == 250
    assert is_truncated is True


def test_live_sqlite_timeout_and_reuse(sqlite_test_engine):
    """Verifies SQLite progress handler interrupt and subsequent connection reuse."""
    slow_cte = """
        WITH RECURSIVE cnt(x) AS (
            SELECT 1
            UNION ALL
            SELECT x + 1 FROM cnt
        )
        SELECT count(*) FROM cnt
    """
    with pytest.raises(QueryTimeoutError):
        execute_safe_query(
            engine=sqlite_test_engine,
            query=slow_cte,
            allowed_tables=None,
            dialect="sqlite",
            timeout_ms=100,
        )

    # Next query on same SQLite engine succeeds and progress handler was cleanly deregistered
    df, _ = execute_safe_query(
        engine=sqlite_test_engine,
        query="SELECT name FROM customers LIMIT 3",
        allowed_tables={"customers"},
        dialect="sqlite",
        timeout_ms=5000,
    )
    assert len(df) == 3


def test_live_sqlite_query_only_enforcement(sqlite_test_engine):
    """Verifies SQLite PRAGMA query_only prevents writes even if raw SQL is executed."""
    with sqlite_test_engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA query_only = ON;")
        with pytest.raises(DBAPIError) as exc_info:
            conn.execute(text("INSERT INTO customers (name) VALUES ('Blocked')"))
        assert "attempt to write a readonly database" in str(exc_info.value).lower()


# ─────────────────────────────────────────────────────────────
# LIVE PRIVILEGE INSPECTION & RESOURCE ISOLATION TESTS
# ─────────────────────────────────────────────────────────────


def test_live_postgres_privilege_inspection(
    postgres_admin_engine, postgres_readonly_engine
):
    """Verifies non-heuristic catalog-based privilege inspection on PostgreSQL."""
    admin_report = inspect_account_privileges(
        postgres_admin_engine, db_type="PostgreSQL"
    )
    assert admin_report.is_privileged is True
    assert any("superuser" in r.lower() for r in admin_report.reasons)

    ro_report = inspect_account_privileges(
        postgres_readonly_engine, db_type="PostgreSQL"
    )
    assert ro_report.is_privileged is False
    assert "least-privilege" in ro_report.summary


def test_live_postgres_enforce_least_privilege_rejection():
    """Verifies that create_safe_engine can reject privileged accounts at connection time."""
    if not is_port_open("127.0.0.1", 5433):
        pytest.skip("PostgreSQL container unavailable.")

    # Privileged connection with enforcement enabled must fail
    with pytest.raises(PrivilegedAccountError):
        create_safe_engine(
            db_type="PostgreSQL",
            host="127.0.0.1",
            port=5433,
            db_name="schemasense_test",
            user="test_admin",
            password="test_admin_pass",
            enforce_least_privilege=True,
        )

    # Least-privilege connection with enforcement enabled must succeed
    ro_engine = create_safe_engine(
        db_type="PostgreSQL",
        host="127.0.0.1",
        port=5433,
        db_name="schemasense_test",
        user="schemasense_readonly",
        password="readonly_pass",
        enforce_least_privilege=True,
    )
    assert ro_engine is not None
    dispose_engine(ro_engine)


def test_live_mysql_privilege_inspection(mysql_admin_engine, mysql_readonly_engine):
    """Verifies non-heuristic grant-based privilege inspection on MySQL."""
    admin_report = inspect_account_privileges(mysql_admin_engine, db_type="MySQL")
    assert admin_report.is_privileged is True
    assert any("all privileges" in r.lower() for r in admin_report.reasons)

    ro_report = inspect_account_privileges(mysql_readonly_engine, db_type="MySQL")
    assert ro_report.is_privileged is False
    assert "least-privilege" in ro_report.summary


def test_live_mysql_enforce_least_privilege_rejection():
    """Verifies that create_safe_engine rejects privileged MySQL accounts when requested."""
    if not is_port_open("127.0.0.1", 3307):
        pytest.skip("MySQL container unavailable.")

    # Admin connection must fail under least-privilege policy
    with pytest.raises(PrivilegedAccountError):
        create_safe_engine(
            db_type="MySQL",
            host="127.0.0.1",
            port=3307,
            db_name="schemasense_test",
            user="test_admin",
            password="test_admin_pass",
            enforce_least_privilege=True,
        )

    # Read-only user passes
    ro_engine = create_safe_engine(
        db_type="MySQL",
        host="127.0.0.1",
        port=3307,
        db_name="schemasense_test",
        user="schemasense_readonly",
        password="readonly_pass",
        enforce_least_privilege=True,
    )
    assert ro_engine is not None
    dispose_engine(ro_engine)


def test_live_pool_resource_bounds_configured(
    postgres_readonly_engine, mysql_readonly_engine
):
    """Verifies that connection pool bounds (size, overflow, timeout) are actively configured."""
    assert postgres_readonly_engine.pool.size() == 5
    assert postgres_readonly_engine.pool._max_overflow == 2
    assert postgres_readonly_engine.pool._timeout == 10

    assert mysql_readonly_engine.pool.size() == 5
    assert mysql_readonly_engine.pool._max_overflow == 2
    assert mysql_readonly_engine.pool._timeout == 10
