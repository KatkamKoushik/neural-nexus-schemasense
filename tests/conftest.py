"""Test fixtures and disposable database setups."""

import pytest
from sqlalchemy import create_engine, text


@pytest.fixture
def sqlite_engine():
    """Provides an in-memory SQLite engine populated with a representative schema."""
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(
            text("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                city TEXT
            );
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                amount REAL,
                order_date TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers (id)
            );
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE secret_passwords (
                id INTEGER PRIMARY KEY,
                hash TEXT NOT NULL
            );
        """)
        )

        # Seed data
        for i in range(1, 26):
            conn.execute(
                text(
                    "INSERT INTO customers (id, name, email, city) VALUES (:id, :name, :email, :city)"
                ),
                {
                    "id": i,
                    "name": f"Customer {i}",
                    "email": f"cust{i}@example.com",
                    "city": "Delhi" if i % 2 == 0 else "Mumbai",
                },
            )
            conn.execute(
                text(
                    "INSERT INTO orders (order_id, customer_id, amount, order_date) VALUES (:oid, :cid, :amt, :dt)"
                ),
                {"oid": 100 + i, "cid": i, "amt": 50.0 * i, "dt": "2026-01-01"},
            )
        conn.commit()

    yield engine
    engine.dispose()
