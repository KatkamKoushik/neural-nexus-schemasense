from sqlalchemy import create_engine, text

from schemasense.schema_inspector import (
    ColumnInfo,
    DatabaseSchema,
    ForeignKeyInfo,
    TableSchema,
    diff_database_schemas,
    inspect_database_schema,
)


def test_inspect_database_schema(sqlite_engine):
    schema = inspect_database_schema(sqlite_engine)
    assert "customers" in schema.tables
    assert "orders" in schema.tables
    assert "secret_passwords" in schema.tables

    cust = schema.tables["customers"]
    assert "id" in cust.primary_key
    assert len(cust.columns) >= 4

    orders = schema.tables["orders"]
    assert len(orders.foreign_keys) == 1
    assert orders.foreign_keys[0].referred_table == "customers"
    assert orders.foreign_keys[0].constrained_column == "customer_id"


def test_schema_exports(sqlite_engine):
    schema = inspect_database_schema(sqlite_engine)

    # Test JSON export
    json_out = schema.to_json()
    assert '"database_name":' in json_out
    assert '"customers"' in json_out

    # Test Markdown export
    md_out = schema.to_markdown()
    assert "# Data Dictionary:" in md_out
    assert "## Table: `customers`" in md_out
    assert "🔑 Yes" in md_out

    # Test Prompt context formatting
    prompt_ctx = schema.to_prompt_context()
    assert "<database_schema>" in prompt_ctx
    assert "</database_schema>" in prompt_ctx
    assert "Table: customers" in prompt_ctx
    assert "Table: secret_passwords" in prompt_ctx

    # Test Selective Disclosure Filter
    filtered_ctx = schema.to_prompt_context(table_filter=["customers"])
    assert "Table: customers" in filtered_ctx
    assert "Table: secret_passwords" not in filtered_ctx


def test_schema_diffing():
    old_schema = DatabaseSchema(
        dialect="postgres",
        database_name="app_db",
        tables={
            "users": TableSchema(
                name="users",
                columns=[
                    ColumnInfo("id", "INTEGER", False, is_primary_key=True),
                    ColumnInfo("name", "VARCHAR(100)", True),
                ],
            ),
            "old_table": TableSchema(name="old_table", columns=[]),
        },
    )

    new_schema = DatabaseSchema(
        dialect="postgres",
        database_name="app_db",
        tables={
            "users": TableSchema(
                name="users",
                columns=[
                    ColumnInfo("id", "INTEGER", False, is_primary_key=True),
                    ColumnInfo("name", "VARCHAR(100)", True),
                    ColumnInfo("created_at", "TIMESTAMP", False),
                ],
            ),
            "new_table": TableSchema(name="new_table", columns=[]),
        },
    )

    diff = diff_database_schemas(old_schema, new_schema)
    assert "new_table" in diff["added_tables"]
    assert "old_table" in diff["removed_tables"]
    assert "users" in diff["modified_tables"]
    assert "created_at" in diff["modified_tables"]["users"]["added_columns"]


def test_composite_foreign_key_two_columns():
    """Verifies that a 2-column composite foreign key preserves both column mappings in order."""
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(
            text("""
            CREATE TABLE customers (
                tenant_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                name TEXT,
                PRIMARY KEY (tenant_id, customer_id)
            );
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE orders (
                tenant_id TEXT NOT NULL,
                order_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                amount REAL,
                PRIMARY KEY (tenant_id, order_id),
                FOREIGN KEY (tenant_id, customer_id) REFERENCES customers (tenant_id, customer_id)
            );
        """)
        )

    schema = inspect_database_schema(engine)
    assert "orders" in schema.tables
    orders = schema.tables["orders"]

    assert len(orders.foreign_keys) == 1
    fk = orders.foreign_keys[0]
    assert fk.is_composite is True
    assert fk.referred_table == "customers"
    assert fk.constrained_columns == ["tenant_id", "customer_id"]
    assert fk.referred_columns == ["tenant_id", "customer_id"]
    assert fk.column_pairs == [
        ("tenant_id", "tenant_id"),
        ("customer_id", "customer_id"),
    ]
    assert fk.constrained_column == "(tenant_id, customer_id)"
    assert fk.referred_column == "(tenant_id, customer_id)"

    # Verify downstream representations
    md_out = schema.to_markdown()
    assert "(tenant_id, customer_id)" in md_out
    assert "customers.(tenant_id, customer_id)" in md_out

    prompt_ctx = schema.to_prompt_context()
    assert (
        "(tenant_id, customer_id -> customers.(tenant_id, customer_id))" in prompt_ctx
    )

    json_dict = schema.to_dict()
    fk_dict = json_dict["tables"]["orders"]["foreign_keys"][0]
    assert fk_dict["constrained_columns"] == ["tenant_id", "customer_id"]
    assert fk_dict["referred_columns"] == ["tenant_id", "customer_id"]


def test_composite_foreign_key_three_columns():
    """Verifies that a 3-column composite foreign key preserves all column pairs and deterministic ordering."""
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(
            text("""
            CREATE TABLE inventory (
                org_id TEXT NOT NULL,
                warehouse_id TEXT NOT NULL,
                sku_id TEXT NOT NULL,
                stock INTEGER,
                PRIMARY KEY (org_id, warehouse_id, sku_id)
            );
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE shipment_items (
                shipment_id TEXT NOT NULL,
                org_id TEXT NOT NULL,
                warehouse_id TEXT NOT NULL,
                sku_id TEXT NOT NULL,
                qty INTEGER,
                PRIMARY KEY (shipment_id, org_id, warehouse_id, sku_id),
                FOREIGN KEY (org_id, warehouse_id, sku_id) REFERENCES inventory (org_id, warehouse_id, sku_id)
            );
        """)
        )

    schema = inspect_database_schema(engine)
    items = schema.tables["shipment_items"]
    assert len(items.foreign_keys) == 1
    fk = items.foreign_keys[0]

    assert fk.is_composite is True
    assert fk.constrained_columns == ["org_id", "warehouse_id", "sku_id"]
    assert fk.referred_columns == ["org_id", "warehouse_id", "sku_id"]
    assert fk.column_pairs == [
        ("org_id", "org_id"),
        ("warehouse_id", "warehouse_id"),
        ("sku_id", "sku_id"),
    ]
    assert fk.constrained_column == "(org_id, warehouse_id, sku_id)"
    assert fk.referred_column == "(org_id, warehouse_id, sku_id)"


def test_foreign_key_info_instantiation_styles():
    """Verifies that ForeignKeyInfo supports both legacy single-column and composite instantiation."""
    # Legacy positional strings
    fk_legacy = ForeignKeyInfo("user_id", "users", "id")
    assert not fk_legacy.is_composite
    assert fk_legacy.constrained_columns == ["user_id"]
    assert fk_legacy.referred_columns == ["id"]
    assert fk_legacy.constrained_column == "user_id"
    assert fk_legacy.referred_column == "id"
    assert fk_legacy.column_pairs == [("user_id", "id")]

    # Legacy keyword args
    fk_kw = ForeignKeyInfo(
        constrained_column="acct_id", referred_table="accounts", referred_column="id"
    )
    assert fk_kw.constrained_columns == ["acct_id"]
    assert fk_kw.referred_column == "id"

    # Modern composite lists
    fk_comp = ForeignKeyInfo(
        constrained_columns=["tenant_id", "order_id"],
        referred_table="orders",
        referred_columns=["tenant_id", "id"],
        name="fk_order_ref",
    )
    assert fk_comp.is_composite is True
    assert fk_comp.name == "fk_order_ref"
    assert fk_comp.column_pairs == [("tenant_id", "tenant_id"), ("order_id", "id")]


def test_defensive_foreign_key_metadata_handling():
    """Verifies defensive handling for mismatched, missing, or malformed metadata."""
    # Mismatched column count
    fk_mismatch = ForeignKeyInfo(["a", "b"], "tbl", ["c"])
    assert (
        fk_mismatch.column_pairs == []
    )  # Cannot safely pair mismatched column lengths

    # Empty columns
    fk_empty = ForeignKeyInfo([], "tbl", [])
    assert not fk_empty.is_composite
    assert fk_empty.constrained_column == ""
    assert fk_empty.referred_column == ""
    assert fk_empty.column_pairs == []
