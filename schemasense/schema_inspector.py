"""Schema Inspector — Deep structural intelligence, constraints extraction, and export."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy import Engine, inspect
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger("schemasense.inspector")


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool
    default: str | None = None
    is_primary_key: bool = False
    comment: str | None = None


@dataclass
class ForeignKeyInfo:
    constrained_columns: list[str] = field(default_factory=list)
    referred_table: str = ""
    referred_columns: list[str] = field(default_factory=list)
    name: str | None = None

    def __init__(
        self,
        constrained_columns: list[str] | str | None = None,
        referred_table: str = "",
        referred_columns: list[str] | str | None = None,
        name: str | None = None,
        constrained_column: str | None = None,
        referred_column: str | None = None,
    ) -> None:
        self.referred_table = referred_table
        self.name = name

        # Resolve constrained column(s)
        if constrained_columns is not None:
            if isinstance(constrained_columns, str):
                self.constrained_columns = [constrained_columns]
            else:
                self.constrained_columns = list(constrained_columns)
        elif constrained_column is not None:
            self.constrained_columns = [constrained_column]
        else:
            self.constrained_columns = []

        # Resolve referred column(s)
        if referred_columns is not None:
            if isinstance(referred_columns, str):
                self.referred_columns = [referred_columns]
            else:
                self.referred_columns = list(referred_columns)
        elif referred_column is not None:
            self.referred_columns = [referred_column]
        else:
            self.referred_columns = []

    @property
    def constrained_column(self) -> str:
        """First constrained column, or formatted tuple if composite."""
        if len(self.constrained_columns) == 1:
            return self.constrained_columns[0]
        if self.constrained_columns:
            return f"({', '.join(self.constrained_columns)})"
        return ""

    @property
    def referred_column(self) -> str:
        """First referred column, or formatted tuple if composite."""
        if len(self.referred_columns) == 1:
            return self.referred_columns[0]
        if self.referred_columns:
            return f"({', '.join(self.referred_columns)})"
        return ""

    @property
    def is_composite(self) -> bool:
        """True if relationship spans more than one column."""
        return len(self.constrained_columns) > 1

    @property
    def column_pairs(self) -> list[tuple[str, str]]:
        """List of paired (constrained_col, referred_col) tuples in deterministic order."""
        if len(self.constrained_columns) == len(self.referred_columns):
            return list(zip(self.constrained_columns, self.referred_columns))
        return []


@dataclass
class IndexInfo:
    name: str
    columns: list[str]
    unique: bool


@dataclass
class TableSchema:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKeyInfo] = field(default_factory=list)
    unique_constraints: list[list[str]] = field(default_factory=list)
    indexes: list[IndexInfo] = field(default_factory=list)
    comment: str | None = None


@dataclass
class DatabaseSchema:
    dialect: str
    database_name: str
    tables: dict[str, TableSchema] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            f"# Data Dictionary: {self.database_name}",
            f"**Dialect:** `{self.dialect}` | **Total Tables:** `{len(self.tables)}`\n",
        ]
        for t_name, table in self.tables.items():
            lines.append(f"## Table: `{t_name}`")
            if table.comment:
                lines.append(f"*{table.comment}*\n")
            lines.append("| Column | Type | Nullable | Primary Key | Default |")
            lines.append("|---|---|---|---|---|")
            for col in table.columns:
                pk_mark = "🔑 Yes" if col.is_primary_key else "No"
                null_mark = "NULL" if col.nullable else "**NOT NULL**"
                default_val = f"`{col.default}`" if col.default is not None else "-"
                lines.append(
                    f"| `{col.name}` | `{col.data_type}` | {null_mark} | {pk_mark} | {default_val} |"
                )

            if table.foreign_keys:
                lines.append("\n**Relationships (Foreign Keys):**")
                for fk in table.foreign_keys:
                    if fk.is_composite:
                        cols_src = ", ".join(fk.constrained_columns)
                        cols_ref = ", ".join(fk.referred_columns)
                        lines.append(
                            f"- `({cols_src})` ➔ `{fk.referred_table}.({cols_ref})`"
                        )
                    else:
                        lines.append(
                            f"- `{fk.constrained_column}` ➔ `{fk.referred_table}.{fk.referred_column}`"
                        )

            if table.indexes:
                lines.append("\n**Indexes:**")
                for idx in table.indexes:
                    uniq = " (UNIQUE)" if idx.unique else ""
                    lines.append(f"- `{idx.name}` on ({', '.join(idx.columns)}){uniq}")

            lines.append("\n---\n")
        return "\n".join(lines)

    def to_prompt_context(
        self, table_filter: Sequence[str] | set[str] | None = None
    ) -> str:
        """Formats the schema into a concise context block for LLM prompts with selective disclosure."""
        filter_set = (
            {t.lower() for t in table_filter} if table_filter is not None else None
        )
        lines = ["<database_schema>"]
        for t_name, table in self.tables.items():
            if filter_set is not None and t_name.lower() not in filter_set:
                continue
            cols = []
            for c in table.columns:
                pk = " [PK]" if c.is_primary_key else ""
                null = " NULL" if c.nullable else " NOT NULL"
                cols.append(f"{c.name} ({c.data_type}){pk}{null}")
            lines.append(f"Table: {t_name}")
            lines.append(f"  Columns: {', '.join(cols)}")
            if table.foreign_keys:
                fks = []
                for fk in table.foreign_keys:
                    if fk.is_composite:
                        cols_src = ", ".join(fk.constrained_columns)
                        cols_ref = ", ".join(fk.referred_columns)
                        fks.append(f"({cols_src} -> {fk.referred_table}.({cols_ref}))")
                    else:
                        fks.append(
                            f"({fk.constrained_column} -> {fk.referred_table}.{fk.referred_column})"
                        )
                lines.append(f"  Foreign Keys: {', '.join(fks)}")
            lines.append("")
        lines.append("</database_schema>")
        return "\n".join(lines)


def inspect_database_schema(
    engine: Engine, schema_name: str | None = None
) -> DatabaseSchema:
    """Extracts tables, primary keys, columns, foreign keys, and indexes from an active engine."""
    inspector = inspect(engine)
    dialect_name = engine.dialect.name
    db_name = str(engine.url.database or "database")

    tables_dict: dict[str, TableSchema] = {}
    table_names = inspector.get_table_names(schema=schema_name)

    for table_name in table_names:
        try:
            # Primary keys
            pk_constraint = (
                inspector.get_pk_constraint(table_name, schema=schema_name) or {}
            )
            pk_cols = set(pk_constraint.get("constrained_columns", []))

            # Columns
            columns: list[ColumnInfo] = []
            for col in inspector.get_columns(table_name, schema=schema_name):
                col_name = col["name"]
                columns.append(
                    ColumnInfo(
                        name=col_name,
                        data_type=str(col.get("type", "UNKNOWN")),
                        nullable=col.get("nullable", True),
                        default=str(col.get("default"))
                        if col.get("default") is not None
                        else None,
                        is_primary_key=col_name in pk_cols,
                        comment=col.get("comment"),
                    )
                )

            # Foreign keys
            fks: list[ForeignKeyInfo] = []
            for fk in inspector.get_foreign_keys(table_name, schema=schema_name):
                raw_constrained = fk.get("constrained_columns") or []
                referred_table = fk.get("referred_table")
                raw_referred_cols = fk.get("referred_columns") or []
                fk_name = fk.get("name")

                if not referred_table:
                    logger.warning(
                        "Foreign key in table '%s' missing referred_table metadata. Skipping.",
                        table_name,
                    )
                    continue

                # Defensively filter and sanitize column names, preserving ordering
                constrained_cols = [
                    str(c) for c in raw_constrained if c is not None and str(c).strip()
                ]
                referred_cols = [
                    str(c)
                    for c in raw_referred_cols
                    if c is not None and str(c).strip()
                ]

                if not constrained_cols or not referred_cols:
                    logger.warning(
                        "Foreign key '%s' on table '%s' has empty constrained or referred columns. Skipping.",
                        fk_name or "unnamed",
                        table_name,
                    )
                    continue

                # Check that parallel column lists have equal lengths before pairing
                if len(constrained_cols) != len(referred_cols):
                    logger.warning(
                        "Foreign key '%s' on table '%s' has mismatched column counts (constrained: %d, referred: %d). Skipping malformed constraint.",
                        fk_name or "unnamed",
                        table_name,
                        len(constrained_cols),
                        len(referred_cols),
                    )
                    continue

                fks.append(
                    ForeignKeyInfo(
                        constrained_columns=constrained_cols,
                        referred_table=str(referred_table),
                        referred_columns=referred_cols,
                        name=fk_name,
                    )
                )

            # Unique constraints
            unique_constraints = []
            try:
                for uc in inspector.get_unique_constraints(
                    table_name, schema=schema_name
                ):
                    cols = uc.get("column_names", [])
                    if cols:
                        unique_constraints.append(cols)
            except (SQLAlchemyError, NotImplementedError) as err:
                logger.debug(
                    f"Could not extract unique constraints for {table_name}: {err}"
                )

            # Indexes
            indexes: list[IndexInfo] = []
            try:
                for idx in inspector.get_indexes(table_name, schema=schema_name):
                    idx_name = idx.get("name") or "unnamed_idx"
                    idx_cols = idx.get("column_names") or []
                    indexes.append(
                        IndexInfo(
                            name=idx_name,
                            columns=idx_cols,
                            unique=bool(idx.get("unique", False)),
                        )
                    )
            except (SQLAlchemyError, NotImplementedError) as err:
                logger.debug(f"Could not extract indexes for {table_name}: {err}")

            tables_dict[table_name] = TableSchema(
                name=table_name,
                columns=columns,
                primary_key=list(pk_cols),
                foreign_keys=fks,
                unique_constraints=unique_constraints,
                indexes=indexes,
            )
        except (SQLAlchemyError, RuntimeError) as e:
            logger.warning(f"Could not inspect table '{table_name}': {e}")
            continue

    return DatabaseSchema(
        dialect=dialect_name,
        database_name=db_name,
        tables=tables_dict,
    )


def diff_database_schemas(
    old_schema: DatabaseSchema, new_schema: DatabaseSchema
) -> dict[str, Any]:
    """Generates a structured schema diff between two schema snapshots."""
    old_tables = set(old_schema.tables.keys())
    new_tables = set(new_schema.tables.keys())

    added_tables = list(new_tables - old_tables)
    removed_tables = list(old_tables - new_tables)
    common_tables = old_tables & new_tables

    modified_tables: dict[str, dict[str, list[str]]] = {}

    for t in common_tables:
        old_cols = {c.name: c.data_type for c in old_schema.tables[t].columns}
        new_cols = {c.name: c.data_type for c in new_schema.tables[t].columns}

        added_cols = [c for c in new_cols if c not in old_cols]
        removed_cols = [c for c in old_cols if c not in new_cols]
        type_changes = [
            f"{c} ({old_cols[c]} -> {new_cols[c]})"
            for c in old_cols
            if c in new_cols and old_cols[c] != new_cols[c]
        ]

        if added_cols or removed_cols or type_changes:
            modified_tables[t] = {
                "added_columns": added_cols,
                "removed_columns": removed_cols,
                "type_changes": type_changes,
            }

    return {
        "added_tables": added_tables,
        "removed_tables": removed_tables,
        "modified_tables": modified_tables,
    }
