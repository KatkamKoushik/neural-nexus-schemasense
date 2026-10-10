"""Data Profiling & Quality Engine — Statistical metrics, outlier detection, and quality scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    data_type: str
    total_count: int
    null_count: int
    null_percentage: float
    distinct_count: int
    uniqueness_ratio: float
    is_candidate_key: bool
    # Numeric statistics
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    median_value: float | None = None
    std_value: float | None = None
    outlier_count: int = 0
    # Top values
    top_values: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class TableQualityReport:
    table_name: str
    row_count: int
    column_count: int
    duplicate_rows_count: int
    duplicate_rows_percentage: float
    overall_quality_score: float
    columns: list[ColumnProfile] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_dataframe(df: pd.DataFrame, table_name: str) -> TableQualityReport:
    """Profiles a pandas DataFrame calculating data quality metrics and composite quality score."""
    row_count = len(df)
    col_count = len(df.columns)

    if row_count == 0 or col_count == 0:
        return TableQualityReport(
            table_name=table_name,
            row_count=0,
            column_count=col_count,
            duplicate_rows_count=0,
            duplicate_rows_percentage=0.0,
            overall_quality_score=100.0,
            columns=[],
        )

    # Duplicate row calculation
    dup_rows = int(df.duplicated().sum())
    dup_percentage = round((dup_rows / row_count) * 100, 2)

    column_profiles: list[ColumnProfile] = []
    total_null_ratio = 0.0
    outlier_ratios = []

    for col in df.columns:
        s = df[col]
        total = len(s)
        null_cnt = int(s.isna().sum())
        null_pct = round((null_cnt / total) * 100, 2) if total > 0 else 0.0
        total_null_ratio += (null_cnt / total) if total > 0 else 0.0

        non_null = s.dropna()
        distinct_cnt = int(non_null.nunique())
        uniqueness = (
            round((distinct_cnt / len(non_null)), 4) if len(non_null) > 0 else 0.0
        )
        is_candidate_key = null_cnt == 0 and distinct_cnt == total and total > 0

        # Top values (most frequent)
        top_vals: list[tuple[str, int]] = []
        if len(non_null) > 0:
            val_counts = non_null.value_counts().head(5)
            top_vals = [(str(idx), int(cnt)) for idx, cnt in val_counts.items()]

        # Numeric stats and outlier detection
        min_v = max_v = mean_v = median_v = std_v = None
        outlier_cnt = 0

        if (
            pd.api.types.is_numeric_dtype(s)
            and not pd.api.types.is_bool_dtype(s)
            and len(non_null) > 0
        ):
            numeric_vals = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(numeric_vals) > 0:
                min_v = round(float(numeric_vals.min()), 3)
                max_v = round(float(numeric_vals.max()), 3)
                mean_v = round(float(numeric_vals.mean()), 3)
                median_v = round(float(numeric_vals.median()), 3)
                std_v = (
                    round(float(numeric_vals.std()), 3)
                    if len(numeric_vals) > 1
                    else 0.0
                )

                # IQR Outlier Detection
                q25 = float(numeric_vals.quantile(0.25))
                q75 = float(numeric_vals.quantile(0.75))
                iqr = q75 - q25
                if iqr > 0:
                    lower_bound = q25 - (1.5 * iqr)
                    upper_bound = q75 + (1.5 * iqr)
                    outliers = numeric_vals[
                        (numeric_vals < lower_bound) | (numeric_vals > upper_bound)
                    ]
                    outlier_cnt = len(outliers)
                    outlier_ratios.append(outlier_cnt / len(numeric_vals))

        column_profiles.append(
            ColumnProfile(
                name=str(col),
                data_type=str(s.dtype),
                total_count=total,
                null_count=null_cnt,
                null_percentage=null_pct,
                distinct_count=distinct_cnt,
                uniqueness_ratio=uniqueness,
                is_candidate_key=is_candidate_key,
                min_value=min_v,
                max_value=max_v,
                mean_value=mean_v,
                median_value=median_v,
                std_value=std_v,
                outlier_count=outlier_cnt,
                top_values=top_vals,
            )
        )

    # Composite Quality Score Calculation
    avg_null_ratio = total_null_ratio / col_count if col_count > 0 else 0.0
    completeness_score = max(0.0, (1.0 - avg_null_ratio) * 100.0)

    uniqueness_score = max(0.0, (1.0 - (dup_rows / row_count)) * 100.0)

    avg_outlier_ratio = (
        (sum(outlier_ratios) / len(outlier_ratios)) if outlier_ratios else 0.0
    )
    validity_score = max(0.0, 100.0 - (avg_outlier_ratio * 50.0))

    # Weights: 50% Completeness, 30% Uniqueness, 20% Validity
    composite_score = round(
        (0.5 * completeness_score) + (0.3 * uniqueness_score) + (0.2 * validity_score),
        1,
    )

    return TableQualityReport(
        table_name=table_name,
        row_count=row_count,
        column_count=col_count,
        duplicate_rows_count=dup_rows,
        duplicate_rows_percentage=dup_percentage,
        overall_quality_score=composite_score,
        columns=column_profiles,
    )


def build_safe_profiling_query(
    table_name: str, dialect: str = "postgres", sample_size: int = 1000
) -> str:
    """Safely quotes table names and appends sample limits to prevent syntax errors and table injection."""
    clean_table = table_name.strip()
    cap = max(1, min(sample_size, 5000))

    # Handle schema.table qualified names
    if "." in clean_table:
        parts = clean_table.split(".", 1)
        if dialect.lower() == "mysql":
            quoted = f"`{parts[0]}`.`{parts[1]}`"
        else:
            quoted = f'"{parts[0]}"."{parts[1]}"'
    else:
        if dialect.lower() == "mysql":
            quoted = f"`{clean_table}`"
        else:
            quoted = f'"{clean_table}"'

    return f"SELECT * FROM {quoted} LIMIT {cap}"
