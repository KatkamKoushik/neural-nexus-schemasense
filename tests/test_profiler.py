"""Tests for Data Quality Engine and statistical profiler."""

import pandas as pd

from schemasense.profiler import (
    build_safe_profiling_query,
    profile_dataframe,
)


def test_profile_dataframe_metrics():
    data = {
        "user_id": [1, 2, 3, 4, 5],
        "email": ["a@test.com", "b@test.com", None, "d@test.com", "e@test.com"],
        "age": [20, 25, 30, 22, 120],  # 120 is an outlier
        "city": ["Delhi", "Delhi", "Mumbai", "Delhi", "Mumbai"],
    }
    df = pd.DataFrame(data)

    report = profile_dataframe(df, "test_users")
    assert report.row_count == 5
    assert report.column_count == 4
    assert report.duplicate_rows_count == 0
    assert 0.0 <= report.overall_quality_score <= 100.0

    # Test candidate key detection
    user_id_col = next(c for c in report.columns if c.name == "user_id")
    assert user_id_col.is_candidate_key is True
    assert user_id_col.null_count == 0
    assert user_id_col.uniqueness_ratio == 1.0

    # Test null detection
    email_col = next(c for c in report.columns if c.name == "email")
    assert email_col.null_count == 1
    assert email_col.null_percentage == 20.0
    assert email_col.is_candidate_key is False

    # Test outlier detection
    age_col = next(c for c in report.columns if c.name == "age")
    assert age_col.outlier_count >= 1
    assert age_col.min_value == 20.0
    assert age_col.max_value == 120.0


def test_profile_dataframe_empty():
    empty_df = pd.DataFrame()
    report = profile_dataframe(empty_df, "empty_table")
    assert report.row_count == 0
    assert report.overall_quality_score == 100.0


def test_build_safe_profiling_query():
    q_pg = build_safe_profiling_query(
        "user orders", dialect="postgres", sample_size=500
    )
    assert q_pg == 'SELECT * FROM "user orders" LIMIT 500'

    q_my = build_safe_profiling_query(
        "schema.orders", dialect="mysql", sample_size=2000
    )
    assert q_my == "SELECT * FROM `schema`.`orders` LIMIT 2000"
