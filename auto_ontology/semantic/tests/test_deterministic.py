# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic column mapping."""

from __future__ import annotations

import pytest

from unittest.mock import patch

from auto_ontology.semantic.deterministic import (
    column_attribute_specs,
    fk_source_columns,
    fk_target_table_names,
    to_term_name,
)


@patch(
    "auto_ontology.semantic.deterministic._generate_column_descriptions",
    return_value={},
)
def test_excludes_suggested_fk_columns(_mock_desc) -> None:
    columns = [
        {"name": "id", "data_type": "integer"},
        {"name": "vendor_id", "data_type": "integer"},
        {"name": "amount", "data_type": "numeric"},
    ]
    specs = column_attribute_specs(
        columns,
        [],
        suggested_fk_columns={"vendor_id"},
    )
    assert {s.source_column for s in specs} == {"id", "amount"}


@patch(
    "auto_ontology.semantic.deterministic._generate_column_descriptions",
    return_value={},
)
def test_excludes_fk_columns(_mock_desc) -> None:
    columns = [
        {"name": "id", "data_type": "integer"},
        {"name": "customer_id", "data_type": "integer"},
        {"name": "amount", "data_type": "numeric"},
    ]
    fks = [{"source_column": "customer_id", "target_table": "customers"}]
    specs = column_attribute_specs(columns, fks)
    assert {s.source_column for s in specs} == {"id", "amount"}
    assert fk_source_columns(fks) == {"customer_id"}


@patch(
    "auto_ontology.semantic.deterministic._generate_column_descriptions",
    return_value={"amount": "The monetary amount of the order."},
)
def test_llm_description_used_as_fallback(_mock_desc) -> None:
    columns = [
        {"name": "id", "data_type": "integer", "description": "Primary key."},
        {"name": "amount", "data_type": "numeric"},
    ]
    specs = {s.source_column: s for s in column_attribute_specs(columns, [])}
    # Existing description wins over the LLM one.
    assert specs["id"].description == "Primary key."
    # LLM description fills in when the column has none.
    assert specs["amount"].description == "The monetary amount of the order."


@patch(
    "auto_ontology.semantic.deterministic._generate_column_descriptions",
    return_value={},
)
def test_date_column_description_states_stored_notation(_mock_desc) -> None:
    columns = [
        {
            "name": "game_date",
            "data_type": "text",
            "description": "Date the match was played.",
        }
    ]
    profiling = {"game_date": {"format": "YYMMDD", "sample_values": []}}
    specs = column_attribute_specs(columns, [], columns_profiling_samples=profiling)
    assert specs[0].description == ("Date the match was played. — format: YYMMDD")


@patch(
    "auto_ontology.semantic.deterministic._generate_column_descriptions",
    return_value={"game_date": "When the match took place."},
)
def test_llm_date_description_also_gets_the_notation(_mock_desc) -> None:
    columns = [{"name": "game_date", "data_type": "date"}]
    profiling = {"game_date": {"format": "YYYY-MM-DD"}}
    specs = column_attribute_specs(columns, [], columns_profiling_samples=profiling)
    assert specs[0].description == ("When the match took place. — format: YYYY-MM-DD")


def test_to_term_name() -> None:
    assert to_term_name("purchase_orders") == "Purchase Orders"
    assert to_term_name("delta_lite_event") == "Delta Lite Event"
    assert to_term_name("gtl_imaging_event") == "Gtl Imaging Event"
    assert to_term_name("gtl_ui_event") == "Gtl Ui Event"
    assert to_term_name("GtlImagingEvent") == "Gtl Imaging Event"
    assert to_term_name("XMLHttpRequest") == "XML Http Request"


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ("고객", "고객"),
        ("주문_내역", "주문 내역"),
        ("TB_고객", "TB 고객"),
        ("顧客マスタ", "顧客マスタ"),
        ("клиенты", "Клиенты"),
        ("Société_Générale", "Société Générale"),
        # Decomposed accents (as macOS tooling often writes them) stay in the word.
        ("Socie\u0301te\u0301", "Société"),
        # Devanagari vowel signs and virama are marks, not separators.
        ("ग्राहक_सूची", "ग्राहक सूची"),
        # Nothing word-like left: the name itself, never "".
        ("___", "___"),
    ],
)
def test_to_term_name_keeps_words_in_any_script(table: str, expected: str) -> None:
    assert to_term_name(table) == expected


def test_fk_targets_deduped() -> None:
    fks = [
        {"target_table": "customers"},
        {"target_table": "customers"},
        {"target_table": "products"},
    ]
    assert fk_target_table_names(fks) == ["customers", "products"]
