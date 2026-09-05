import json
import pytest
from backend.core.answer_quality_config import (
    DEVELOPMENT_NOT_VALIDATED_STATUS,
    VALIDATED_STATUS,
    load_answer_quality_config,
)

def _write_config(
    tmp_path,
    *,
    schema_version=1,
    status=(
        DEVELOPMENT_NOT_VALIDATED_STATUS
    ),
    covered_threshold=0.8,
    partial_threshold=0.4,
    support_threshold=0.8,
):
    path = (
        tmp_path
        / "answer_quality.json"
    )

    path.write_text(
        json.dumps(
            {
                "schema_version": (
                    schema_version
                ),
                "status": status,
                "proposition_coverage": {
                    "covered_threshold": (
                        covered_threshold
                    ),
                    "partial_threshold": (
                        partial_threshold
                    ),
                },
                "claim_grounding": {
                    "support_threshold": (
                        support_threshold
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    return path

def test_loads_default_answer_quality_config():
    config = load_answer_quality_config()

    assert config.schema_version == 1

    assert config.status == (
        DEVELOPMENT_NOT_VALIDATED_STATUS
    )

    assert (
        config.proposition_covered_threshold
        == 0.8
    )

    assert (
        config.proposition_partial_threshold
        == 0.4
    )

    assert (
        config.claim_support_threshold
        == 0.8
    )

def test_validated_status_is_supported(
    tmp_path,
):
    path = _write_config(
        tmp_path,
        status=VALIDATED_STATUS,
    )

    config = load_answer_quality_config(
        path
    )

    assert config.status == (
        VALIDATED_STATUS
    )

def test_unsupported_schema_version_is_rejected(
    tmp_path,
):
    path = _write_config(
        tmp_path,
        schema_version=2,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported answer quality "
            "schema version"
        ),
    ):
        load_answer_quality_config(
            path
        )

def test_invalid_status_is_rejected(
    tmp_path,
):
    path = _write_config(
        tmp_path,
        status="unknown",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported answer quality "
            "configuration status"
        ),
    ):
        load_answer_quality_config(
            path
        )

def test_invalid_proposition_threshold_order_is_rejected(
    tmp_path,
):
    path = _write_config(
        tmp_path,
        covered_threshold=0.4,
        partial_threshold=0.8,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Proposition coverage thresholds"
        ),
    ):
        load_answer_quality_config(
            path
        )

def test_invalid_claim_support_threshold_is_rejected(
    tmp_path,
):
    path = _write_config(
        tmp_path,
        support_threshold=1.1,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Claim support threshold"
        ),
    ):
        load_answer_quality_config(
            path
        )

def test_missing_configuration_section_is_rejected(
    tmp_path,
):
    path = (
        tmp_path
        / "answer_quality.json"
    )

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": (
                    DEVELOPMENT_NOT_VALIDATED_STATUS
                ),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "proposition_coverage "
            "must be an object"
        ),
    ):
        load_answer_quality_config(
            path
        )

def test_invalid_json_is_rejected(
    tmp_path,
):
    path = (
        tmp_path
        / "answer_quality.json"
    )

    path.write_text(
        "{ invalid json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "contains invalid JSON"
        ),
    ):
        load_answer_quality_config(
            path
        )