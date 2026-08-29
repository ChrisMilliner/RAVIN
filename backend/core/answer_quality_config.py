import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_ANSWER_QUALITY_SCHEMA_VERSION = 1
DEVELOPMENT_NOT_VALIDATED_STATUS = (
    "development-not-validated"
)
VALIDATED_STATUS = "validated"
_ALLOWED_STATUSES = frozenset(
    {
        DEVELOPMENT_NOT_VALIDATED_STATUS,
        VALIDATED_STATUS,
    }
)
DEFAULT_ANSWER_QUALITY_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "answer_quality.json"
)

@dataclass(frozen=True)
class AnswerQualityConfig:
    schema_version: int
    status: str
    proposition_covered_threshold: float
    proposition_partial_threshold: float
    claim_support_threshold: float

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != SUPPORTED_ANSWER_QUALITY_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported answer quality "
                "schema version."
            )

        if self.status not in _ALLOWED_STATUSES:
            raise ValueError(
                "Unsupported answer quality "
                "configuration status."
            )

        if not (
            0.0
            <= self.proposition_partial_threshold
            < self.proposition_covered_threshold
            <= 1.0
        ):
            raise ValueError(
                "Proposition coverage thresholds "
                "must satisfy "
                "0.0 <= partial < covered <= 1.0."
            )

        if not (
            0.0
            <= self.claim_support_threshold
            <= 1.0
        ):
            raise ValueError(
                "Claim support threshold must be "
                "between 0 and 1."
            )

def load_answer_quality_config(
    path: str | Path | None = None,
) -> AnswerQualityConfig:
    target_path = (
        DEFAULT_ANSWER_QUALITY_CONFIG_PATH
        if path is None
        else Path(path)
    )

    try:
        raw_data = json.loads(
            target_path.read_text(
                encoding="utf-8"
            )
        )

    except FileNotFoundError as error:
        raise ValueError(
            "Answer quality configuration "
            "file was not found."
        ) from error

    except json.JSONDecodeError as error:
        raise ValueError(
            "Answer quality configuration "
            "contains invalid JSON."
        ) from error

    root = _require_mapping(
        raw_data,
        "Answer quality configuration",
    )

    proposition_coverage = (
        _require_mapping(
            root.get(
                "proposition_coverage"
            ),
            "proposition_coverage",
        )
    )

    claim_grounding = (
        _require_mapping(
            root.get(
                "claim_grounding"
            ),
            "claim_grounding",
        )
    )

    return AnswerQualityConfig(
        schema_version=_require_integer(
            root.get(
                "schema_version"
            ),
            "schema_version",
        ),
        status=_require_string(
            root.get(
                "status"
            ),
            "status",
        ),
        proposition_covered_threshold=(
            _require_number(
                proposition_coverage.get(
                    "covered_threshold"
                ),
                "covered_threshold",
            )
        ),
        proposition_partial_threshold=(
            _require_number(
                proposition_coverage.get(
                    "partial_threshold"
                ),
                "partial_threshold",
            )
        ),
        claim_support_threshold=(
            _require_number(
                claim_grounding.get(
                    "support_threshold"
                ),
                "support_threshold",
            )
        ),
    )

def _require_mapping(
    value: Any,
    name: str,
) -> dict[str, Any]:
    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"{name} must be an object."
        )

    return value

def _require_integer(
    value: Any,
    name: str,
) -> int:
    if (
        not isinstance(
            value,
            int,
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise ValueError(
            f"{name} must be an integer."
        )

    return value

def _require_string(
    value: Any,
    name: str,
) -> str:
    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    return value

def _require_number(
    value: Any,
    name: str,
) -> float:
    if (
        not isinstance(
            value,
            (
                int,
                float,
            ),
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise ValueError(
            f"{name} must be a number."
        )

    return float(
        value
    )