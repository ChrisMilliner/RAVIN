"""
Deterministic evidence support windows for claim grounding.

Generated claims should not be validated against an entire policy
evidence block when the relevant support may occupy only a small
part of that block.

This module splits approved evidence into sentence-like support
units and creates consecutive windows containing up to a configured
number of units.

For example, with a maximum size of three:

    unit 1
    unit 2
    unit 3
    unit 4

produces windows such as:

    unit 1
    unit 1 + unit 2
    unit 1 + unit 2 + unit 3
    unit 2
    unit 2 + unit 3
    unit 2 + unit 3 + unit 4

The window builder contains no model-specific logic. It prepares
evidence for later scoring by an EntailmentProvider.
"""

from dataclasses import dataclass
import re

_SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[.!?])\s+"
)

@dataclass(frozen=True)
class EvidenceSupportWindow:
    """One consecutive region of an approved evidence block."""

    start_unit_index: int
    end_unit_index: int
    text: str

    def __post_init__(self) -> None:
        if self.start_unit_index < 0:
            raise ValueError(
                "Support window start index cannot be negative."
            )

        if self.end_unit_index < self.start_unit_index:
            raise ValueError(
                "Support window end index cannot precede "
                "the start index."
            )

        if not self.text.strip():
            raise ValueError(
                "Support window text cannot be empty."
            )

class EvidenceSupportWindowBuilder:
    """Build consecutive sentence-like evidence windows."""

    def __init__(
        self,
        max_units: int = 3,
    ) -> None:
        if max_units < 1:
            raise ValueError(
                "Maximum support window units must be at least 1."
            )

        self._max_units = max_units

    @property
    def max_units(self) -> int:
        """Return the maximum number of units in one window."""

        return self._max_units

    def build(
        self,
        evidence_text: str,
    ) -> tuple[EvidenceSupportWindow, ...]:
        """Create deterministic support windows for one evidence block."""

        units = split_evidence_support_units(
            evidence_text
        )

        windows: list[
            EvidenceSupportWindow
        ] = []

        seen_texts: set[str] = set()

        for start_index in range(
            len(units)
        ):
            for size in range(
                1,
                self._max_units + 1,
            ):
                end_index = (
                    start_index
                    + size
                )

                if end_index > len(units):
                    break

                text = " ".join(
                    units[
                        start_index:end_index
                    ]
                )

                if text in seen_texts:
                    continue

                seen_texts.add(
                    text
                )

                windows.append(
                    EvidenceSupportWindow(
                        start_unit_index=start_index,
                        end_unit_index=end_index - 1,
                        text=text,
                    )
                )

        return tuple(
            windows
        )

def split_evidence_support_units(
    evidence_text: str,
) -> tuple[str, ...]:
    """
    Split one evidence block into sentence-like support units.

    Empty input is rejected because a grounding validator should
    never attempt to validate a claim against missing evidence.
    """

    normalized = " ".join(
        evidence_text.split()
    )

    if not normalized:
        raise ValueError(
            "Evidence text cannot be empty."
        )

    units = tuple(
        unit.strip()
        for unit in _SENTENCE_BOUNDARY_PATTERN.split(
            normalized
        )
        if unit.strip()
    )

    if not units:
        raise ValueError(
            "Evidence text did not produce any support units."
        )

    return units