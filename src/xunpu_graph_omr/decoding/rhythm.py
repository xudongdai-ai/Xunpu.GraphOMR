from __future__ import annotations

from xunpu_graph_omr.model import MeasureInfo


def analyze_rhythm(measures: tuple[MeasureInfo, ...]) -> tuple[str, ...]:
    issues: list[str] = []
    for measure in measures:
        if measure.status in {"complete", "empty"}:
            continue
        issues.append(
            "staff "
            f"{measure.staff_index + 1} measure {measure.index} is {measure.status}: "
            f"{measure.total_duration}/{measure.expected_duration}"
        )
    return tuple(issues)
