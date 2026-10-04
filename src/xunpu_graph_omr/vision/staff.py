from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from xunpu_graph_omr.model import Staff
from xunpu_graph_omr.vision.preprocess import horizontal_lines, vertical_lines


def _runs(values: Iterable[int]) -> list[tuple[int, int]]:
    sorted_values = sorted(set(int(v) for v in values))
    if not sorted_values:
        return []

    runs: list[tuple[int, int]] = []
    start = prev = sorted_values[0]
    for value in sorted_values[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _weighted_centers(row_counts: np.ndarray, runs: list[tuple[int, int]]) -> list[float]:
    centers: list[float] = []
    for start, end in runs:
        ys = np.arange(start, end + 1)
        weights = row_counts[start : end + 1].astype(float)
        if weights.sum() <= 0:
            centers.append((start + end) / 2.0)
        else:
            centers.append(float(np.average(ys, weights=weights)))
    return centers


def _estimate_interline(line_ys: list[float]) -> float:
    diffs = np.diff(sorted(line_ys))
    diffs = diffs[diffs > 3]
    if len(diffs) == 0:
        raise ValueError("Could not estimate staff interline")
    cutoff = np.percentile(diffs, 70)
    staff_diffs = diffs[diffs <= cutoff]
    if len(staff_diffs) == 0:
        staff_diffs = diffs
    return float(np.median(staff_diffs))


def detect_staves(binary: np.ndarray) -> tuple[Staff, ...]:
    lines = horizontal_lines(binary)
    height, width = binary.shape
    row_counts = np.count_nonzero(lines, axis=1)
    if row_counts.max(initial=0) == 0:
        raise ValueError("No staff-like horizontal lines found")

    threshold = max(int(width * 0.20), int(row_counts.max() * 0.35))
    rows = np.where(row_counts >= threshold)[0]
    line_runs = _runs(rows)
    line_ys = _weighted_centers(row_counts, line_runs)
    if len(line_ys) < 5:
        raise ValueError(f"Found only {len(line_ys)} staff lines")

    interline = _estimate_interline(line_ys)
    grouped: list[list[float]] = []
    current: list[float] = []
    for y in sorted(line_ys):
        if not current or y - current[-1] <= interline * 1.7:
            current.append(y)
        else:
            grouped.append(current)
            current = [y]
    if current:
        grouped.append(current)

    staves: list[Staff] = []
    for group in grouped:
        if len(group) < 5:
            continue
        for start in range(0, len(group) - 4, 5):
            candidate = group[start : start + 5]
            diffs = np.diff(candidate)
            if np.std(diffs) > max(2.5, interline * 0.18):
                continue
            y0 = max(0, int(candidate[0] - 2))
            y1 = min(height, int(candidate[-1] + 3))
            col_counts = np.count_nonzero(lines[y0:y1, :], axis=0)
            cols = np.where(col_counts > 0)[0]
            if len(cols) == 0:
                x_start, x_end = 0, width - 1
            else:
                x_start, x_end = int(cols.min()), int(cols.max())
            staves.append(
                Staff(
                    index=len(staves),
                    line_ys=tuple(float(v) for v in candidate),  # type: ignore[arg-type]
                    x_start=x_start,
                    x_end=x_end,
                )
            )

    if not staves:
        raise ValueError("Could not group detected lines into five-line staves")
    return tuple(staves)


def detect_barlines(binary: np.ndarray, staff: Staff) -> tuple[int, ...]:
    interline = staff.interline
    y0 = max(0, int(staff.top - interline * 0.6))
    y1 = min(binary.shape[0], int(staff.bottom + interline * 0.6))
    x0 = max(0, staff.x_start)
    x1 = min(binary.shape[1], staff.x_end + 1)
    crop = binary[y0:y1, x0:x1]
    min_height = max(8, int((staff.bottom - staff.top) * 0.75))
    vlines = vertical_lines(crop, min_height)
    col_counts = np.count_nonzero(vlines, axis=0)
    staff_height = max(1.0, staff.bottom - staff.top)
    threshold = max(5, int(staff_height * 0.82))
    cols = np.where(col_counts >= threshold)[0]
    runs = _runs(cols)
    barlines: list[int] = []
    for start, end in runs:
        x = x0 + int(round((start + end) / 2))
        if x <= staff.x_start + interline * 1.5:
            continue
        line_hits = 0
        lx0 = max(0, start - 1)
        lx1 = min(vlines.shape[1], end + 2)
        for line_y in staff.line_ys:
            ly0 = max(0, int(round(line_y)) - y0 - 2)
            ly1 = min(vlines.shape[0], int(round(line_y)) - y0 + 3)
            if np.count_nonzero(vlines[ly0:ly1, lx0:lx1]) > 0:
                line_hits += 1
        if line_hits < 5:
            continue
        barlines.append(x)
    return tuple(barlines)
