from __future__ import annotations

import numpy as np

from xunpu_graph_omr.model import ScaleInfo, Staff
from xunpu_graph_omr.vision.preprocess import horizontal_lines


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    if len(values) == 0:
        return []
    values = np.array(sorted(set(int(v) for v in values)))
    starts = [int(values[0])]
    ends: list[int] = []
    for previous, current in zip(values[:-1], values[1:], strict=False):
        if int(current) != int(previous) + 1:
            ends.append(int(previous))
            starts.append(int(current))
    ends.append(int(values[-1]))
    return list(zip(starts, ends, strict=True))


def estimate_scale(binary: np.ndarray, staves: tuple[Staff, ...]) -> ScaleInfo:
    hlines = horizontal_lines(binary)
    row_counts = np.count_nonzero(hlines, axis=1)
    threshold = max(1, int(row_counts.max(initial=0) * 0.35))
    line_runs = _runs(np.where(row_counts >= threshold)[0])
    thicknesses = [end - start + 1 for start, end in line_runs]
    staff_line_thickness = float(np.median(thicknesses)) if thicknesses else 1.0
    interlines = [staff.interline for staff in staves]
    interline = float(np.median(interlines)) if interlines else 0.0

    # Audiveris treats beams as their own scale item. Until beam detection is
    # implemented, this estimate gives downstream filters a stable size prior.
    beam_thickness = max(staff_line_thickness * 1.8, interline * 0.22)
    return ScaleInfo(
        staff_line_thickness=staff_line_thickness,
        interline=interline,
        beam_thickness=float(beam_thickness),
    )
