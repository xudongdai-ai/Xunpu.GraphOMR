from __future__ import annotations

from dataclasses import replace

import numpy as np

from xunpu_graph_omr.model import MeasureInfo, NoteEvent, Staff
from xunpu_graph_omr.vision.staff import detect_barlines


def build_measures(
    binary: np.ndarray,
    staves: tuple[Staff, ...],
    notes: tuple[NoteEvent, ...],
    expected_duration: int = 4,
) -> tuple[MeasureInfo, ...]:
    measures: list[MeasureInfo] = []
    for staff in staves:
        barlines = detect_barlines(binary, staff)
        boundaries = _distinct_boundaries((staff.x_start, *barlines, staff.x_end), min_gap=staff.interline)
        if len(boundaries) < 2:
            boundaries = (float(staff.x_start), float(staff.x_end))

        staff_notes = [note for note in notes if note.staff_index == staff.index]
        for index, (x_start, x_end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=False), start=1):
            if x_end - x_start < staff.interline * 1.5:
                continue
            contained = [note for note in staff_notes if x_start <= note.x < x_end]
            measures.append(
                MeasureInfo(
                    staff_index=staff.index,
                    index=index,
                    x_start=float(x_start),
                    x_end=float(x_end),
                    note_count=len(contained),
                    total_duration=sum(note.duration for note in contained),
                    expected_duration=expected_duration,
                )
            )
    return tuple(measures)


def assign_onsets(notes: tuple[NoteEvent, ...], measures: tuple[MeasureInfo, ...]) -> tuple[NoteEvent, ...]:
    measure_lookup = {(measure.staff_index, measure.index): measure for measure in measures}
    grouped: dict[tuple[int, int], list[NoteEvent]] = {}
    for note in notes:
        grouped.setdefault((note.staff_index, note.measure), []).append(note)

    updated: list[NoteEvent] = []
    for key, measure_notes in grouped.items():
        if key not in measure_lookup:
            updated.extend(measure_notes)
            continue
        onset = 0.0
        for note in sorted(measure_notes, key=lambda item: (item.x, item.y)):
            updated.append(replace(note, onset=onset))
            onset += float(note.duration)
    return tuple(sorted(updated, key=lambda item: (item.staff_index, item.measure, item.x, item.y)))


def _distinct_boundaries(values: tuple[float, ...], min_gap: float) -> tuple[float, ...]:
    sorted_values = sorted(float(value) for value in values)
    kept: list[float] = []
    for value in sorted_values:
        if not kept or abs(value - kept[-1]) >= min_gap:
            kept.append(value)
        else:
            kept[-1] = (kept[-1] + value) / 2.0
    return tuple(kept)
