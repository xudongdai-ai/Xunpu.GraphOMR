from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from xunpu_graph_omr.model import NoteEvent, Staff, StemEvent
from xunpu_graph_omr.vision.preprocess import horizontal_lines, vertical_lines
from xunpu_graph_omr.vision.staff import detect_barlines


def detect_stems(binary: np.ndarray, staves: tuple[Staff, ...]) -> tuple[StemEvent, ...]:
    stems: list[StemEvent] = []
    for staff in staves:
        interline = staff.interline
        y0 = max(0, int(staff.top - interline * 4))
        y1 = min(binary.shape[0], int(staff.bottom + interline * 4))
        x0 = max(0, int(staff.x_start - interline))
        x1 = min(binary.shape[1], int(staff.x_end + interline))
        crop = binary[y0:y1, x0:x1]

        vlines = vertical_lines(crop, min_height=max(12, int(interline * 2.1)))
        contours, _ = cv2.findContours(vlines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        barlines = detect_barlines(binary, staff)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < interline * 2.0 or h > interline * 5.5:
                continue
            if w < 1 or w > max(6, interline * 0.45):
                continue
            abs_x = x0 + x + (w / 2.0)
            if any(abs(abs_x - bar_x) <= interline * 0.45 for bar_x in barlines):
                continue
            abs_top = y0 + y
            abs_bottom = y0 + y + h
            ink_area = float(cv2.contourArea(contour))
            expected_area = max(1.0, h * max(1.0, w))
            confidence = max(0.0, min(1.0, 0.45 + (ink_area / expected_area)))
            stems.append(
                StemEvent(
                    x=float(abs_x),
                    y_top=float(abs_top),
                    y_bottom=float(abs_bottom),
                    staff_index=staff.index,
                    width=float(w),
                    confidence=float(confidence),
                )
            )
    return tuple(_dedupe_stems(stems))


def link_stems_to_notes(
    notes: tuple[NoteEvent, ...],
    stems: tuple[StemEvent, ...],
    staves: tuple[Staff, ...],
) -> tuple[NoteEvent, ...]:
    staff_by_index = {staff.index: staff for staff in staves}
    linked: list[NoteEvent] = []
    for note in notes:
        staff = staff_by_index[note.staff_index]
        interline = staff.interline
        candidates = [
            stem
            for stem in stems
            if stem.staff_index == note.staff_index
            and abs(stem.x - note.x) <= interline * 0.9
            and stem.y_top <= note.y + interline
            and stem.y_bottom >= note.y - interline
        ]
        if not candidates:
            if note.head_type == "empty":
                linked.append(replace(note, duration=4, note_type="whole", stem_direction=None, stem_x=None))
            else:
                linked.append(note)
            continue

        stem = min(candidates, key=lambda item: abs(item.x - note.x))
        direction = "up" if stem.y_top < note.y - interline * 0.8 else "down"
        duration = 2 if note.head_type == "empty" else note.duration
        note_type = "half" if note.head_type == "empty" else note.note_type
        linked.append(
            replace(
                note,
                duration=duration,
                note_type=note_type,
                stem_direction=direction,
                stem_x=stem.x,
                confidence=min(1.0, note.confidence + 0.10),
            )
        )
    return tuple(linked)


def _dedupe_stems(stems: list[StemEvent]) -> list[StemEvent]:
    kept: list[StemEvent] = []
    for stem in sorted(stems, key=lambda item: (item.staff_index, item.x, item.y_top)):
        duplicate = False
        for existing in kept:
            if stem.staff_index != existing.staff_index:
                continue
            if abs(stem.x - existing.x) <= 3 and abs(stem.y_top - existing.y_top) <= 4:
                duplicate = True
                break
        if not duplicate:
            kept.append(stem)
    return kept
