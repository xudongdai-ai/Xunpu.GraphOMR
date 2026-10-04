from __future__ import annotations

import cv2
import numpy as np

from xunpu_graph_omr.model import NoteEvent, Staff
from xunpu_graph_omr.vision.pitch import y_to_treble_pitch
from xunpu_graph_omr.vision.preprocess import horizontal_lines, vertical_lines
from xunpu_graph_omr.vision.staff import detect_barlines


def _line_removed_symbol_mask(binary: np.ndarray, staff: Staff) -> np.ndarray:
    interline = staff.interline
    y0 = max(0, int(staff.top - interline * 4))
    y1 = min(binary.shape[0], int(staff.bottom + interline * 4))
    x0 = max(0, int(staff.x_start - interline))
    x1 = min(binary.shape[1], int(staff.x_end + interline))
    crop = binary[y0:y1, x0:x1]

    hlines = horizontal_lines(crop, min_width=max(20, int((x1 - x0) * 0.18)))
    vlines = vertical_lines(crop, min_height=max(8, int(interline * 2.2)))
    mask = cv2.subtract(crop, hlines)
    mask = cv2.subtract(mask, vlines)
    close_height = max(3, int(round(interline * 0.25)))
    if close_height % 2 == 0:
        close_height += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, close_height))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return mask


def _candidate_confidence(area: float, bbox_area: float, interline: float) -> float:
    fill = area / max(1.0, bbox_area)
    size_score = min(1.0, area / max(1.0, interline * interline * 0.45))
    return float(max(0.0, min(1.0, (fill * 0.55) + (size_score * 0.45))))


def _head_type(mask: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    crop = mask[y : y + h, x : x + w]
    ink_density = float(np.count_nonzero(crop)) / max(1.0, float(w * h))
    return "empty" if ink_density < 0.48 else "filled"


def _measure_for_x(x: float, barlines: tuple[int, ...]) -> int:
    return 1 + sum(1 for bar_x in barlines if bar_x < x)


def detect_noteheads(binary: np.ndarray, staves: tuple[Staff, ...]) -> tuple[NoteEvent, ...]:
    notes: list[NoteEvent] = []
    for staff in staves:
        interline = staff.interline
        y0 = max(0, int(staff.top - interline * 4))
        x0 = max(0, int(staff.x_start - interline))
        mask = _line_removed_symbol_mask(binary, staff)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        barlines = detect_barlines(binary, staff)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < interline * interline * 0.10 or area > interline * interline * 2.2:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < interline * 0.45 or w > interline * 1.9:
                continue
            if h < interline * 0.35 or h > interline * 1.55:
                continue
            aspect = w / max(1.0, h)
            if aspect < 0.7 or aspect > 2.4:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                cx = x + w / 2.0
                cy = y + h / 2.0
            else:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
            abs_x = x0 + cx
            abs_y = y0 + cy
            if abs_y < staff.top - interline * 3.2 or abs_y > staff.bottom + interline * 3.2:
                continue

            step, octave = y_to_treble_pitch(abs_y, staff)
            confidence = _candidate_confidence(area, w * h, interline)
            head_type = _head_type(mask, x, y, w, h)
            notes.append(
                NoteEvent(
                    x=float(abs_x),
                    y=float(abs_y),
                    staff_index=staff.index,
                    measure=_measure_for_x(abs_x, barlines),
                    step=step,
                    octave=octave,
                    duration=2 if head_type == "empty" else 1,
                    note_type="half" if head_type == "empty" else "quarter",
                    head_type=head_type,
                    bbox=(float(x0 + x), float(y0 + y), float(w), float(h)),
                    confidence=confidence,
                )
            )

    return tuple(_dedupe_notes(notes))


def _dedupe_notes(notes: list[NoteEvent]) -> list[NoteEvent]:
    sorted_notes = sorted(notes, key=lambda n: (n.staff_index, n.x, n.y, -n.confidence))
    kept: list[NoteEvent] = []
    for note in sorted_notes:
        duplicate = False
        for existing in kept:
            if note.staff_index != existing.staff_index:
                continue
            if abs(note.x - existing.x) <= 5 and abs(note.y - existing.y) <= 5:
                duplicate = True
                break
        if not duplicate:
            kept.append(note)
    return kept
