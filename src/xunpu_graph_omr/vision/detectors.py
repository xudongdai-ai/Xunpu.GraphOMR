from __future__ import annotations

from xunpu_graph_omr.model import MeasureInfo, NoteEvent, Staff, StemEvent, SymbolDetection
from xunpu_graph_omr.vision.staff import detect_barlines


CORE_SYMBOL_TYPES = (
    "notehead_filled",
    "notehead_empty",
    "stem",
    "beam",
    "flag",
    "dot",
    "sharp",
    "flat",
    "natural",
    "double_sharp",
    "double_flat",
    "rest_whole",
    "rest_half",
    "rest_quarter",
    "rest_eighth",
    "clef_g",
    "clef_f",
    "clef_c",
    "barline",
    "time_number",
    "key_signature",
    "slur",
    "tie",
    "tuplet",
    "staff",
)


class ClassicalSymbolDetector:
    """Adapter that exposes the current OpenCV recognizers as symbol detections."""

    name = "classical-opencv"

    def detect(
        self,
        binary,
        staves: tuple[Staff, ...],
        notes: tuple[NoteEvent, ...],
        stems: tuple[StemEvent, ...],
        measures: tuple[MeasureInfo, ...],
    ) -> tuple[SymbolDetection, ...]:
        detections: list[SymbolDetection] = []
        for staff in staves:
            detections.append(
                SymbolDetection(
                    id=f"staff:{staff.index}",
                    type="staff",
                    bbox=(
                        float(staff.x_start),
                        float(staff.top),
                        float(staff.x_end - staff.x_start),
                        float(staff.bottom - staff.top),
                    ),
                    confidence=1.0,
                    staff_index=staff.index,
                    source_ref=f"staff:{staff.index}",
                )
            )
            for bar_index, x in enumerate(detect_barlines(binary, staff), start=1):
                detections.append(
                    SymbolDetection(
                        id=f"barline:{staff.index}:{bar_index}",
                        type="barline",
                        bbox=(float(x - 1), float(staff.top), 2.0, float(staff.bottom - staff.top)),
                        confidence=0.95,
                        staff_index=staff.index,
                        source_ref=f"barline:{staff.index}:{bar_index}",
                    )
                )

        for index, measure in enumerate(measures):
            detections.append(
                SymbolDetection(
                    id=f"measure:{measure.staff_index}:{measure.index}",
                    type="measure",
                    bbox=(
                        float(measure.x_start),
                        0.0,
                        float(measure.x_end - measure.x_start),
                        0.0,
                    ),
                    confidence=1.0,
                    staff_index=measure.staff_index,
                    measure_index=measure.index,
                    source_ref=f"measure:{index}",
                )
            )

        for index, note in enumerate(notes):
            bbox = note.bbox or _note_bbox(note, staves[note.staff_index])
            detections.append(
                SymbolDetection(
                    id=f"notehead:{index}",
                    type="notehead_empty" if note.head_type == "empty" else "notehead_filled",
                    bbox=bbox,
                    confidence=note.confidence,
                    staff_index=note.staff_index,
                    measure_index=note.measure,
                    source_ref=f"note:{index}",
                )
            )

        for index, stem in enumerate(stems):
            detections.append(
                SymbolDetection(
                    id=f"stem:{index}",
                    type="stem",
                    bbox=(
                        float(stem.x - stem.width / 2.0),
                        float(stem.y_top),
                        float(stem.width),
                        float(stem.y_bottom - stem.y_top),
                    ),
                    confidence=stem.confidence,
                    staff_index=stem.staff_index,
                    source_ref=f"stem:{index}",
                )
            )
        return tuple(detections)


def _note_bbox(note: NoteEvent, staff: Staff) -> tuple[float, float, float, float]:
    width = staff.interline * 1.2
    height = staff.interline * 0.9
    return (float(note.x - width / 2.0), float(note.y - height / 2.0), float(width), float(height))
