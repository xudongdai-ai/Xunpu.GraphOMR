from __future__ import annotations

from xunpu_graph_omr.model import Staff

STEPS = ("C", "D", "E", "F", "G", "A", "B")
STEP_TO_INDEX = {step: index for index, step in enumerate(STEPS)}
TREBLE_BOTTOM_LINE = STEP_TO_INDEX["E"] + 4 * 7


def pitch_to_y(step: str, octave: int, staff: Staff) -> float:
    pitch_index = STEP_TO_INDEX[step.upper()] + octave * 7
    offset = pitch_index - TREBLE_BOTTOM_LINE
    return staff.bottom - offset * (staff.interline / 2.0)


def y_to_treble_pitch(y: float, staff: Staff) -> tuple[str, int]:
    offset = round((staff.bottom - y) / (staff.interline / 2.0))
    pitch_index = TREBLE_BOTTOM_LINE + offset
    step = STEPS[pitch_index % 7]
    octave = pitch_index // 7
    return step, octave
