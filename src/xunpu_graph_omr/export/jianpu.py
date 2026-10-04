"""
五线谱(MusicXML 音符)→ 简谱(数字谱)转换。首调(movable do)。
"""
from __future__ import annotations

import math

from xunpu_graph_omr.export.musicxml_parse import ParsedNote

_STEP_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SEMI_TO_DEGREE = {0: "1", 2: "2", 4: "3", 5: "4", 7: "5", 9: "6", 11: "7"}
_SEMI_TO_ALTERED = {1: "#1", 3: "#2", 6: "#4", 8: "#5", 10: "#6"}
_DUR_SUFFIX = {"whole": " - - -", "half": " -", "quarter": "", "eighth": "_",
               "16th": "__", "sixteenth": "__", "32nd": "___"}


def _pitch_class(step: str, alter: int) -> int:
    return (_STEP_SEMITONE.get(step.upper(), 0) + alter) % 12


def tonic_pitch_class(fifths: int) -> int:
    return (fifths * 7) % 12


def _base_do_midi(fifths: int) -> int:
    midi = 60 + tonic_pitch_class(fifths)
    while midi > 65:
        midi -= 12
    while midi < 54:
        midi += 12
    return midi


def note_to_jianpu(note: ParsedNote, fifths: int) -> str:
    if note.is_rest:
        return "0" + _DUR_SUFFIX.get(note.note_type, "")
    pc = _pitch_class(note.step, note.alter)
    midi = (note.octave + 1) * 12 + pc
    base = _base_do_midi(fifths)
    delta = midi - base
    group = math.floor(delta / 12.0)
    degree = delta % 12
    sym = _SEMI_TO_DEGREE.get(degree) or _SEMI_TO_ALTERED.get(degree, "?")
    if group > 0:
        sym += "'" * group
    elif group < 0:
        sym += "," * (-group)
    return sym + _DUR_SUFFIX.get(note.note_type, "")


def to_jianpu(notes: list[ParsedNote], fifths: int) -> str:
    by_measure: dict[int, list[str]] = {}
    for n in notes:
        by_measure.setdefault(n.measure, []).append(note_to_jianpu(n, fifths))
    measures = [" ".join(by_measure[m]) for m in sorted(by_measure)]
    return "  |  ".join(measures)


_KEY_NAME = {0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#",
             -1: "F", -2: "Bb", -3: "Eb", -4: "Ab", -5: "Db", -6: "Gb"}


def key_name(fifths: int) -> str:
    return _KEY_NAME.get(fifths, "C")


def jianpu_from_result(notes: list[ParsedNote], musicxml_path) -> tuple[str, str]:
    """便捷封装,给 GUI 用。返回 (简谱文本, "1=调名")。"""
    from xunpu_graph_omr.export.musicxml_parse import extract_score_meta
    try:
        fifths = extract_score_meta(musicxml_path)["fifths"]
    except Exception:  # noqa: BLE001
        fifths = 0
    return to_jianpu(notes, fifths), "1=" + key_name(fifths)
