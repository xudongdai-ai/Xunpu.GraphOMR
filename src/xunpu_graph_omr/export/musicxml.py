from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

from xunpu_graph_omr.vision.image_io import ensure_parent
from xunpu_graph_omr.model import NoteEvent, RecognitionResult


def _text(parent: ET.Element, tag: str, value: str | int) -> ET.Element:
    child = ET.SubElement(parent, tag)
    child.text = str(value)
    return child


def _append_pitch(parent: ET.Element, note: NoteEvent) -> None:
    pitch = ET.SubElement(parent, "pitch")
    _text(pitch, "step", note.step)
    _text(pitch, "octave", note.octave)


def _append_note(parent: ET.Element, note: NoteEvent) -> None:
    note_el = ET.SubElement(parent, "note")
    _append_pitch(note_el, note)
    _text(note_el, "duration", note.duration)
    _text(note_el, "voice", "1")
    _text(note_el, "type", note.note_type)
    if note.stem_direction:
        ET.SubElement(note_el, "stem").text = note.stem_direction


def to_musicxml(result: RecognitionResult, title: str = "Xunpu GraphOMR Export") -> str:
    score = ET.Element("score-partwise", version="4.0")
    work = ET.SubElement(score, "work")
    _text(work, "work-title", title)

    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    _text(score_part, "part-name", "Music")

    part = ET.SubElement(score, "part", id="P1")
    notes_by_measure: dict[int, list[NoteEvent]] = defaultdict(list)
    for note in sorted(result.notes, key=lambda item: (item.measure, item.staff_index, item.x)):
        notes_by_measure[note.measure].append(note)

    if not notes_by_measure:
        notes_by_measure[1] = []

    for measure_number in sorted(notes_by_measure):
        measure = ET.SubElement(part, "measure", number=str(measure_number))
        if measure_number == 1:
            attributes = ET.SubElement(measure, "attributes")
            _text(attributes, "divisions", "1")
            key = ET.SubElement(attributes, "key")
            _text(key, "fifths", "0")
            time = ET.SubElement(attributes, "time")
            _text(time, "beats", "4")
            _text(time, "beat-type", "4")
            clef = ET.SubElement(attributes, "clef")
            _text(clef, "sign", "G")
            _text(clef, "line", "2")
        for note in notes_by_measure[measure_number]:
            _append_note(measure, note)

    rough = ET.tostring(score, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def write_musicxml(result: RecognitionResult, path: str | Path, title: str = "Xunpu GraphOMR Export") -> None:
    ensure_parent(path)
    Path(path).write_text(to_musicxml(result, title=title), encoding="utf-8")
