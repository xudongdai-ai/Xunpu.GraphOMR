"""
Dependency-free MusicXML reader.

Parses a `.musicxml` / `.xml` / `.mxl` file into a flat list of notes so the GUI
can display results from ANY engine (Xunpu pipeline, oemer, Audiveris) without
pulling in music21. Only the stdlib (ElementTree + zipfile) is used.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

_ALTER_SYM = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}


@dataclass(frozen=True)
class ParsedNote:
    step: str
    octave: int
    alter: int
    measure: int
    is_rest: bool
    note_type: str
    is_chord: bool = False

    @property
    def pitch_name(self) -> str:
        if self.is_rest:
            return "rest"
        return f"{self.step}{_ALTER_SYM.get(self.alter, '')}{self.octave}"


def _read_xml_bytes(path: Path) -> bytes:
    """Return raw MusicXML bytes, transparently unzipping .mxl containers."""
    if path.suffix.lower() != ".mxl":
        return path.read_bytes()
    with zipfile.ZipFile(path) as zf:
        # the container points at the real score; fall back to first non-meta xml
        rootfile = None
        try:
            container = ET.fromstring(zf.read("META-INF/container.xml"))
            el = container.find(".//{*}rootfile")
            if el is not None:
                rootfile = el.get("full-path")
        except KeyError:
            pass
        if rootfile is None:
            names = [n for n in zf.namelist()
                     if n.lower().endswith((".xml", ".musicxml")) and not n.startswith("META-INF")]
            if not names:
                raise ValueError(f"No score xml inside {path}")
            rootfile = names[0]
        return zf.read(rootfile)


def read_musicxml_bytes(path: str | Path) -> bytes:
    """
    返回未压缩的 MusicXML 字节，自动解开 .mxl 压缩包。

    导出时必须用它写出纯文本 .musicxml，外部软件(MuseScore 等)才能打开；
    若直接把 .mxl 复制成 .musicxml 后缀，得到的是 zip 内容，会打不开。
    """
    return _read_xml_bytes(Path(path))


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def extract_score_meta(path: str | Path) -> dict:
    """
    提取调号(fifths)与拍号,供简谱定调使用。取乐谱中首个出现的值。
    fifths: 五度圈数，0=C大调, 1=G, 2=D, -1=F, -2=Bb ...
    """
    root = ET.fromstring(_read_xml_bytes(Path(path)))
    fifths = 0
    beats, beat_type = 4, 4
    got_key = got_time = False
    for el in root.iter():
        ln = _localname(el.tag)
        if ln == "fifths" and not got_key:
            try:
                fifths = int((el.text or "0").strip())
                got_key = True
            except ValueError:
                pass
        elif ln == "beats" and not got_time:
            try:
                beats = int((el.text or "4").strip())
            except ValueError:
                pass
        elif ln == "beat-type" and not got_time:
            try:
                beat_type = int((el.text or "4").strip())
                got_time = True
            except ValueError:
                pass
    return {"fifths": fifths, "beats": beats, "beat_type": beat_type}


def parse_musicxml(path: str | Path) -> list[ParsedNote]:
    """Parse a MusicXML file into an ordered list of ParsedNote."""
    root = ET.fromstring(_read_xml_bytes(Path(path)))
    notes: list[ParsedNote] = []

    # partwise: <part><measure number=..><note>...   (most common export)
    for part in root.iter():
        if _localname(part.tag) != "part":
            continue
        measure_no = 0
        for measure in part:
            if _localname(measure.tag) != "measure":
                continue
            try:
                measure_no = int(measure.get("number", measure_no + 1))
            except ValueError:
                measure_no += 1
            for note in measure:
                if _localname(note.tag) != "note":
                    continue
                notes.append(_parse_note(note, measure_no))
    return notes


def _parse_note(note: ET.Element, measure_no: int) -> ParsedNote:
    is_rest = False
    is_chord = False
    step, octave, alter, note_type = "", 0, 0, "quarter"

    for child in note:
        tag = _localname(child.tag)
        if tag == "rest":
            is_rest = True
        elif tag == "chord":
            is_chord = True
        elif tag == "type":
            note_type = (child.text or "quarter").strip()
        elif tag == "pitch":
            for p in child:
                ptag = _localname(p.tag)
                if ptag == "step":
                    step = (p.text or "").strip()
                elif ptag == "octave":
                    octave = int((p.text or "0").strip() or 0)
                elif ptag == "alter":
                    alter = int(float((p.text or "0").strip() or 0))

    return ParsedNote(
        step=step, octave=octave, alter=alter, measure=measure_no,
        is_rest=is_rest, note_type=note_type, is_chord=is_chord,
    )


def measure_count(notes: list[ParsedNote]) -> int:
    return len({n.measure for n in notes}) if notes else 0
