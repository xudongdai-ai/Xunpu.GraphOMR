"""
Dependency-free MIDI (type-0) writer.

Used by both the GUI and CLI so the public package has no heavyweight MIDI dependency.
"""
from __future__ import annotations

import struct
from pathlib import Path

from xunpu_graph_omr.vision.image_io import ensure_parent
from xunpu_graph_omr.model import RecognitionResult

_TICKS_PER_QUARTER = 480
_STEP_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_TYPE_QUARTERS = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "16th": 0.25,
    "sixteenth": 0.25,
}


def _midi_number(step: str, octave: int) -> int:
    semitone = _STEP_SEMITONE.get(step.upper(), 0)
    return max(0, min(127, (octave + 1) * 12 + semitone))


def _var_len(value: int) -> bytes:
    buffer = value & 0x7F
    value >>= 7
    out = bytearray()
    while value:
        buffer <<= 8
        buffer |= (value & 0x7F) | 0x80
        value >>= 7
    while True:
        out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(out)


def write_midi(result: RecognitionResult, path: str | Path, *, tempo_bpm: int = 120) -> None:
    """Write a single-track monophonic MIDI file from a RecognitionResult."""
    notes = sorted(result.notes, key=lambda n: (n.measure, n.staff_index, n.x))
    write_midi_notes(notes, path, tempo_bpm=tempo_bpm)


def write_midi_notes(notes, path: str | Path, *, tempo_bpm: int = 120) -> None:
    """
    Write MIDI from any iterable of note-like objects exposing ``step``,
    ``octave`` and ``note_type``. Objects with a truthy ``is_rest`` are skipped.
    Lets external-engine results (ParsedNote) export MIDI without a full pipeline.
    """
    notes = [n for n in notes if not getattr(n, "is_rest", False)]

    track = bytearray()
    # tempo meta event
    micros_per_quarter = int(60_000_000 / max(1, tempo_bpm))
    track += _var_len(0) + b"\xff\x51\x03" + struct.pack(">I", micros_per_quarter)[1:]

    for event in notes:
        pitch = _midi_number(event.step, event.octave)
        quarters = _TYPE_QUARTERS.get(event.note_type, 1.0)
        dur_ticks = int(quarters * _TICKS_PER_QUARTER)
        track += _var_len(0) + bytes((0x90, pitch, 80))            # note on
        track += _var_len(dur_ticks) + bytes((0x80, pitch, 0))     # note off

    track += _var_len(0) + b"\xff\x2f\x00"                          # end of track

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, _TICKS_PER_QUARTER)
    track_chunk = b"MTrk" + struct.pack(">I", len(track)) + bytes(track)

    ensure_parent(path)
    Path(path).write_bytes(header + track_chunk)
