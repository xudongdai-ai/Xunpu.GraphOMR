from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xunpu_graph_omr.vision.image_io import read_json
from xunpu_graph_omr.model import NoteEvent, RecognitionResult


@dataclass(frozen=True)
class ScoreMetrics:
    expected_count: int
    actual_count: int
    pitch_matches: int
    sequence_accuracy: float

    def to_dict(self) -> dict:
        return {
            "expected_count": self.expected_count,
            "actual_count": self.actual_count,
            "pitch_matches": self.pitch_matches,
            "sequence_accuracy": self.sequence_accuracy,
        }


def truth_pitch_names(path: str | Path) -> list[str]:
    truth = read_json(path)
    return [note["pitch_name"] for note in truth.get("notes", [])]


def result_pitch_names(result: RecognitionResult) -> list[str]:
    return [note.pitch_name for note in sorted(result.notes, key=lambda n: (n.measure, n.staff_index, n.x))]


def note_events_from_musicxml(path: str | Path) -> list[str]:
    # Use the repository's dependency-free MusicXML reader for baseline evaluation.
    from xunpu_graph_omr.export.musicxml_parse import parse_musicxml

    return [note.pitch_name for note in parse_musicxml(path) if not note.is_rest]


def compare_pitch_sequences(expected: list[str], actual: list[str]) -> ScoreMetrics:
    matches = sum(1 for exp, got in zip(expected, actual, strict=False) if exp == got)
    denominator = max(len(expected), len(actual), 1)
    return ScoreMetrics(
        expected_count=len(expected),
        actual_count=len(actual),
        pitch_matches=matches,
        sequence_accuracy=matches / denominator,
    )


def compare_truth_to_result(truth_path: str | Path, result: RecognitionResult) -> ScoreMetrics:
    return compare_pitch_sequences(truth_pitch_names(truth_path), result_pitch_names(result))
