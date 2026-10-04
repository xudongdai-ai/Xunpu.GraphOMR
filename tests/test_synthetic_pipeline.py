from __future__ import annotations

from xunpu_graph_omr.decoding.decoder import build_omr_document
from xunpu_graph_omr.evaluation.evaluation import compare_truth_to_result
from xunpu_graph_omr.export.musicxml import write_musicxml
from xunpu_graph_omr.pipeline import recognize_path
from xunpu_graph_omr.evaluation.synthetic import draw_duration_score, draw_simple_score


def test_synthetic_score_round_trip(tmp_path):
    image_path = tmp_path / "score.png"
    truth_path = tmp_path / "truth.json"
    xml_path = tmp_path / "score.musicxml"

    draw_simple_score(image_path, truth_path=truth_path)
    result = recognize_path(image_path)
    write_musicxml(result, xml_path)

    metrics = compare_truth_to_result(truth_path, result)
    assert metrics.expected_count == 8
    assert metrics.actual_count == 8
    assert metrics.sequence_accuracy == 1.0
    assert result.scale is not None
    assert round(result.scale.interline) == 16
    assert len(result.stems) == 8
    assert [measure.status for measure in result.measures] == ["complete", "complete"]
    assert result.rhythm_issues == ()
    assert [step.name for step in result.steps] == [
        "LOAD",
        "PREPROCESS",
        "BINARY",
        "SCALE",
        "GRID",
        "HEADS",
        "STEMS",
        "MEASURES",
        "RHYTHMS",
        "DETECT",
        "GRAPH",
        "GNN",
        "DECODE",
    ]
    assert result.graph is not None
    assert result.graph.node_type_counts()["notehead_filled"] == 8
    assert result.graph.node_type_counts()["stem"] == 8
    assert result.graph.edge_type_counts()["notehead_to_stem"] == 8
    omr_document = build_omr_document(result)
    assert omr_document["page"] == 1
    assert sum(len(measure["events"]) for measure in omr_document["systems"][0]["measures"]) == 8
    assert xml_path.exists()


def test_duration_score_round_trip(tmp_path):
    image_path = tmp_path / "durations.png"
    truth_path = tmp_path / "durations_truth.json"

    draw_duration_score(image_path, truth_path=truth_path)
    result = recognize_path(image_path)

    notes = sorted(result.notes, key=lambda item: (item.measure, item.x))
    assert [note.pitch_name for note in notes] == ["F4", "A4", "C5"]
    assert [note.note_type for note in notes] == ["whole", "half", "half"]
    assert [note.duration for note in notes] == [4, 2, 2]
    assert [note.head_type for note in notes] == ["empty", "empty", "empty"]
    assert [measure.status for measure in result.measures] == ["complete", "complete"]
    assert result.graph is not None
    assert result.graph.node_type_counts()["notehead_empty"] == 3
