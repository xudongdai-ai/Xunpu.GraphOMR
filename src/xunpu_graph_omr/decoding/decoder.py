from __future__ import annotations

from collections import defaultdict

from xunpu_graph_omr.model import NoteEvent, RecognitionResult


def build_omr_document(result: RecognitionResult) -> dict:
    notes_by_measure: dict[tuple[int, int], list[NoteEvent]] = defaultdict(list)
    for note in sorted(result.notes, key=lambda item: (item.staff_index, item.measure, item.x, item.y)):
        notes_by_measure[(note.staff_index, note.measure)].append(note)

    system = {
        "index": 1,
        "staves": [
            {
                "index": staff.index + 1,
                "line_ys": list(staff.line_ys),
                "x_start": staff.x_start,
                "x_end": staff.x_end,
                "interline": staff.interline,
            }
            for staff in result.staves
        ],
        "measures": [],
    }

    for measure in result.measures:
        events = [
            {
                "type": "note",
                "pitch": note.pitch_name,
                "step": note.step,
                "octave": note.octave,
                "duration": float(note.duration),
                "note_type": note.note_type,
                "head_type": note.head_type,
                "onset": float(note.onset),
                "voice": note.voice,
                "staff": note.staff_index + 1,
                "bbox": list(note.bbox) if note.bbox else None,
                "confidence": note.confidence,
                "stem_direction": note.stem_direction,
            }
            for note in notes_by_measure.get((measure.staff_index, measure.index), [])
        ]
        system["measures"].append(
            {
                "number": measure.index,
                "staff": measure.staff_index + 1,
                "time_signature": measure.time_signature,
                "key_signature": measure.key_signature,
                "duration_total": float(measure.total_duration),
                "duration_expected": float(measure.expected_duration),
                "status": measure.status,
                "events": events,
            }
        )

    return {
        "source": result.image_path,
        "page": result.page_number,
        "systems": [system],
        "graph_summary": result.graph.to_dict() if result.graph else None,
        "steps": [step.__dict__ for step in result.steps],
        "issues": list(result.rhythm_issues),
    }


def build_analysis_report(result: RecognitionResult) -> dict:
    graph = result.graph
    return {
        "source": result.image_path,
        "page": result.page_number,
        "note_count": len(result.notes),
        "staff_count": len(result.staves),
        "measure_count": len(result.measures),
        "rhythm_issues": list(result.rhythm_issues),
        "graph": {
            "node_type_counts": graph.node_type_counts() if graph else {},
            "edge_type_counts": graph.edge_type_counts() if graph else {},
        },
        "gnn": result.graph_inference.__dict__ if result.graph_inference else None,
        "steps": [step.__dict__ for step in result.steps],
    }
