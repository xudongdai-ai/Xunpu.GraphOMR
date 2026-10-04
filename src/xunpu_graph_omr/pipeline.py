from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from xunpu_graph_omr.vision.detectors import ClassicalSymbolDetector
from xunpu_graph_omr.vision.document import DocumentPage, load_first_page_gray
from xunpu_graph_omr.graph.gnn import RuleBasedGraphInferenceEngine
from xunpu_graph_omr.graph.graph import build_heterograph
from xunpu_graph_omr.vision.image_io import ensure_parent
from xunpu_graph_omr.decoding.measures import assign_onsets, build_measures
from xunpu_graph_omr.model import RecognitionResult
from xunpu_graph_omr.vision.preprocess import binarize, preprocess_gray
from xunpu_graph_omr.decoding.rhythm import analyze_rhythm
from xunpu_graph_omr.vision.scale import estimate_scale
from xunpu_graph_omr.vision.staff import detect_staves
from xunpu_graph_omr.vision.stems import detect_stems, link_stems_to_notes
from xunpu_graph_omr.steps import report, scale_detail
from xunpu_graph_omr.vision.symbols import detect_noteheads


def recognize_path(path: str | Path) -> RecognitionResult:
    page = load_first_page_gray(path)
    return recognize_page(page)


def recognize_page(page: DocumentPage) -> RecognitionResult:
    gray = page.gray
    steps = [report("LOAD", f"page={page.page_number}, {gray.shape[1]}x{gray.shape[0]} grayscale image")]
    gray, preprocess_stats = preprocess_gray(gray)
    steps.append(report("PREPROCESS", f"denoise=median3, deskew={preprocess_stats['skew_degrees']:.2f}deg"))
    binary = binarize(gray)
    foreground = float(np.count_nonzero(binary)) / float(binary.size)
    steps.append(report("BINARY", f"foreground={foreground:.4f}"))
    staves = detect_staves(binary)
    scale = estimate_scale(binary, staves)
    steps.append(report("SCALE", scale_detail(scale)))
    steps.append(report("GRID", f"staves={len(staves)}"))
    stems = detect_stems(binary, staves)
    notes = detect_noteheads(binary, staves)
    steps.append(report("HEADS", f"noteheads={len(notes)}"))
    notes = link_stems_to_notes(notes, stems, staves)
    linked = sum(1 for note in notes if note.stem_x is not None)
    steps.append(report("STEMS", f"stems={len(stems)}, linked_notes={linked}"))
    measures = build_measures(binary, staves, notes)
    notes = assign_onsets(notes, measures)
    complete = sum(1 for measure in measures if measure.status == "complete")
    steps.append(report("MEASURES", f"measures={len(measures)}, complete={complete}"))
    rhythm_issues = analyze_rhythm(measures)
    steps.append(report("RHYTHMS", f"issues={len(rhythm_issues)}", status="warn" if rhythm_issues else "ok"))
    height, width = gray.shape
    detector = ClassicalSymbolDetector()
    symbols = detector.detect(binary, staves, notes, stems, measures)
    steps.append(report("DETECT", f"engine={detector.name}, symbols={len(symbols)}"))
    graph = build_heterograph(symbols, staves, width, height)
    steps.append(report("GRAPH", f"nodes={len(graph.nodes)}, edges={len(graph.edges)}"))
    graph_inference = RuleBasedGraphInferenceEngine().infer(graph)
    steps.append(report("GNN", f"engine={graph_inference.engine}, notes={len(graph_inference.notes)}"))
    steps.append(report("DECODE", f"notes={len(notes)}, measures={len(measures)}, issues={len(rhythm_issues)}"))
    return RecognitionResult(
        image_path=page.source_path,
        page_number=page.page_number,
        width=width,
        height=height,
        staves=staves,
        notes=notes,
        scale=scale,
        stems=stems,
        measures=measures,
        symbols=symbols,
        graph=graph,
        graph_inference=graph_inference,
        rhythm_issues=rhythm_issues,
        steps=tuple(steps),
    )


def write_debug_overlay(result: RecognitionResult, output_path: str | Path) -> None:
    gray, _ = preprocess_gray(load_first_page_gray(result.image_path).gray)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for staff in result.staves:
        for y in staff.line_ys:
            cv2.line(canvas, (staff.x_start, int(round(y))), (staff.x_end, int(round(y))), (0, 180, 0), 1)

    for measure in result.measures:
        y0 = int(round(result.staves[measure.staff_index].top))
        y1 = int(round(result.staves[measure.staff_index].bottom))
        x0 = int(round(measure.x_start))
        x1 = int(round(measure.x_end))
        cv2.rectangle(canvas, (x0, y0 - 10), (x1, y1 + 10), (220, 180, 0), 1)

    for stem in result.stems:
        cv2.line(
            canvas,
            (int(round(stem.x)), int(round(stem.y_top))),
            (int(round(stem.x)), int(round(stem.y_bottom))),
            (255, 0, 180),
            1,
        )

    for note in result.notes:
        center = (int(round(note.x)), int(round(note.y)))
        cv2.circle(canvas, center, 8, (0, 0, 255), 2)
        cv2.putText(
            canvas,
            note.pitch_name,
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 220),
            1,
            cv2.LINE_AA,
        )

    ensure_parent(output_path)
    cv2.imwrite(str(output_path), canvas)
