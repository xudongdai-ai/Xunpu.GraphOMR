from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

from xunpu_graph_omr.audiveris import run_audiveris_export
from xunpu_graph_omr.evaluation.evaluation import compare_pitch_sequences, compare_truth_to_result, note_events_from_musicxml, truth_pitch_names
from xunpu_graph_omr.pipeline import recognize_path


def benchmark_image(
    image_path: str | Path,
    truth_path: str | Path,
    audiveris: str | Path | None = None,
    audiveris_output_dir: str | Path | None = None,
) -> dict:
    result = recognize_path(image_path)
    own_metrics = compare_truth_to_result(truth_path, result).to_dict()
    payload = {
        "image": str(image_path),
        "truth": str(truth_path),
        "xunpu_graph_omr": own_metrics,
    }

    if audiveris:
        output_context = (
            nullcontext(Path(audiveris_output_dir))
            if audiveris_output_dir
            else TemporaryDirectory(prefix="xunpu-graph-omr-audiveris-")
        )
        with output_context as output_dir:
            try:
                exported = run_audiveris_export(image_path, output_dir, executable=audiveris)
                audiveris_pitches = note_events_from_musicxml(exported)
                metrics = compare_pitch_sequences(
                    truth_pitch_names(truth_path),
                    audiveris_pitches,
                ).to_dict()
                if audiveris_output_dir:
                    metrics["export"] = str(exported)
                payload["audiveris"] = metrics
            except Exception as exc:
                payload["audiveris"] = {"error": str(exc)}
    return payload
