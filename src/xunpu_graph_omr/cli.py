from __future__ import annotations

import argparse
from pathlib import Path

from xunpu_graph_omr.decoding.decoder import build_analysis_report, build_omr_document
from xunpu_graph_omr.vision.image_io import write_json
from xunpu_graph_omr.export.musicxml import write_musicxml
from xunpu_graph_omr.pipeline import recognize_path, write_debug_overlay
from xunpu_graph_omr.evaluation.synthetic import draw_duration_score, draw_simple_score


def _recognize(args: argparse.Namespace) -> int:
    result = recognize_path(args.input)
    write_musicxml(result, args.output, title=Path(args.input).stem)
    if args.json:
        write_json(args.json, result.to_dict())
    if args.omr_json:
        write_json(args.omr_json, build_omr_document(result))
    if args.report:
        write_json(args.report, build_analysis_report(result))
    if args.midi:
        from xunpu_graph_omr.export.midi import write_midi

        write_midi(result, args.midi)
    if args.debug_dir:
        debug_path = Path(args.debug_dir) / f"{Path(args.input).stem}.overlay.png"
        write_debug_overlay(result, debug_path)
    print(f"recognized {len(result.notes)} notes -> {args.output}")
    return 0


def _synthesize(args: argparse.Namespace) -> int:
    if args.kind == "durations":
        truth = draw_duration_score(args.output, truth_path=args.truth)
    else:
        truth = draw_simple_score(args.output, truth_path=args.truth)
    print(f"wrote synthetic score with {len(truth['notes'])} notes -> {args.output}")
    if args.truth:
        print(f"wrote truth -> {args.truth}")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from xunpu_graph_omr.evaluation.benchmark import benchmark_image

    payload = benchmark_image(
        args.input,
        args.truth,
        audiveris=args.audiveris,
        audiveris_output_dir=args.audiveris_output_dir,
    )
    if args.json:
        write_json(args.json, payload)
    print(payload)
    return 0


def _portfolio(args: argparse.Namespace) -> int:
    from xunpu_graph_omr.evaluation.portfolio import build_portfolio

    payload = build_portfolio(args.output)
    print(payload["summary"])
    print(f"Demo: {Path(args.output) / 'index.html'}")
    return 0 if all(c["exact"] and c["xml_parseable"] for c in payload["cases"]
                    if c["variant"] == "clean") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xunpu-graph-omr")
    sub = parser.add_subparsers(dest="command", required=True)

    recognize = sub.add_parser("recognize", help="Recognize a score image and export MusicXML")
    recognize.add_argument("input")
    recognize.add_argument("-o", "--output", required=True)
    recognize.add_argument("--json", help="Write raw recognition JSON with detections and graph")
    recognize.add_argument("--omr-json", help="Write structured OMR JSON for downstream analysis")
    recognize.add_argument("--report", help="Write concise analysis/debug report JSON")
    recognize.add_argument("--midi", help="Write MIDI output")
    recognize.add_argument("--debug-dir")
    recognize.set_defaults(func=_recognize)

    synth = sub.add_parser("synthesize", help="Create a synthetic score fixture")
    synth.add_argument("output")
    synth.add_argument("--truth")
    synth.add_argument("--kind", choices=("simple", "durations"), default="simple")
    synth.set_defaults(func=_synthesize)

    bench = sub.add_parser("benchmark", help="Compare Xunpu GraphOMR and optional Audiveris against truth JSON")
    bench.add_argument("input")
    bench.add_argument("--truth", required=True)
    bench.add_argument("--audiveris")
    bench.add_argument("--audiveris-output-dir")
    bench.add_argument("--json")
    bench.set_defaults(func=_benchmark)
    portfolio = sub.add_parser("portfolio", help="Build reproducible stress evaluation and static demo")
    portfolio.add_argument("--output", default="out/demo")
    portfolio.set_defaults(func=_portfolio)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
