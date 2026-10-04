from __future__ import annotations

from xunpu_graph_omr.model import ScaleInfo, StepReport


AUDIVERIS_STEP_ORDER = (
    "LOAD",
    "BINARY",
    "SCALE",
    "GRID",
    "HEADERS",
    "STEM_SEEDS",
    "BEAMS",
    "LEDGERS",
    "HEADS",
    "STEMS",
    "REDUCTION",
    "CUE_BEAMS",
    "TEXTS",
    "MEASURES",
    "CHORDS",
    "CURVES",
    "SYMBOLS",
    "LINKS",
    "RHYTHMS",
    "PAGE",
)


IMPLEMENTED_STEP_ORDER = (
    "LOAD",
    "PREPROCESS",
    "BINARY",
    "SCALE",
    "GRID",
    "HEADS",
    "STEMS",
    "DETECT",
    "GRAPH",
    "GNN",
    "MEASURES",
    "RHYTHMS",
    "DECODE",
)


def report(name: str, detail: str, status: str = "ok") -> StepReport:
    return StepReport(name=name, status=status, detail=detail)


def scale_detail(scale: ScaleInfo) -> str:
    return (
        f"line={scale.staff_line_thickness:.2f}, "
        f"interline={scale.interline:.2f}, "
        f"beam={scale.beam_thickness:.2f}"
    )
