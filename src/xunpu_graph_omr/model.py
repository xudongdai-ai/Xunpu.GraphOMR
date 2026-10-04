from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class ScaleInfo:
    staff_line_thickness: float
    interline: float
    beam_thickness: float


@dataclass(frozen=True)
class StepReport:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class SymbolDetection:
    id: str
    type: str
    bbox: BBox
    confidence: float
    staff_index: int | None = None
    measure_index: int | None = None
    source_ref: str | None = None


@dataclass(frozen=True)
class GraphNode:
    id: str
    type: str
    bbox: BBox
    confidence: float
    staff_index: int | None
    measure_index: int | None
    features: dict[str, float]
    source_ref: str | None = None


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    type: str
    confidence: float
    features: dict[str, float]


@dataclass(frozen=True)
class HeteroGraph:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    def node_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in self.nodes:
            counts[node.type] = counts.get(node.type, 0) + 1
        return counts

    def edge_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for edge in self.edges:
            counts[edge.type] = counts.get(edge.type, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "node_type_counts": self.node_type_counts(),
            "edge_type_counts": self.edge_type_counts(),
        }


@dataclass(frozen=True)
class GraphInferenceSummary:
    engine: str
    node_updates: int
    edge_updates: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Staff:
    index: int
    line_ys: tuple[float, float, float, float, float]
    x_start: int
    x_end: int

    @property
    def top(self) -> float:
        return self.line_ys[0]

    @property
    def bottom(self) -> float:
        return self.line_ys[-1]

    @property
    def interline(self) -> float:
        diffs = [self.line_ys[i + 1] - self.line_ys[i] for i in range(4)]
        return sum(diffs) / len(diffs)


@dataclass(frozen=True)
class NoteEvent:
    x: float
    y: float
    staff_index: int
    measure: int
    step: str
    octave: int
    duration: int = 1
    note_type: str = "quarter"
    head_type: str = "filled"
    onset: float = 0.0
    voice: int = 1
    bbox: BBox | None = None
    stem_direction: str | None = None
    stem_x: float | None = None
    confidence: float = 0.0

    @property
    def pitch_name(self) -> str:
        return f"{self.step}{self.octave}"


@dataclass(frozen=True)
class StemEvent:
    x: float
    y_top: float
    y_bottom: float
    staff_index: int
    width: float
    confidence: float = 0.0


@dataclass(frozen=True)
class MeasureInfo:
    staff_index: int
    index: int
    x_start: float
    x_end: float
    note_count: int
    total_duration: int
    expected_duration: int = 4
    time_signature: str = "4/4"
    key_signature: str = "C"

    @property
    def status(self) -> str:
        if self.note_count == 0:
            return "empty"
        if self.total_duration == self.expected_duration:
            return "complete"
        if self.total_duration < self.expected_duration:
            return "underfull"
        return "overfull"


@dataclass(frozen=True)
class RecognitionResult:
    image_path: str
    page_number: int
    width: int
    height: int
    staves: tuple[Staff, ...]
    notes: tuple[NoteEvent, ...]
    scale: ScaleInfo | None = None
    stems: tuple[StemEvent, ...] = ()
    measures: tuple[MeasureInfo, ...] = ()
    symbols: tuple[SymbolDetection, ...] = ()
    graph: HeteroGraph | None = None
    graph_inference: GraphInferenceSummary | None = None
    rhythm_issues: tuple[str, ...] = ()
    steps: tuple[StepReport, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "scale": asdict(self.scale) if self.scale else None,
            "staves": [asdict(staff) for staff in self.staves],
            "stems": [asdict(stem) for stem in self.stems],
            "measures": [asdict(measure) | {"status": measure.status} for measure in self.measures],
            "symbols": [asdict(symbol) for symbol in self.symbols],
            "graph": self.graph.to_dict() if self.graph else None,
            "graph_inference": asdict(self.graph_inference) if self.graph_inference else None,
            "notes": [asdict(note) | {"pitch_name": note.pitch_name} for note in self.notes],
            "rhythm_issues": list(self.rhythm_issues),
            "steps": [asdict(step) for step in self.steps],
        }
