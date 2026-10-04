from __future__ import annotations

from xunpu_graph_omr.model import GraphInferenceSummary, HeteroGraph


class RuleBasedGraphInferenceEngine:
    """Drop-in graph inference stage until trained HGT/R-GAT weights exist."""

    name = "rule-based-heterograph"

    def infer(self, graph: HeteroGraph) -> GraphInferenceSummary:
        edge_counts = graph.edge_type_counts()
        notes: list[str] = []
        if edge_counts.get("notehead_to_stem", 0) == 0 and graph.node_type_counts().get("notehead_filled", 0):
            notes.append("no notehead_to_stem edges found")
        if edge_counts.get("inside_measure", 0) == 0:
            notes.append("no inside_measure edges found")
        return GraphInferenceSummary(
            engine=self.name,
            node_updates=0,
            edge_updates=0,
            notes=tuple(notes),
        )
