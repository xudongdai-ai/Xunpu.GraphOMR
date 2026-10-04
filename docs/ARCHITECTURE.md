# Architecture

Xunpu GraphOMR separates optical perception, relational structure, musical reconstruction, and output serialization.

```text
PDF / image
  → vision
  → typed symbol graph
  → relation reasoning
  → decoding
  → MusicXML / MIDI / OMR JSON / diagnostics
```

## 1. Vision

`xunpu_graph_omr.vision` contains the runnable image-processing baseline:

- `document.py` — image loading and first-page PDF rasterization;
- `preprocess.py` — denoising, skew estimation, deskewing, binarization;
- `staff.py` / `scale.py` — staff geometry and scale estimation;
- `symbols.py` / `stems.py` — notehead and stem extraction;
- `detection/` — experimental Faster R-CNN research path.

The current baseline produces inspectable symbol candidates rather than hiding all perception inside a monolithic network.

## 2. Typed music-symbol graph

`xunpu_graph_omr.graph.graph` converts detections into typed nodes and relations.

Representative node types include noteheads, stems, barlines, staves, and measures. Relations include staff membership, measure containment, notehead-to-stem attachment, and sequential context.

Node features include normalized geometry, confidence, and staff-relative position. Edge features include normalized displacement, distance, overlap, angle, and staff context.

## 3. Relation reasoning

Two paths are kept distinct:

- `graph/gnn.py` — rule-based graph inspection used by the runnable baseline;
- `graph/hetero_gnn.py` — experimental HGT relation model.

The HGT module is research scaffolding and is not represented as a trained production model.

## 4. Decoding

`xunpu_graph_omr.decoding` converts visual/relational evidence into structured musical events:

- measure assembly and onset assignment;
- rhythm consistency checks;
- structured OMR document generation;
- reconstruction diagnostics.

The current validated scope assumes printed single-staff melody, treble clef, C major, and 4/4.

## 5. Export

`xunpu_graph_omr.export` provides:

- MusicXML serialization/parsing helpers;
- lightweight MIDI output;
- Jianpu helpers for downstream presentation.

The project also emits raw recognition JSON, structured OMR JSON, and visual overlays for inspection.

## 6. Evaluation

`xunpu_graph_omr.evaluation` contains procedural fixtures, metrics, optional external-baseline adapters, and deterministic degradation tests.

The committed results are regression evidence, not a claim of real-world benchmark superiority. The full workbench can be regenerated with:

```bash
xunpu-graph-omr portfolio --output out/evaluation
```

## Source layout

```text
src/xunpu_graph_omr/
├── vision/
├── graph/
├── decoding/
├── export/
├── evaluation/
├── model.py
├── pipeline.py
├── engines.py
├── audiveris.py
├── gui.py
└── cli.py
```
