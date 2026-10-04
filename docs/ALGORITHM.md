# Algorithm Design

## Convolutional symbol perception, heterogeneous relation reasoning, and music-aware decoding for OMR

Xunpu GraphOMR studies optical music recognition as **structured reconstruction**, not only symbol classification. Faster R-CNN/FPN and HGT are established model families; the research contribution explored here is their use within an inspectable music-symbol graph, relation-reconstruction layer, and musical-consistency decoding framework.

The public release currently runs an independent OpenCV/geometric baseline. Neural components are experimental scaffolding and no trained end-to-end OMR weights are released.

![System architecture](figures/architecture.png)

The figure distinguishes implemented counterparts from modules that still require training/integration.

## 1. Problem formulation

Given a score image $I\in\mathbb{R}^{H\times W\times C}$, the target is a structured musical event set $Y$ containing pitch, duration, onset, staff, measure, and eventually voice information.

The proposed research route is:

$$I\rightarrow D_{CNN}\rightarrow G_{hetero}\rightarrow P_{HGT}\rightarrow Y_{constraint}\rightarrow MusicXML/MIDI.$$

The deployed baseline replaces learned perception/relation inference with deterministic image processing and geometry while retaining inspectable intermediate structures.

## 2. Symbol perception

The experimental detector path uses Faster R-CNN with a ResNet-50 + FPN backbone. Candidate detections are represented as:

$$D=\{(b_i,c_i,s_i)\}_{i=1}^{N},\quad b_i=(x_i,y_i,w_i,h_i).$$

The current coarse symbol taxonomy includes accidental, barline, beam, clef, dot, flag, key signature, empty/filled notehead, rest, stem, and time signature. Staff and measure entities are derived structural nodes.

The detector scaffold uses small anchors suitable for compact notation symbols. No trained detector weights are committed, and the detector is not the default CLI path.

Code: `src/xunpu_graph_omr/vision/detection/`.

## 3. Heterogeneous graph representation

The score is represented as:

$$G=(V,E,\mathcal{T}_V,\mathcal{T}_E).$$

Nodes are typed notation/structural entities. Relations may include notehead–stem attachment, stem–beam connectivity, notehead–dot association, accidental–note association, staff membership, measure containment, and sequential context.

The HGT training adapter currently uses an 8-dimensional geometry vector:

$$g_i=[x/W,y/H,w/W,h/H,wh/(WH),w/h,x_c/W,y_c/H].$$

Candidate edges use 5-dimensional geometry features:

$$e_{ij}=[\Delta x/W,\Delta y/H,d/\sqrt{W^2+H^2},IoU,\theta/\pi].$$

Training adapters may add annotated positive edges to avoid losing ground-truth relations during training. Deployment evaluation must separately report candidate-edge recall without ground-truth completion.

A future extension may concatenate CNN crop embeddings $v_i$ and staff-position features:

$$h_i^{(0)}=W_{\tau(i)}[g_i;v_i;p_{staff}(i);s_i].$$

Crop-feature fusion is a research target, not a released capability of the current geometry-only HGT scaffold.

Code: `src/xunpu_graph_omr/graph/`.

## 4. HGT relation prediction

The experimental path uses multi-layer heterogeneous graph attention. An illustrative relation-aware message formulation is:

$$\alpha_{ij}^{r}=softmax_j\left(\frac{q_i^{\top}W_r k_j}{\sqrt{d}}\right),\quad m_i=\sum_r\sum_{j\in\mathcal{N}_r(i)}\alpha_{ij}^{r}W_r^Vh_j.$$

The current implementation uses PyTorch Geometric's HGTConv and separate edge-classification MLPs. Geometric edge features enter those MLP heads; they are not directly injected into a custom HGT attention equation.

A binary relation head can be written as:

$$P(r_{ij}=1)=softmax\left(MLP_{\tau(i),\tau(j)}[z_i;z_j;e_{ij}]\right)_1.$$

with edge loss:

$$\mathcal{L}_{edge}=-\sum_{(i,j)}\log P(r_{ij}=r_{ij}^{gt}).$$

The repository does not claim a trained, benchmarked HGT model.

## 5. Music-aware decoding

The intended decoder combines local confidence with musical consistency:

$$Y^*=\arg\max_Y\left[\sum_i\log P(y_i)+\sum_{ij}\log P(r_{ij})-\lambda\Omega_{music}(Y)\right].$$

For a time signature $a/b$, the expected measure duration per voice is:

$$T_m=a\cdot 4/b,$$

and a simple bar-duration penalty is:

$$\Omega_{bar}=\sum_{m,v}\left|\sum_{i\in(m,v)}d_i-T_m\right|.$$

For a dotted value with $k$ dots:

$$d_i=d_i^{base}\sum_{n=0}^{k}2^{-n}.$$

The public baseline currently implements treble-clef pitch reconstruction, whole/half/quarter notes, measure assembly, and rhythm checks under C-major / 4/4 assumptions. Full key-signature state, dotted values, voices, tuplets, beams/flags, and global constrained search remain future work.

Code: `src/xunpu_graph_omr/decoding/`.

## 6. Research pseudocode

```text
ResearchPipeline(image, detector_weights, graph_weights):
    normalized, staff_grid = preprocess(image)
    symbols, visual_features = CNNDetector(normalized, detector_weights)
    candidates = make_geometry_and_type_candidates(symbols, staff_grid)
    graph = build_heterograph(symbols, candidates, visual_features)
    relation_probs = HGT(graph, graph_weights)
    events = constrained_decode(symbols, relation_probs, staff_grid)
    return events, graph, MusicXML(events), MIDI(events)
```

This pseudocode describes the research direction. The runnable entry point is `xunpu-graph-omr recognize`.

## 7. Current evidence

The committed regression evidence contains two procedurally generated fixtures under eight deterministic image conditions each.

![Synthetic evaluation](figures/evaluation.png)

Current case-level macro averages are **96.9% pitch match** and **94.3% duration match**, with **14/16 exact pitch+duration sequences** and **16/16 lightweight XML parse checks**.

These cases are correlated transformations of two source fixtures. They are not an independent real-score test set and do not establish superiority over Audiveris or other OMR systems. See `examples/output/EVALUATION.md`.

## 8. Evaluation plan

A future real-score evaluation should isolate source works/writers across splits and report at least:

| Experiment | Perception | Relations | Purpose |
| --- | --- | --- | --- |
| A | fixed detector output | geometry/rules | relation baseline |
| B | fixed detector output | HGT | graph-learning contribution |
| C | fixed detector output | HGT + constraints | decoding contribution |
| D | external OMR system | native pipeline | external reference |

Relevant metrics include detector recall/AP, candidate-edge recall, relation precision/recall/F1, pitch/duration event error, measure legality, and score-level confidence intervals.

## 9. References

1. Ren et al., *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*.
2. Lin et al., *Feature Pyramid Networks for Object Detection*.
3. Hu et al., *Heterogeneous Graph Transformer*.

The architecture and evaluation figures in this repository are project-generated diagrams; they are not copied from those papers.
