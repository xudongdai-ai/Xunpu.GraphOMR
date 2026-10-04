import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import HeteroData
from xunpu_graph_omr.graph.hetero_gnn import OMRHeteroGNN
from xunpu_graph_omr.graph.muscima_dataset import CANDIDATE_PAIRS, EDGE_FEAT_DIM, NODE_FEAT_DIM, NODE_TYPES


def graph_fixture():
    graph = HeteroData()
    for kind in NODE_TYPES:
        count = 1 if kind in ("notehead_filled", "stem", "rest") else 0
        graph[kind].x = torch.ones(count, NODE_FEAT_DIM)
    for source, target in CANDIDATE_PAIRS:
        linked = (source, target) == ("notehead_filled", "stem")
        relation = graph[source, "to", target]
        relation.edge_index = torch.tensor([[0], [0]]) if linked else torch.empty((2, 0), dtype=torch.long)
        relation.edge_attr = torch.zeros(1 if linked else 0, EDGE_FEAT_DIM)
    return graph


def test_multilayer_hgt_keeps_source_and_isolated_nodes():
    model = OMRHeteroGNN(hidden_dim=16, num_layers=3, num_heads=4)
    graph = graph_fixture()
    encoded = model.encode(graph)
    assert set(encoded) == {"notehead_filled", "stem", "rest"}
    logits = model(graph)["notehead_filled", "stem"]
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, torch.ones(1, dtype=torch.long)).backward()
    assert model.input_proj["notehead_filled"].weight.grad is not None


def test_detector_anchor_head_and_small_image_forward():
    pytest.importorskip("torchvision")
    from xunpu_graph_omr.vision.detection.model import build_fasterrcnn
    from xunpu_graph_omr.vision.detection.labels import NUM_CLASSES

    model = build_fasterrcnn(pretrained_backbone=False).eval()
    assert model.rpn.head.cls_logits.out_channels == 4
    assert model.roi_heads.box_predictor.cls_score.out_features == NUM_CLASSES
    model.transform.min_size = (64,)
    model.transform.max_size = 96
    with torch.no_grad():
        result = model([torch.ones(3, 64, 96)])
    assert result[0]["boxes"].shape[1] == 4
