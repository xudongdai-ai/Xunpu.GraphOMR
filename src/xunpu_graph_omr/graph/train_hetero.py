"""
Train OMRHeteroGNN on MUSCIMA++.

Quick start
-----------
    python -m xunpu_graph_omr.graph.train_hetero \\
        --data /path/to/muscima_pp/v2.0/data \\
        --out  out/hetero_gnn

The data directory must contain ``annotations/*.xml`` (MUNG format).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from xunpu_graph_omr.graph.hetero_gnn import OMRHeteroGNN
from xunpu_graph_omr.graph.muscima_dataset import CANDIDATE_PAIRS, MuscimaPlusPlusDataset

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Loss                                                                         #
# --------------------------------------------------------------------------- #

def compute_loss(
    model: OMRHeteroGNN,
    data,
    device: torch.device,
    pos_weight_factor: float = 5.0,
) -> torch.Tensor:
    """Cross-entropy averaged over relation types; positives are upweighted."""
    data = data.to(device)
    logits_dict = model(data)
    total = torch.tensor(0.0, device=device)
    count = 0
    for src_type, dst_type in CANDIDATE_PAIRS:
        if (src_type, dst_type) not in logits_dict:
            continue
        logits = logits_dict[(src_type, dst_type)]
        labels = data[(src_type, "to", dst_type)].y.to(device)
        if labels.numel() == 0:
            continue
        pos = labels.sum().float()
        neg = (labels == 0).sum().float()
        weight = torch.ones(2, device=device)
        if pos > 0:
            weight[1] = (neg / pos) * pos_weight_factor
        total = total + F.cross_entropy(logits, labels, weight=weight)
        count += 1
    return total / max(1, count)


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model: OMRHeteroGNN, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    for data in loader:
        data = data.to(device)
        logits_dict = model(data)
        for src_type, dst_type in CANDIDATE_PAIRS:
            if (src_type, dst_type) not in logits_dict:
                continue
            preds = logits_dict[(src_type, dst_type)].argmax(dim=-1)
            labels = data[(src_type, "to", dst_type)].y.to(device)
            tp += int(((preds == 1) & (labels == 1)).sum())
            fp += int(((preds == 1) & (labels == 0)).sum())
            fn += int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------- #
# Training loop                                                                #
# --------------------------------------------------------------------------- #

def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    train_ds = MuscimaPlusPlusDataset(args.data, split="train")
    val_ds = MuscimaPlusPlusDataset(args.data, split="val")
    log.info("dataset: train=%d val=%d", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    model = OMRHeteroGNN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    log.info("params: %d", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = compute_loss(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, val_loader, device)
            log.info(
                "epoch=%d  loss=%.4f  prec=%.3f  rec=%.3f  f1=%.3f",
                epoch, avg_loss,
                metrics["precision"], metrics["recall"], metrics["f1"],
            )
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                ckpt = out_dir / "best_model.pt"
                torch.save(model.state_dict(), ckpt)
                log.info("checkpoint → %s  (f1=%.3f)", ckpt, best_f1)
        else:
            log.info("epoch=%d  loss=%.4f", epoch, avg_loss)

    log.info("done. best_f1=%.3f", best_f1)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train OMR HeteroGNN on MUSCIMA++")
    p.add_argument("--data", required=True, help="MUSCIMA++ data dir (contains annotations/)")
    p.add_argument("--out", default="out/hetero_gnn", help="output dir for checkpoints")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=5, help="evaluate on val every N epochs")
    return p


if __name__ == "__main__":
    train(_parser().parse_args())
