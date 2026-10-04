"""
Train a Faster R-CNN OMR symbol detector on DoReMi and/or MUSCIMA++.

Examples
--------
Printed only::

    python -m xunpu_graph_omr.vision.detection.train_detector \\
        --doremi-images /data/doremi/images \\
        --doremi-coco   /data/doremi/annotations.json \\
        --out out/detector

Both printed + handwritten::

    python -m xunpu_graph_omr.vision.detection.train_detector \\
        --doremi-images /data/doremi/images --doremi-coco /data/doremi/ann.json \\
        --muscima-ann /data/muscima/annotations --muscima-img /data/muscima/images \\
        --out out/detector
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

from xunpu_graph_omr.vision.detection.datasets import (
    DoReMiDataset,
    MuscimaDetectionDataset,
    collate_fn,
)
from xunpu_graph_omr.vision.detection.model import build_fasterrcnn

log = logging.getLogger(__name__)


def build_dataset(args: argparse.Namespace):
    parts = []
    if args.doremi_images and args.doremi_coco:
        ds = DoReMiDataset(args.doremi_images, args.doremi_coco)
        log.info("DoReMi (printed): %d images", len(ds))
        parts.append(ds)
    if args.muscima_ann and args.muscima_img:
        ds = MuscimaDetectionDataset(args.muscima_ann, args.muscima_img)
        log.info("MUSCIMA++ (handwritten): %d images", len(ds))
        parts.append(ds)
    if not parts:
        raise SystemExit("Provide --doremi-* and/or --muscima-* dataset paths.")
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


@torch.no_grad()
def eval_loss(model, loader, device) -> float:
    # torchvision detection models only emit losses in train() mode; we keep
    # train() but disable grad to get a comparable validation loss signal.
    model.train()
    total, n = 0.0, 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss = sum(model(images, targets).values())
        total += float(loss)
        n += 1
    return total / max(1, n)


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    dataset = build_dataset(args)
    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    log.info("split: train=%d val=%d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.workers,
    )

    model = build_fasterrcnn().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            running += float(loss)
            if step % args.log_every == 0:
                log.info("epoch=%d step=%d loss=%.4f", epoch, step, running / step)
        scheduler.step()

        val = eval_loss(model, val_loader, device)
        log.info("epoch=%d  train_loss=%.4f  val_loss=%.4f", epoch, running / max(1, len(train_loader)), val)
        if val < best_val:
            best_val = val
            ckpt = out_dir / "best_detector.pt"
            torch.save(model.state_dict(), ckpt)
            log.info("checkpoint → %s  (val_loss=%.4f)", ckpt, best_val)

    torch.save(model.state_dict(), out_dir / "last_detector.pt")
    log.info("done. best_val_loss=%.4f", best_val)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Faster R-CNN OMR detector")
    p.add_argument("--doremi-images")
    p.add_argument("--doremi-coco")
    p.add_argument("--muscima-ann")
    p.add_argument("--muscima-img")
    p.add_argument("--out", default="out/detector")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=20)
    return p


if __name__ == "__main__":
    train(_parser().parse_args())
