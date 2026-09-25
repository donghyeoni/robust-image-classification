from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from rc.data import build_dataloader, seed_everything
from rc.engine import evaluate, train_epoch
from rc.model import build_resnet18

NOISE_LEVELS = (0.05, 0.10, 0.25, 0.50)
EVAL_SEED = 1000


@dataclass
class Pipeline:
    transform: Optional[Callable] = None
    collate_fn: Optional[Callable] = None
    batch_fn: Optional[Callable] = None


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def run(args, name: str, in_channels: int, train: Pipeline,
        tests: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.join(args.results_dir, name, f"seed{args.seed}")
    ckpt_dir = os.path.join(args.checkpoint_dir, name, f"seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    seed_everything(args.seed)
    trainset, loader = build_dataloader(
        args.data_root, "Train", train.transform, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
        collate_fn=train.collate_fn)
    classes = trainset.classes
    model = build_resnet18(in_channels=in_channels,
                           num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history, epochs_saved = [], []
    for epoch in range(1, args.epochs + 1):
        h = train_epoch(model, device, loader, optimizer, criterion,
                        batch_fn=train.batch_fn)
        history.append({"epoch": epoch, "loss": f"{h['loss']:.6f}",
                        "accuracy": f"{h['accuracy']:.4f}",
                        "seconds": f"{h['seconds']:.1f}"})
        print(f"{name} epoch {epoch}/{args.epochs} loss {h['loss']:.6f} "
              f"acc {h['accuracy']:.4f} {h['seconds']:.1f}s", flush=True)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f"weight{epoch}.pth"))
            epochs_saved.append(epoch)
    _write_csv(os.path.join(out, "train.csv"), history)

    rows = []
    for epoch in epochs_saved:
        model.load_state_dict(torch.load(
            os.path.join(ckpt_dir, f"weight{epoch}.pth"), map_location=device,
            weights_only=True))
        for noise, pipe in tests.items():
            seed_everything(EVAL_SEED)
            _, test_loader = build_dataloader(
                args.data_root, "Test", pipe.transform,
                batch_size=args.batch_size, num_workers=args.num_workers,
                seed=EVAL_SEED, collate_fn=pipe.collate_fn)
            m = evaluate(model, device, test_loader, criterion,
                         num_classes=len(classes), batch_fn=pipe.batch_fn)
            rows.append({"epoch": epoch,
                         "test_noise": "" if noise is None else noise,
                         "images": m["images"], "loss": f"{m['loss']:.4f}",
                         "accuracy": f"{m['accuracy']:.4f}",
                         "balanced_accuracy": f"{m['balanced_accuracy']:.4f}",
                         **{f"recall_{c}": f"{r:.4f}"
                            for c, r in zip(classes, m["recall"])}})
            print(f"{name} test epoch {epoch} noise {noise} "
                  f"acc {m['accuracy']:.4f} bal {m['balanced_accuracy']:.4f}",
                  flush=True)
    _write_csv(os.path.join(out, "metrics.csv"), rows)

    with open(os.path.join(out, "run.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump({"experiment": name, **{k: v for k, v in vars(args).items()
                                          if k not in ("data_root",)},
                   "device": torch.cuda.get_device_name(device)
                   if device.type == "cuda" else "cpu",
                   "torch": torch.__version__,
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, indent=2)
        f.write("\n")
