from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch

BatchFn = Callable[[object, object, torch.device], "tuple[torch.Tensor, torch.Tensor]"]


def _to_device_batch(batch, device, batch_fn: Optional[BatchFn]):
    if batch_fn is None:
        X, y = batch
        return X.to(device), y.to(device)
    images, labels = batch
    return batch_fn(images, labels, device)


def train_epoch(model, device, loader, optimizer, criterion,
                batch_fn: Optional[BatchFn] = None):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    start = time.perf_counter()
    for batch in loader:
        X, y = _to_device_batch(batch, device, batch_fn)
        optimizer.zero_grad()
        predict = model(X)
        loss = criterion(predict, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (predict.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return {"loss": loss_sum / total, "accuracy": correct / total,
            "seconds": time.perf_counter() - start}


@torch.no_grad()
def evaluate(model, device, loader, criterion, num_classes: int = 4,
             batch_fn: Optional[BatchFn] = None):
    model.eval()
    loss_sum = 0.0
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    for batch in loader:
        X, y = _to_device_batch(batch, device, batch_fn)
        predict = model(X)
        loss_sum += criterion(predict, y).item() * y.size(0)
        pred = predict.argmax(dim=1)
        for c in range(num_classes):
            mask = y == c
            total[c] += mask.sum().item()
            correct[c] += (pred[mask] == c).sum().item()
    recall = correct / np.maximum(total, 1)
    return {"images": int(total.sum()), "loss": loss_sum / total.sum(),
            "accuracy": correct.sum() / total.sum(),
            "balanced_accuracy": recall.mean(), "recall": recall.tolist()}
