"""Shared training / evaluation engine.

The nine original notebooks each copy-pasted a nearly identical ``train`` /
``test`` loop. They are consolidated here into a single implementation.

Two hooks make the single loop cover every experiment:

- ``batch_fn``: an optional callable ``(images, labels, device) -> (X, y)``.
  When ``None`` (the common case) each mini-batch is assumed to already be a
  ``(tensor, label)`` pair from a standard DataLoader. When provided (used by
  the "ours" denoising experiment) it receives the *raw* batch — e.g. a list
  of PIL images plus a label tensor - and is responsible for producing the
  model input ``X`` and target ``y`` on ``device``.
- ``save_path`` / ``save_every``: checkpoint control. When ``save_path`` is
  set, ``model.state_dict()`` is written every ``save_every`` epochs.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import torch
from tqdm import tqdm

BatchFn = Callable[[object, object, torch.device], "tuple[torch.Tensor, torch.Tensor]"]


def _to_device_batch(batch, device, batch_fn: Optional[BatchFn]):
    if batch_fn is None:
        X, y = batch
        return X.to(device), y.to(device)
    images, labels = batch
    return batch_fn(images, labels, device)


def train(model, device, trainloader, optimizer, criterion, num_epochs,
          save_path: Optional[str] = None, save_every: int = 10,
          checkpoint_prefix: str = "weight",
          batch_fn: Optional[BatchFn] = None) -> np.ndarray:
    """Train ``model`` and return a ``(num_epochs, 3)`` history array.

    Each history row is ``[epoch, avg_loss, avg_accuracy]``.
    """
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    history = np.zeros((0, 3))

    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for batch in trainloader:
            X, y = _to_device_batch(batch, device, batch_fn)

            optimizer.zero_grad()
            predict = model(X)
            loss = criterion(predict, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred_class = predict.argmax(dim=1)
            correct += (pred_class == y).sum().item()
            total += y.size(0)

        avg_loss = epoch_loss / len(trainloader)
        avg_accuracy = correct / total
        history = np.vstack((history, [epoch + 1, avg_loss, avg_accuracy]))
        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Loss: {avg_loss:.6f}, Accuracy: {avg_accuracy:.4f}")

        if save_path is not None and (epoch + 1) % save_every == 0:
            filename = os.path.join(save_path, f"{checkpoint_prefix}{epoch + 1}.pth")
            torch.save(model.state_dict(), filename)
            print(f"Saved checkpoint: {filename}")

    return history


def test(model, device, test_loader, criterion,
         batch_fn: Optional[BatchFn] = None):
    """Evaluate ``model`` and print/return ``(avg_loss, avg_accuracy)``.

    Accuracy is averaged over batches (mean of per-batch accuracy), matching
    the original notebooks.
    """
    test_loss, test_accuracy = [], []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            X, y = _to_device_batch(batch, device, batch_fn)

            predict = model(X)
            loss = criterion(predict, y)

            pred_class = predict.argmax(dim=1)
            accuracy = (pred_class == y).float().mean()

            test_accuracy.append(accuracy.item())
            test_loss.append(loss.item())

    avg_loss = sum(test_loss) / len(test_loss)
    avg_accuracy = sum(test_accuracy) / len(test_accuracy)
    print(f"test loss : {avg_loss:.4f} / test_accuracy : {avg_accuracy:.4f}")
    return avg_loss, avg_accuracy


def evaluate_checkpoints(model_factory: Callable[[], torch.nn.Module],
                         checkpoint_paths, device, test_loader, criterion,
                         batch_fn: Optional[BatchFn] = None):
    """Load and evaluate a series of checkpoints.

    ``model_factory`` must return a *fresh* (untrained) model with the correct
    architecture for the checkpoints. This consolidates the "loop over saved
    weights and test each" block repeated across the original denoising notebooks.
    """
    results = []
    for path in checkpoint_paths:
        print(f"\nTesting {path}")
        model = model_factory()
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        results.append((path, *test(model, device, test_loader, criterion,
                                     batch_fn=batch_fn)))
    return results


def get_confusion_matrix(model, dataloader, device, num_classes: int = 4,
                         batch_fn: Optional[BatchFn] = None) -> np.ndarray:
    """Return the ``num_classes x num_classes`` confusion matrix on ``dataloader``.

    Rows are the true classes, columns the predicted ones. ``batch_fn`` has the
    same meaning as in :func:`test`, so the per-pixel-denoising experiments can
    reuse this without materialising a separate loader.
    """
    from sklearn.metrics import confusion_matrix

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            X, y = _to_device_batch(batch, device, batch_fn)
            preds = model(X).argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    if not all_preds:
        raise ValueError("dataloader yielded no batches")

    return confusion_matrix(torch.cat(all_labels).numpy(),
                            torch.cat(all_preds).numpy(),
                            labels=range(num_classes))


def plot_confusion_matrix(cm: np.ndarray, class_names=None, save_path=None,
                          cmap: str = "Oranges"):
    """Render ``cm`` as an annotated heatmap.

    Writes to ``save_path`` when given, otherwise shows the figure. Returns the
    matplotlib figure so callers can compose it further.
    """
    import matplotlib
    if save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"C{i}" for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap=cmap)

    ax.set_xticks(range(num_classes), class_names)
    ax.set_yticks(range(num_classes), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # Annotate each cell, flipping the text colour on dark backgrounds.
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig
