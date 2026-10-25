"""Training loop for the SignGRU, with early stopping and checkpointing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import config, evaluate
from . import dataset as ds
from . import model as model_mod


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, hit, loss_sum = 0, 0, 0.0
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(seqs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * len(labels)
        hit += int((logits.argmax(1) == labels).sum())
        total += len(labels)
    return loss_sum / max(total, 1), hit / max(total, 1)


@torch.no_grad()
def evaluate_split(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, labels_all = [], []
    for seqs, labels in loader:
        seqs = seqs.to(device)
        logits_all.append(model(seqs).cpu().numpy())
        labels_all.append(labels.numpy())
    return (np.concatenate(logits_all) if logits_all else np.zeros((0, config.N_CLASSES)),
            np.concatenate(labels_all) if labels_all else np.zeros(0, dtype=np.int64))


def run_training(epochs: int = config.EPOCHS, batch_size: int = config.BATCH_SIZE,
                 smoke: bool = False, out_path: Path | None = None,
                 results_dir: Path | None = None) -> dict:
    """Train the biGRU and return the final metrics dict.

    ``smoke`` caps the dataset and epochs so unit tests stay fast.
    """
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = torch.device("cpu")
    results_dir = Path(results_dir or config.RESULTS_DIR)
    out_path = Path(out_path or config.MODEL_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ds.SignDataset("train", training=True)
    val_ds = ds.SignDataset("val")
    test_ds = ds.SignDataset("test")
    if smoke:
        train_ds.items = train_ds.items[:64]
        val_ds.items = val_ds.items[:32]
        test_ds.items = test_ds.items[:32]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=ds.collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=ds.collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=ds.collate)

    model = model_mod.SignGRU().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=ds.class_weights().to(device) if not smoke else None,
        label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    best_acc, best_state, history = 0.0, None, []
    patience = 0
    n_epochs = 2 if smoke else epochs
    for epoch in range(1, n_epochs + 1):
        start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_logits, val_labels = evaluate_split(model, val_loader, device)
        val_metrics = evaluate.compute_metrics(val_labels, val_logits.argmax(1),
                                               ds.class_list())
        scheduler.step(val_metrics["accuracy"])
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "seconds": round(time.time() - start, 1),
        })
        print(f"epoch {epoch:2d}  train_loss {tr_loss:.4f}  "
              f"train_acc {tr_acc:.3f}  val_acc {val_metrics['accuracy']:.3f}  "
              f"val_f1 {val_metrics['macro_f1']:.3f}  ({history[-1]['seconds']}s)", flush=True)
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.PATIENCE and not smoke:
                print(f"early stop at epoch {epoch} (no val improvement for {patience})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_logits, test_labels = evaluate_split(model, test_loader, device)
    test_metrics = evaluate.compute_metrics(test_labels, test_logits.argmax(1),
                                            ds.class_list())
    topk = evaluate.topk_accuracy(test_logits, test_labels, (1, 5))
    test_metrics["top5"] = topk["top5"]
    test_metrics["model"] = "SignGRU (biGRU, 2x256, 32-frame landmarks)"
    test_metrics["n_params"] = model_mod.count_parameters(model)

    torch.save({"state_dict": model.state_dict(),
                "labels": ds.class_list(),
                "config": {"hidden": config.HIDDEN, "layers": config.LAYERS,
                           "n_frames": config.N_FRAMES}},
               out_path)
    np.savez_compressed(results_dir / "sign_predictions.npz",
                        logits=test_logits, labels=test_labels)
    (results_dir / "sign_train_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    summary = {"best_val_acc": best_acc, "test": test_metrics,
               "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
               "split_report": ds.split_report()}
    (results_dir / "sign_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run for tests/CI")
    args = parser.parse_args()
    summary = run_training(epochs=args.epochs, batch_size=args.batch_size,
                           smoke=args.smoke)
    print(json.dumps({"test_acc": summary["test"]["accuracy"],
                      "test_macro_f1": summary["test"]["macro_f1"],
                      "top5": summary["test"]["top5"]}, indent=2))


if __name__ == "__main__":
    main()
