"""
src/evaluate.py
---------------
Evaluation utilities: classification report, ROC / PR curves, confusion
matrix, training-curve plots, and uncertainty distribution visualisation.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix,
)

import torch
import torch.nn.functional as F

from src.model import HierarchicalPainRecognition


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation entry points
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(trainer, config: dict) -> None:
    """Run and report full evaluation on the held-out test set."""
    out_dir = config.get("output_dir", "results")
    os.makedirs(out_dir, exist_ok=True)

    print("\nRunning test-set evaluation...")
    _, test_acc, preds, labels, probs, uncertainties = trainer.validate(
        trainer.test_loader
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 54)
    print(f"  Test Accuracy : {test_acc:.2f}%")
    print("\n  Classification Report:")
    print(
        classification_report(
            labels, preds, target_names=["Low Pain", "High Pain"]
        )
    )
    cm = confusion_matrix(labels, preds)
    print("  Confusion Matrix:")
    print(cm)
    print("=" * 54)

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_training_curves(trainer.history, out_dir)
    plot_evaluation_metrics(labels, probs, uncertainties, out_dir)
    plot_confusion_matrix(labels, preds, out_dir)

    print(f"\nFigures saved to '{out_dir}/'")


def evaluate_from_checkpoint(config: dict) -> None:
    """Load a saved checkpoint and evaluate on the test split."""
    from src.dataset import OptimizedPainRecognitionDataset, PainDataAugmentation
    from src.trainer import EnhancedTrainer
    from src.utils import get_dataloader

    device = config["device"]
    ckpt   = torch.load(config["MODEL_PATH"], map_location=device)

    model_cfg = ckpt.get("config", config)
    model = HierarchicalPainRecognition(
        num_classes=model_cfg.get("num_classes", 2),
        dropout_rate=model_cfg.get("dropout_rate", 0.4),
        segment_length_sec=model_cfg.get("segment_length_sec", 10.0),
        feature_dim=model_cfg.get("feature_dim", 256),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print("Checkpoint loaded.")

    aug = PainDataAugmentation(training=False)
    test_ds = OptimizedPainRecognitionDataset(
        config["data_path"], "test", config["segment_length_sec"],
        train_split=config["train_split"], val_split=config["val_split"],
        augmentations=aug,
    )
    test_loader = get_dataloader(test_ds, config, shuffle=False)

    # Wrap in a minimal trainer-like object for validate()
    class _Wrapper:
        def __init__(self, m, cfg):
            self.model       = m
            self.device      = cfg["device"]
            self.config      = cfg
            self.test_loader = test_loader
            from src.losses import FocalLoss
            self.cls_criterion = FocalLoss()
            self.history     = {"train_loss": [], "val_loss": [], "val_acc": []}

        def validate(self, loader):
            return _trainer_validate(self, loader)

    wrapper = _Wrapper(model, config)
    run_evaluation(wrapper, config)


def _trainer_validate(self, loader):
    """Reusable validate logic (mirrors EnhancedTrainer.validate)."""
    self.model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs, all_unc = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            physio = batch["physio"].to(self.device)
            video  = {k: v.to(self.device) for k, v in batch["video"].items()}
            audio  = batch["audio"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                logits, _, uncertainty = self.model(
                    physio, video, audio, return_uncertainty=True
                )
                loss = self.cls_criterion(logits["ensemble"], labels)

            probs    = F.softmax(logits["ensemble"], dim=1)
            preds    = probs.argmax(dim=1)
            total   += labels.size(0)
            correct += preds.eq(labels).sum().item()
            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_unc.extend(uncertainty.squeeze().cpu().numpy())

    avg_loss = total_loss / len(loader) if loader else 0.0
    accuracy = 100.0 * correct / total   if total  else 0.0
    return avg_loss, accuracy, all_preds, all_labels, all_probs, all_unc


# ─────────────────────────────────────────────────────────────────────────────
# Plotting utilities
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, out_dir: str = "results") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history["val_acc"], color="orange", label="Val Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_evaluation_metrics(
    labels, probs, uncertainties, out_dir: str = "results"
) -> None:
    probs_arr = np.array(probs)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC
    fpr, tpr, _ = roc_curve(labels, probs_arr[:, 1])
    roc_auc     = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curve"); axes[0].legend(); axes[0].grid(True)

    # PR
    precision, recall, _ = precision_recall_curve(labels, probs_arr[:, 1])
    ap = average_precision_score(labels, probs_arr[:, 1])
    axes[1].plot(recall, precision, color="purple", label=f"AP = {ap:.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(); axes[1].grid(True)

    # Uncertainty distribution
    if uncertainties:
        sns.histplot(uncertainties, bins=50, kde=True, ax=axes[2])
    axes[2].set_title("Uncertainty Distribution")
    axes[2].set_xlabel("Predicted Uncertainty"); axes[2].grid(True)

    plt.tight_layout()
    path = os.path.join(out_dir, "evaluation_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrix(labels, preds, out_dir: str = "results") -> None:
    cm   = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Low Pain", "High Pain"],
        yticklabels=["Low Pain", "High Pain"],
        ax=ax,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
