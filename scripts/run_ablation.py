"""
scripts/run_ablation.py
-----------------------
Reproduces the ablation study from Table II of the ACII 2025 paper.

Each experiment systematically removes or modifies one component of the
full framework and runs the complete training + evaluation pipeline,
reporting accuracy, F1, precision and recall for the High Pain class.

Usage
-----
    python scripts/run_ablation.py --config configs/default.yaml

The script writes a summary CSV to ``results/ablation_results.csv``.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import yaml
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score

from src.dataset  import OptimizedPainRecognitionDataset, PainDataAugmentation
from src.model    import HierarchicalPainRecognition
from src.trainer  import EnhancedTrainer
from src.utils    import set_seed, get_dataloader, print_banner


# ─────────────────────────────────────────────────────────────────────────────
# Ablation experiment definitions
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    # ── Full model ────────────────────────────────────────────────────────────
    {
        "name":          "Full Model (Benchmark)",
        "modalities":    ["physio", "video", "audio"],
        "use_focal_loss":True,
        "contrastive_loss_weight": 0.2,
        "simple_concat": False,
        "end_to_end":    False,
    },
    # ── Fusion architecture ───────────────────────────────────────────────────
    {
        "name":          "Ablation: Simple Concat Fusion",
        "modalities":    ["physio", "video", "audio"],
        "use_focal_loss":True,
        "contrastive_loss_weight": 0.2,
        "simple_concat": True,
        "end_to_end":    False,
    },
    # ── Training strategy ─────────────────────────────────────────────────────
    {
        "name":          "Ablation: End-to-End Training",
        "modalities":    ["physio", "video", "audio"],
        "use_focal_loss":True,
        "contrastive_loss_weight": 0.2,
        "simple_concat": False,
        "end_to_end":    True,
    },
    # ── Loss functions ────────────────────────────────────────────────────────
    {
        "name":          "Ablation: No Contrastive Loss",
        "modalities":    ["physio", "video", "audio"],
        "use_focal_loss":True,
        "contrastive_loss_weight": 0.0,
        "simple_concat": False,
        "end_to_end":    False,
    },
    {
        "name":          "Ablation: Cross-Entropy Loss Only",
        "modalities":    ["physio", "video", "audio"],
        "use_focal_loss":False,
        "contrastive_loss_weight": 0.0,
        "simple_concat": False,
        "end_to_end":    False,
    },
    # ── Modality contribution ─────────────────────────────────────────────────
    {
        "name":          "Ablation: Physio + Video",
        "modalities":    ["physio", "video"],
        "use_focal_loss":True,
        "contrastive_loss_weight": 0.2,
        "simple_concat": False,
        "end_to_end":    False,
    },
    {
        "name":          "Ablation: Physio + Audio",
        "modalities":    ["physio", "audio"],
        "use_focal_loss":True,
        "contrastive_loss_weight": 0.2,
        "simple_concat": False,
        "end_to_end":    False,
    },
    {
        "name":          "Ablation: Video + Audio",
        "modalities":    ["video", "audio"],
        "use_focal_loss":True,
        "contrastive_loss_weight": 0.2,
        "simple_concat": False,
        "end_to_end":    False,
    },
    {
        "name":          "Ablation: Physio Only",
        "modalities":    ["physio"],
        "use_focal_loss":True,
        "contrastive_loss_weight": 0.2,
        "simple_concat": False,
        "end_to_end":    False,
    },
    {
        "name":          "Ablation: Video Only",
        "modalities":    ["video"],
        "use_focal_loss":True,
        "contrastive_loss_weight": 0.2,
        "simple_concat": False,
        "end_to_end":    False,
    },
    {
        "name":          "Ablation: Audio Only",
        "modalities":    ["audio"],
        "use_focal_loss":True,
        "contrastive_loss_weight": 0.2,
        "simple_concat": False,
        "end_to_end":    False,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mask_modalities(batch: dict, active_modalities: list, device: str) -> tuple:
    """Zero-out modalities that are excluded for this experiment."""
    physio = batch["physio"].to(device)
    video  = {k: v.to(device) for k, v in batch["video"].items()}
    audio  = batch["audio"].to(device)

    if "physio" not in active_modalities:
        physio = torch.zeros_like(physio)
    if "video" not in active_modalities:
        video  = {k: torch.zeros_like(v) for k, v in video.items()}
    if "audio" not in active_modalities:
        audio  = torch.zeros_like(audio)

    return physio, video, audio


def _run_one(exp: dict, base_config: dict, datasets: tuple) -> dict:
    """Train and evaluate one ablation configuration."""
    config = copy.deepcopy(base_config)
    config["use_focal_loss"]          = exp["use_focal_loss"]
    config["contrastive_loss_weight"] = exp["contrastive_loss_weight"]

    if exp.get("end_to_end"):
        config["epochs_phase1"] = 0
        config["epochs_phase2"] = 0
        config["epochs_phase3"] = config["epochs_phase1"] + config["epochs_phase2"] + 20

    train_ds, val_ds, test_ds = datasets
    train_loader = get_dataloader(train_ds, config, shuffle=True,  drop_last=True)
    val_loader   = get_dataloader(val_ds,   config, shuffle=False)
    test_loader  = get_dataloader(test_ds,  config, shuffle=False)

    augs  = PainDataAugmentation(training=True)
    model = HierarchicalPainRecognition(
        num_classes=2, dropout_rate=0.4,
        segment_length_sec=config["segment_length_sec"],
        augmentations=augs,
    )
    trainer = EnhancedTrainer(model, train_loader, val_loader, test_loader, config)
    trainer.train()

    # ── Custom validate that masks inactive modalities ────────────────────────
    model.eval()
    all_preds, all_labels = [], []
    device = config["device"]
    with torch.no_grad():
        for batch in test_loader:
            physio, video, audio = _mask_modalities(batch, exp["modalities"], device)
            labels = batch["label"].to(device)
            logits, _ = model(physio, video, audio)
            preds = logits["ensemble"].argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc     = 100.0 * correct / len(all_labels)
    report  = classification_report(
        all_labels, all_preds,
        target_names=["Low Pain", "High Pain"],
        output_dict=True, zero_division=0,
    )
    hp = report.get("High Pain", {})
    return {
        "Experiment":          exp["name"],
        "Accuracy (%)":        round(acc, 2),
        "F1 Weighted (%)":     round(report["weighted avg"]["f1-score"] * 100, 2),
        "Precision (High) (%)":round(hp.get("precision", 0) * 100, 2),
        "Recall (High) (%)":   round(hp.get("recall",    0) * 100, 2),
        "F1 (High) (%)":       round(hp.get("f1-score",  0) * 100, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(config.get("random_seed", 42))
    print_banner("Ablation Study")

    # Build datasets once (shared across experiments for fairness)
    train_augs = PainDataAugmentation(training=True)
    val_augs   = PainDataAugmentation(training=False)

    print("Loading datasets (shared across all experiments)...")
    train_ds = OptimizedPainRecognitionDataset(
        config["data_path"], "train", config["segment_length_sec"],
        train_split=config["train_split"], val_split=config["val_split"],
        augmentations=train_augs,
    )
    val_ds = OptimizedPainRecognitionDataset(
        config["data_path"], "val", config["segment_length_sec"],
        train_split=config["train_split"], val_split=config["val_split"],
        augmentations=val_augs,
    )
    test_ds = OptimizedPainRecognitionDataset(
        config["data_path"], "test", config["segment_length_sec"],
        train_split=config["train_split"], val_split=config["val_split"],
        augmentations=val_augs,
    )

    results = []
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] {exp['name']}")
        try:
            row = _run_one(exp, config, (train_ds, val_ds, test_ds))
            results.append(row)
            print(
                f"  → Acc={row['Accuracy (%)']:.2f}%  "
                f"F1w={row['F1 Weighted (%)']:.2f}%  "
                f"Prec(H)={row['Precision (High) (%)']:.2f}%  "
                f"Rec(H)={row['Recall (High) (%)']:.2f}%"
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"Experiment": exp["name"], "ERROR": str(e)})

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(config.get("output_dir", "results"), exist_ok=True)
    out_csv = os.path.join(config.get("output_dir", "results"), "ablation_results.csv")
    if results:
        keys = [k for k in results[0] if k != "ERROR"]
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nAblation results saved → {out_csv}")

    # ── Console table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    header = f"{'Experiment':<45} {'Acc':>6} {'F1w':>6} {'P(H)':>6} {'R(H)':>6} {'F1(H)':>6}"
    print(header)
    print("-" * 90)
    for r in results:
        if "ERROR" in r:
            print(f"{r['Experiment']:<45}  ERROR")
        else:
            print(
                f"{r['Experiment']:<45} "
                f"{r['Accuracy (%)']:>6.2f} "
                f"{r['F1 Weighted (%)']:>6.2f} "
                f"{r['Precision (High) (%)']:>6.2f} "
                f"{r['Recall (High) (%)']:>6.2f} "
                f"{r['F1 (High) (%)']:>6.2f}"
            )
    print("=" * 90)


if __name__ == "__main__":
    main()
