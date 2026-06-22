"""
main.py
-------
Entry point for the Hierarchical Multimodal Pain Recognition system.

Usage:
    python main.py --mode train --config configs/default.yaml
    python main.py --mode evaluate --checkpoint path/to/model.pth
    python main.py --mode predict --checkpoint path/to/model.pth --data path/to/test_data
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.dataset import OptimizedPainRecognitionDataset, PainDataAugmentation
from src.model import HierarchicalPainRecognition
from src.trainer import EnhancedTrainer
from src.utils import set_seed, get_dataloader, print_banner
from src.evaluate import run_evaluation
from src.inference import generate_submission


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_training(config: dict):
    print_banner("Training Mode")
    set_seed(config.get("random_seed", 42))

    train_augs = PainDataAugmentation(training=True)
    val_test_augs = PainDataAugmentation(training=False)

    print("Loading datasets...")
    train_dataset = OptimizedPainRecognitionDataset(
        config["data_path"], "train", config["segment_length_sec"],
        train_split=config["train_split"], val_split=config["val_split"],
        augmentations=train_augs
    )
    val_dataset = OptimizedPainRecognitionDataset(
        config["data_path"], "val", config["segment_length_sec"],
        train_split=config["train_split"], val_split=config["val_split"],
        augmentations=val_test_augs
    )
    test_dataset = OptimizedPainRecognitionDataset(
        config["data_path"], "test", config["segment_length_sec"],
        train_split=config["train_split"], val_split=config["val_split"],
        augmentations=val_test_augs
    )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Check your data path and structure.")

    train_loader = get_dataloader(train_dataset, config, shuffle=True, drop_last=True)
    val_loader   = get_dataloader(val_dataset,   config, shuffle=False)
    test_loader  = get_dataloader(test_dataset,  config, shuffle=False)

    print("\nInitialising model...")
    model = HierarchicalPainRecognition(
        num_classes=config.get("num_classes", 2),
        dropout_rate=config.get("dropout_rate", 0.4),
        segment_length_sec=config["segment_length_sec"],
        augmentations=train_augs,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    trainer = EnhancedTrainer(model, train_loader, val_loader, test_loader, config)
    trainer.train()

    print("\nRunning final evaluation on test set...")
    run_evaluation(trainer, config)


def run_predict(config: dict, checkpoint_path: str, data_path: str):
    print_banner("Inference / Submission Mode")
    config["MODEL_PATH"] = checkpoint_path
    config["TEST_DATA_PATH"] = data_path
    generate_submission(config)


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Multimodal Pain Recognition — X-ITE Challenge"
    )
    parser.add_argument(
        "--mode", choices=["train", "evaluate", "predict"], default="train",
        help="Operating mode."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a saved model checkpoint (required for evaluate / predict)."
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to test data directory (required for predict)."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {config['device']}")

    if args.mode == "train":
        run_training(config)
    elif args.mode == "evaluate":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for evaluate mode.")
        config["MODEL_PATH"] = args.checkpoint
        # Load and evaluate
        from src.evaluate import evaluate_from_checkpoint
        evaluate_from_checkpoint(config)
    elif args.mode == "predict":
        if not args.checkpoint or not args.data:
            raise ValueError("--checkpoint and --data are required for predict mode.")
        run_predict(config, args.checkpoint, args.data)


if __name__ == "__main__":
    main()
