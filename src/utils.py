"""
src/utils.py
------------
Shared utility functions: seeding, DataLoader factory, banner printing.
"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def get_dataloader(
    dataset,
    config: dict,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """
    Construct a DataLoader from a dataset and the project config dict.

    Parameters
    ----------
    dataset   : torch.utils.data.Dataset
    config    : dict   Must contain ``batch_size`` and ``num_workers``.
    shuffle   : bool
    drop_last : bool
    """
    return DataLoader(
        dataset,
        batch_size=config.get("batch_size", 1),
        shuffle=shuffle,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=drop_last,
    )


def print_banner(title: str) -> None:
    """Print a formatted section banner."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def count_parameters(model: torch.nn.Module) -> str:
    """Return a human-readable parameter count string."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total:,}  |  Trainable: {trainable:,}"
