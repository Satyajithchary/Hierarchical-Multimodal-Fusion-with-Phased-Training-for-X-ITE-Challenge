"""
src/trainer.py
--------------
Three-phase training curriculum for the Hierarchical Multimodal model.

Phase 1 — Unimodal Focus
    Only unimodal processors and their dedicated classifiers are trained.
    Allows each backbone to learn stable, modality-specific representations
    without interference from fusion objectives.

Phase 2 — Fusion Focus
    Unimodal processors are frozen.  Pairwise fusion classifiers, the
    cross-modal attention module, and the full-fusion classifier are trained.

Phase 3 — Full Fine-tuning
    All parameters are unfrozen and fine-tuned end-to-end at a lower LR.
"""

import os
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ChainedScheduler, CosineAnnealingLR, LinearLR

from src.losses import FocalLoss, HierarchicalContrastiveLoss


class EnhancedTrainer:
    """
    Manages the full three-phase training loop.

    Parameters
    ----------
    model       : HierarchicalPainRecognition
    train_loader, val_loader, test_loader : DataLoader
    config      : dict
        Must contain the keys defined in ``configs/default.yaml``.
    """

    def __init__(self, model, train_loader, val_loader, test_loader, config: dict):
        self.config       = config
        self.device       = config["device"]
        self.model        = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda"))

        self.cls_criterion = (
            FocalLoss(
                alpha=config.get("focal_loss_alpha", 1.0),
                gamma=config.get("focal_loss_gamma", 2.0),
            )
            if config.get("use_focal_loss", True)
            else torch.nn.CrossEntropyLoss()
        )
        self.con_criterion  = HierarchicalContrastiveLoss()
        self.con_weight     = config.get("contrastive_loss_weight", 0.2)
        self.unc_weight     = config.get("uncertainty_loss_weight", 0.1)
        self.grad_accum     = config.get("gradient_accumulation_steps", 16)

        self.history: dict = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.best_val_acc    = 0.0
        self.best_model_state = None

        os.makedirs(config.get("output_dir", "results"), exist_ok=True)

    # ── Phase management ──────────────────────────────────────────────────────

    def _set_trainable(self, phase: int) -> None:
        print(f"\n─── Phase {phase}: setting trainable parameters ───")
        for p in self.model.parameters():
            p.requires_grad = False

        if phase == 1:
            for m in self.model.MODALITIES:
                for p in getattr(self.model, f"{m}_processor").parameters():
                    p.requires_grad = True
                for p in self.model.classifiers[m].parameters():
                    p.requires_grad = True

        elif phase == 2:
            for p in self.model.pairwise_fusions.parameters():
                p.requires_grad = True
            for p in self.model.cross_modal_attention.parameters():
                p.requires_grad = True
            for p in self.model.full_fusion_classifier.parameters():
                p.requires_grad = True

        elif phase == 3:
            for p in self.model.parameters():
                p.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"    Trainable parameters: {trainable:,}")

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> None:
        lr_base       = self.config["learning_rate"]
        phase_epochs  = [
            self.config["epochs_phase1"],
            self.config["epochs_phase2"],
            self.config["epochs_phase3"],
        ]

        for phase, n_epochs in enumerate(phase_epochs, start=1):
            if n_epochs == 0:
                continue

            self._set_trainable(phase)

            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr_base / (2 ** (phase - 1)),
                weight_decay=self.config.get("weight_decay", 1e-4),
            )

            total_steps = n_epochs * len(self.train_loader)
            warmup_steps = max(1, int(total_steps * 0.1))
            sched_warmup = LinearLR(
                self.optimizer, start_factor=0.1, total_iters=warmup_steps
            )
            sched_main = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=1e-6,
            )
            self.scheduler = ChainedScheduler([sched_warmup, sched_main])

            patience_counter = 0
            for epoch in range(1, n_epochs + 1):
                train_loss, train_acc = self._train_epoch(phase)
                val_loss, val_acc, *_ = self.validate(self.val_loader)

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                print(
                    f"P{phase} | E{epoch:03d}/{n_epochs} | "
                    f"Train  loss={train_loss:.4f}  acc={train_acc:.2f}% | "
                    f"Val    loss={val_loss:.4f}  acc={val_acc:.2f}%"
                )

                if val_acc > self.best_val_acc:
                    self.best_val_acc    = val_acc
                    self.best_model_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                    patience_counter = 0
                    self._save_checkpoint()
                    print(f"    ✓ New best validation accuracy: {val_acc:.2f}%")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.get("patience", 7):
                        print(f"    Early stopping triggered in Phase {phase}.")
                        break

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nLoaded best model (val acc = {self.best_val_acc:.2f}%)")

    # ── Single epoch ──────────────────────────────────────────────────────────

    def _train_epoch(self, phase: int):
        self.model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            physio = batch["physio"].to(self.device)
            video  = {k: v.to(self.device) for k, v in batch["video"].items()}
            audio  = batch["audio"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                logits, features, uncertainty = self.model(
                    physio, video, audio, return_uncertainty=True
                )

                loss = torch.tensor(0.0, device=self.device)
                if phase == 1:
                    for m in self.model.MODALITIES:
                        loss = loss + self.cls_criterion(logits[m], labels)
                elif phase == 2:
                    for k in self.model.pairwise_fusions:
                        loss = loss + self.cls_criterion(logits[k], labels)
                    loss = loss + self.cls_criterion(logits["full"], labels)
                else:
                    loss = loss + self.cls_criterion(logits["ensemble"], labels)

                loss = (
                    loss
                    + self.con_weight * self.con_criterion(features, labels)
                    + self.unc_weight * uncertainty.mean()
                )
                loss = loss / self.grad_accum

            self.scaler.scale(loss).backward()

            total_loss += loss.item() * self.grad_accum
            ref_logits  = logits["ensemble"] if phase == 3 else logits["full"]
            preds       = ref_logits.argmax(dim=1)
            total_correct += preds.eq(labels).sum().item()
            total_n       += labels.size(0)

            if (step + 1) % self.grad_accum == 0 or (step + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()), 1.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

        # Free GPU memory explicitly
        del physio, video, audio, labels, logits, features, uncertainty, loss
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return total_loss / len(self.train_loader), 100.0 * total_correct / total_n

    # ── Validation / Evaluation ───────────────────────────────────────────────

    def validate(self, loader):
        """
        Returns
        -------
        avg_loss, accuracy, all_preds, all_labels, all_probs, all_uncertainties
        """
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

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save_checkpoint(self) -> None:
        out_dir  = self.config.get("output_dir", "results")
        out_name = self.config.get("checkpoint_name", "hierarchical_pain_model_optimized.pth")
        path     = os.path.join(out_dir, out_name)
        torch.save(
            {"model_state_dict": self.best_model_state, "config": self.config},
            path,
        )
        print(f"    Checkpoint saved → {path}")
