import os
import csv
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from src.utils.helpers import set_seed, Logger


class Trainer:
    def __init__(self, model, config, device, log_dir="experiments"):
        self.model = model.to(device)
        self.config = config
        self.device = device

        lr = config.get("training", "lr", default=0.001)
        weight_decay = config.get("training", "weight_decay", default=0.0001)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.get("training", "epochs", default=100),
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(log_dir) / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.checkpoint_dir = run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
        self.logger = Logger(str(run_dir / "logs"))

        self.early_stop_patience = config.get("training", "early_stop_patience", default=10)
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_model_state = None
        self.current_epoch = 0
        # per-epoch metrics for CSV export
        self.history = []

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for X, y in pbar:
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(X)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})

        return total_loss / total, 100.0 * correct / total

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for X, y in val_loader:
            X, y = X.to(self.device), y.to(self.device)
            outputs = self.model(X)
            loss = self.criterion(outputs, y)

            total_loss += loss.item() * len(X)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        return total_loss / total, 100.0 * correct / total

    def fit(self, train_loader, val_loader, epochs):
        set_seed(self.config.get("data", "random_seed", default=42))
        self.logger.write(f"Training started: {epochs} epochs, device={self.device}")

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step()

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Acc/train", train_acc, epoch)
            self.writer.add_scalar("Acc/val", val_acc, epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            log_msg = (
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
            )
            self.logger.write(log_msg)
            print(log_msg)
            self.history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 4),
                "lr": round(self.optimizer.param_groups[0]["lr"], 8),
            })

            # early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                self.patience_counter = 0
                ckpt_path = self.checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.best_model_state,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": self.config.to_dict(),
                }, ckpt_path)
                self.logger.write(f"  → Checkpoint saved: {ckpt_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    self.logger.write(f"Early stopping triggered at epoch {epoch}")
                    break

        # load best model
        self.model.load_state_dict(self.best_model_state)

        self.save_history_csv()
        self.logger.write(f"Training completed. Best val loss: {self.best_val_loss:.4f}")
        self.writer.close()

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_logits = []

        for X, y in test_loader:
            X = X.to(self.device)
            outputs = self.model(X)
            _, predicted = outputs.max(1)

            all_preds.append(predicted.cpu())
            all_labels.append(y)
            all_logits.append(outputs.cpu())

        return {
            "predictions": torch.cat(all_preds).numpy(),
            "labels": torch.cat(all_labels).numpy(),
            "logits": torch.cat(all_logits).numpy(),
        }

    def save_history_csv(self):
        csv_path = self.run_dir / "training_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr",
            ])
            w.writeheader()
            w.writerows(self.history)
        self.logger.write(f"Training metrics saved to {csv_path}")
        return csv_path

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.logger.write(f"Loaded checkpoint from {path}")
        return ckpt
