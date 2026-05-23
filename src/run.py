import sys
import csv
import argparse
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
import torch
from src.utils.helpers import Config, set_seed, get_device, CLASS_NAMES
from src.data.dataset import create_kfold_dataloaders
from src.data.transforms import build_transform
from src.models.cnn1d import CNN1D
from src.trainer.trainer import Trainer
from src.evaluation.metrics import ClassificationMetrics
from src.evaluation.visualizer import Visualizer


def train_kfold(config):
    device = get_device()
    print(f"Using device: {device}")

    n_splits = config.get("kfold", "n_splits", default=5)
    transform = build_transform(config)
    if transform is not None:
        print(f"Using data augmentation: {transform.transforms}")
    else:
        print("No data augmentation")

    # store per-fold results
    fold_results = []

    for fold in range(n_splits):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"{'=' * 60}")

        train_loader, val_loader, splits, _ = create_kfold_dataloaders(
            config, fold, transform=transform,
        )
        n_train = len(splits[0])
        n_val = len(splits[1])
        print(f"Train samples: {n_train}, Val samples: {n_val}")

        model = CNN1D(
            in_channels=config.get("model", "in_channels"),
            num_classes=config.get("model", "num_classes"),
            channels=config.get("model", "cnn_channels"),
            kernels=config.get("model", "cnn_kernels"),
            strides=config.get("model", "cnn_strides"),
            dropout=config.get("model", "dropout"),
        )
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        trainer = Trainer(
            model, config, device,
            log_dir=f"experiments/fold_{fold + 1}",
        )
        trainer.fit(
            train_loader, val_loader,
            config.get("training", "epochs"),
        )

        # evaluate
        results = trainer.evaluate(val_loader)
        metrics = ClassificationMetrics(
            results["labels"], results["predictions"],
            y_score=results["logits"],
        )

        fold_metrics = {
            "fold": fold + 1,
            "val_loss": trainer.best_val_loss,
            "accuracy": metrics.overall_accuracy(),
            "macro_f1": metrics.macro_f1(),
            "weighted_f1": metrics.weighted_f1(),
            "stopped_epoch": trainer.current_epoch,
        }
        fold_results.append(fold_metrics)

        print(f"  Fold {fold + 1} - Acc: {fold_metrics['accuracy']:.4f}, "
              f"Macro F1: {fold_metrics['macro_f1']:.4f}, "
              f"Best Val Loss: {fold_metrics['val_loss']:.4f}")

        # per-fold confusion matrix
        viz = Visualizer()
        viz.plot_confusion_matrix(
            metrics.confusion_matrix(),
            f"fold{fold + 1}_confusion_matrix.png",
        )

    # aggregate results
    print(f"\n{'=' * 60}")
    print("K-Fold Cross Validation Results")
    print(f"{'=' * 60}")

    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["macro_f1"] for r in fold_results]
    wf1s = [r["weighted_f1"] for r in fold_results]

    header = f"{'Fold':>6} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12} {'Best Loss':>10}"
    print(header)
    print("-" * 50)
    for r in fold_results:
        print(f"{r['fold']:>6} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} "
              f"{r['weighted_f1']:>12.4f} {r['val_loss']:>10.4f}")
    print("-" * 50)
    print(f"{'Mean':>6} {np.mean(accs):>10.4f} {np.mean(f1s):>10.4f} "
          f"{np.mean(wf1s):>12.4f}")
    print(f"{'Std':>6} {np.std(accs):>10.4f} {np.std(f1s):>10.4f} "
          f"{np.std(wf1s):>12.4f}")

    # save aggregated results
    result_dir = Path("experiments")
    result_dir.mkdir(exist_ok=True)
    csv_path = result_dir / "kfold_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "fold", "accuracy", "macro_f1", "weighted_f1", "val_loss", "stopped_epoch",
        ])
        w.writeheader()
        for r in fold_results:
            w.writerow(r)
        # summary row
        w.writerow({
            "fold": "mean",
            "accuracy": np.mean(accs),
            "macro_f1": np.mean(f1s),
            "weighted_f1": np.mean(wf1s),
        })
        w.writerow({
            "fold": "std",
            "accuracy": np.std(accs),
            "macro_f1": np.std(f1s),
            "weighted_f1": np.std(wf1s),
        })
    print(f"\nResults saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = Config(args.config)
    if args.seed is not None:
        config.get("data", "random_seed", default=args.seed)

    set_seed(config.get("data", "random_seed", default=42))
    train_kfold(config)


if __name__ == "__main__":
    main()
