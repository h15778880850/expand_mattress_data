import sys
import argparse
sys.path.insert(0, ".")

import torch
from src.utils.helpers import Config, set_seed, get_device, Logger, CLASS_NAMES
from src.data.dataset import create_dataloaders
from src.models.cnn1d import CNN1D
from src.trainer.trainer import Trainer
from src.evaluation.metrics import ClassificationMetrics
from src.evaluation.visualizer import Visualizer
from src.ablation.ablation import AblationStudy


def train(args, config):
    device = get_device()
    print(f"Using device: {device}")

    # data
    print("Loading data...")
    train_loader, val_loader, test_loader, test_extra_loader = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}, Extra test batches: {len(test_extra_loader)}")

    # model
    model = CNN1D(
        in_channels=config.get("model", "in_channels"),
        num_classes=config.get("model", "num_classes"),
        channels=config.get("model", "cnn_channels"),
        kernels=config.get("model", "cnn_kernels"),
        strides=config.get("model", "cnn_strides"),
        dropout=config.get("model", "dropout"),
    )
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # train
    trainer = Trainer(model, config, device)
    trainer.fit(train_loader, val_loader, config.get("training", "epochs"))

    # evaluate on held-out test set
    print("\nEvaluating on held-out test set...")
    test_results = trainer.evaluate(test_loader)
    metrics = ClassificationMetrics(
        test_results["labels"], test_results["predictions"],
        y_score=test_results["logits"],
    )

    # evaluate on independent test_data
    print("Evaluating on independent test_data...")
    extra_results = trainer.evaluate(test_extra_loader)
    metrics_extra = ClassificationMetrics(
        extra_results["labels"], extra_results["predictions"],
        y_score=extra_results["logits"],
    )

    # print results
    print("\n" + "=" * 60)
    print("Results on held-out test set (from train_data)")
    print("=" * 60)
    print(metrics.summary())

    print("\n" + "=" * 60)
    print("Results on independent test_data")
    print("=" * 60)
    print(metrics_extra.summary())

    # visualize
    viz = Visualizer()
    viz.plot_confusion_matrix(metrics.confusion_matrix(), "confusion_matrix_test.png")
    viz.plot_confusion_matrix(metrics_extra.confusion_matrix(), "confusion_matrix_extra.png")

    roc_curves = metrics.roc_curves()
    if roc_curves:
        viz.plot_roc_curves(roc_curves, "roc_curves_test.png")
    roc_curves_extra = metrics_extra.roc_curves()
    if roc_curves_extra:
        viz.plot_roc_curves(roc_curves_extra, "roc_curves_extra.png")

    viz.plot_per_class_bar(metrics.per_class_metrics(), "per_class_metrics_test.png")

    # t-SNE
    features = test_results["logits"]
    viz.plot_tsne(features, test_results["labels"], "tsne_logits.png")

    # bootstrap confidence intervals
    print("\nBootstrapping confidence intervals (98% might take a moment)...")
    acc_lower, acc_upper = metrics.bootstrap_ci("overall_accuracy")
    f1_lower, f1_upper = metrics.bootstrap_ci("macro_f1")
    print(f"Test Accuracy 95% CI: [{acc_lower:.4f}, {acc_upper:.4f}]")
    print(f"Macro F1 95% CI:      [{f1_lower:.4f}, {f1_upper:.4f}]")

    print(f"\nFigures saved to {viz.save_dir}")


def run_ablation(args, config):
    print("Running ablation study A1...")
    study = AblationStudy(config)
    study.run_A1(epochs=config.get("training", "epochs"))
    print("Ablation complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--mode", type=str, choices=["train", "ablation"],
                        default="train")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = Config(args.config)
    if args.seed is not None:
        config.get("data", "random_seed", default=args.seed)

    set_seed(config.get("data", "random_seed", default=42))

    if args.mode == "train":
        train(args, config)
    elif args.mode == "ablation":
        run_ablation(args, config)


if __name__ == "__main__":
    main()
