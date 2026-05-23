import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from src.models.cnn1d import CNN1D
from src.trainer.trainer import Trainer
from src.data.dataset import create_dataloaders
from src.evaluation.metrics import ClassificationMetrics
from src.utils.helpers import set_seed, get_device, Logger


class AblationStudy:
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        self.results_dir = Path("experiments/ablation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = Logger(str(self.results_dir), name=f"ablation_{timestamp}")

    def _build_variant(self, variant_name, variant_config):
        if variant_name == "cnn1d_shallow":
            return CNN1D(
                in_channels=7, num_classes=5,
                channels=(64,), kernels=(7,), strides=(2,),
                dropout=0.5,
            )
        elif variant_name == "cnn1d_deep":
            return CNN1D(
                in_channels=7, num_classes=5,
                channels=(32, 64, 128, 256, 512),
                kernels=(7, 5, 5, 3, 3),
                strides=(2, 2, 2, 2, 2),
                dropout=0.5,
            )
        elif variant_name == "cnn1d_wide":
            return CNN1D(
                in_channels=7, num_classes=5,
                channels=(128, 256, 512),
                kernels=(7, 5, 3),
                strides=(2, 2, 2),
                dropout=0.5,
            )
        elif variant_name == "cnn1d_narrow":
            return CNN1D(
                in_channels=7, num_classes=5,
                channels=(32, 64, 128),
                kernels=(7, 5, 3),
                strides=(2, 2, 2),
                dropout=0.5,
            )
        elif variant_name == "cnn1d_nodropout":
            return CNN1D(
                in_channels=7, num_classes=5,
                channels=(64, 128, 256),
                kernels=(7, 5, 3),
                strides=(2, 2, 2),
                dropout=0.0,
            )
        else:  # cnn1d (default)
            return CNN1D(
                in_channels=7, num_classes=5,
                channels=(64, 128, 256),
                kernels=(7, 5, 3),
                strides=(2, 2, 2),
                dropout=0.5,
            )

    def run_experiment(self, variant_name, train_loader, val_loader, test_loader,
                       test_extra_loader, epochs=50):
        self.logger.write(f"Running ablation: {variant_name}")
        set_seed(self.config.get("data", "random_seed", default=42))

        model = self._build_variant(variant_name, self.config)
        trainer = Trainer(model, self.config, self.device)
        trainer.fit(train_loader, val_loader, epochs)

        # evaluate on held-out test set (from train_data)
        test_results = trainer.evaluate(test_loader)
        metrics_test = ClassificationMetrics(
            test_results["labels"], test_results["predictions"],
            y_score=test_results["logits"],
        )

        # evaluate on independent test_data
        extra_results = trainer.evaluate(test_extra_loader)
        metrics_extra = ClassificationMetrics(
            extra_results["labels"], extra_results["predictions"],
            y_score=extra_results["logits"],
        )

        return {
            "variant": variant_name,
            "val_loss": trainer.best_val_loss,
            "test_accuracy": metrics_test.overall_accuracy(),
            "test_macro_f1": metrics_test.macro_f1(),
            "extra_accuracy": metrics_extra.overall_accuracy(),
            "extra_macro_f1": metrics_extra.macro_f1(),
            "params": sum(p.numel() for p in model.parameters()),
        }

    def run_A1(self, epochs=50):
        self.logger.write("=" * 60)
        self.logger.write("Ablation A1: Model Architecture Comparison")
        self.logger.write("=" * 60)

        train_loader, val_loader, test_loader, test_extra_loader = create_dataloaders(
            self.config,
        )

        variants = ["cnn1d", "cnn1d_shallow", "cnn1d_deep",
                     "cnn1d_wide", "cnn1d_narrow", "cnn1d_nodropout"]
        results = []

        for variant in variants:
            result = self.run_experiment(
                variant, train_loader, val_loader, test_loader,
                test_extra_loader, epochs,
            )
            results.append(result)
            self.logger.write(
                f"  {variant}: test_acc={result['test_accuracy']:.4f}, "
                f"extra_acc={result['extra_accuracy']:.4f}, "
                f"params={result['params']:,}"
            )

        # summary table
        self.logger.write("\n" + "=" * 60)
        self.logger.write("Ablation A1 Summary")
        self.logger.write("=" * 60)
        header = f"{'Variant':<20} {'Params':>10} {'Test Acc':>10} {'Test F1':>10} {'Extra Acc':>10} {'Extra F1':>10}"
        self.logger.write(header)
        self.logger.write("-" * 70)
        for r in results:
            self.logger.write(
                f"{r['variant']:<20} {r['params']:>10,} {r['test_accuracy']:>10.4f} "
                f"{r['test_macro_f1']:>10.4f} {r['extra_accuracy']:>10.4f} "
                f"{r['extra_macro_f1']:>10.4f}"
            )

        # save to CSV
        import pandas as pd
        df = pd.DataFrame(results)
        csv_path = self.results_dir / "ablation_A1_results.csv"
        df.to_csv(csv_path, index=False)
        self.logger.write(f"\nResults saved to {csv_path}")

        return results
