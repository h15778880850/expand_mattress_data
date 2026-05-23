import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
from src.utils.helpers import CLASS_NAMES

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 10,
})


class Visualizer:
    def __init__(self, save_dir="experiments/figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(self, cm, normalize=True, filename="confusion_matrix.png"):
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            fmt = ".2f"
            vmin, vmax = 0, 1
        else:
            fmt = "d"
            vmin, vmax = 0, cm.max()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    vmin=vmin, vmax=vmax, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        fig.savefig(self.save_dir / filename)
        plt.close(fig)

    def plot_roc_curves(self, roc_curves, filename="roc_curves.png"):
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, curve in roc_curves.items():
            ax.plot(curve["fpr"], curve["tpr"],
                    label=f"{name} (AUC={curve['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(self.save_dir / filename)
        plt.close(fig)

    def plot_training_curves(self, train_losses, val_losses, train_accs, val_accs,
                              filename="training_curves.png"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, label="Train Loss")
        ax1.plot(epochs, val_losses, label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curves")
        ax1.legend()

        ax2.plot(epochs, train_accs, label="Train Acc")
        ax2.plot(epochs, val_accs, label="Val Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Accuracy Curves")
        ax2.legend()

        plt.tight_layout()
        fig.savefig(self.save_dir / filename)
        plt.close(fig)

    def plot_tsne(self, features, labels, filename="tsne.png"):
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb = tsne.fit_transform(features)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = sns.color_palette("husl", len(CLASS_NAMES))
        for i, name in enumerate(CLASS_NAMES):
            mask = labels == i
            ax.scatter(emb[mask, 0], emb[mask, 1], c=[colors[i]], label=name,
                       alpha=0.6, s=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("Feature Visualization (t-SNE)")
        ax.legend(markerscale=3)
        plt.tight_layout()
        fig.savefig(self.save_dir / filename)
        plt.close(fig)

    def plot_per_class_bar(self, metrics_dict, filename="per_class_metrics.png"):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(CLASS_NAMES))
        width = 0.25

        precisions = [metrics_dict[n]["precision"] for n in CLASS_NAMES]
        recalls = [metrics_dict[n]["recall"] for n in CLASS_NAMES]
        f1s = [metrics_dict[n]["f1_score"] for n in CLASS_NAMES]

        ax.bar(x - width, precisions, width, label="Precision")
        ax.bar(x, recalls, width, label="Recall")
        ax.bar(x + width, f1s, width, label="F1-Score")

        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, rotation=15)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Performance")
        ax.legend()
        plt.tight_layout()
        fig.savefig(self.save_dir / filename)
        plt.close(fig)
