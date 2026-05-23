import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve,
)
from src.utils.helpers import CLASS_NAMES


class ClassificationMetrics:
    def __init__(self, y_true, y_pred, y_score=None):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_score = np.array(y_score) if y_score is not None else None

    def overall_accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def macro_f1(self):
        return f1_score(self.y_true, self.y_pred, average="macro")

    def weighted_f1(self):
        return f1_score(self.y_true, self.y_pred, average="weighted")

    def per_class_metrics(self):
        precision = precision_score(self.y_true, self.y_pred, average=None)
        recall = recall_score(self.y_true, self.y_pred, average=None)
        f1 = f1_score(self.y_true, self.y_pred, average=None)
        cm = confusion_matrix(self.y_true, self.y_pred)

        results = {}
        for i, name in enumerate(CLASS_NAMES):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fp - fn
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            results[name] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1_score": f1[i],
                "specificity": specificity,
                "support": cm[i, :].sum(),
            }
        return results

    def confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def roc_auc(self):
        if self.y_score is None:
            return None
        n_classes = len(CLASS_NAMES)
        y_true_onehot = np.eye(n_classes)[self.y_true]
        try:
            auc = roc_auc_score(y_true_onehot, self.y_score, average="macro",
                                multi_class="ovr")
            return auc
        except ValueError:
            return None

    def roc_curves(self):
        if self.y_score is None:
            return None
        n_classes = len(CLASS_NAMES)
        y_true_onehot = np.eye(n_classes)[self.y_true]
        curves = {}
        for i, name in enumerate(CLASS_NAMES):
            fpr, tpr, thresholds = roc_curve(y_true_onehot[:, i], self.y_score[:, i])
            auc = roc_auc_score(y_true_onehot[:, i], self.y_score[:, i])
            curves[name] = {"fpr": fpr, "tpr": tpr, "auc": auc, "thresholds": thresholds}
        return curves

    def summary(self):
        metrics = self.per_class_metrics()
        rows = []
        rows.append(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Specificity':>12} {'Support':>8}")
        rows.append("-" * 70)
        for name in CLASS_NAMES:
            m = metrics[name]
            rows.append(
                f"{name:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                f"{m['f1_score']:>10.4f} {m['specificity']:>12.4f} {m['support']:>8}"
            )
        rows.append("-" * 70)
        rows.append(f"{'Accuracy':<20} {self.overall_accuracy():>10.4f}")
        rows.append(f"{'Macro F1':<20} {self.macro_f1():>10.4f}")
        rows.append(f"{'Weighted F1':<20} {self.weighted_f1():>10.4f}")
        if self.y_score is not None:
            auc = self.roc_auc()
            if auc is not None:
                rows.append(f"{'ROC-AUC (macro)':<20} {auc:>10.4f}")
        return "\n".join(rows)

    def bootstrap_ci(self, metric_fn, n_iterations=1000, ci=0.95):
        np.random.seed(42)
        n = len(self.y_true)
        scores = []
        for _ in range(n_iterations):
            indices = np.random.choice(n, n, replace=True)
            y_true_b = self.y_true[indices]
            y_pred_b = self.y_pred[indices]
            y_score_b = self.y_score[indices] if self.y_score is not None else None
            m = ClassificationMetrics(y_true_b, y_pred_b, y_score_b)
            scores.append(getattr(m, metric_fn)())
        scores = sorted(scores)
        alpha = 1 - ci
        lower = scores[int(n_iterations * alpha / 2)]
        upper = scores[int(n_iterations * (1 - alpha / 2))]
        return lower, upper
