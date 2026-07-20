from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.metrics import (
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score,
)


def compute_metrics(y_true, y_pred, y_prob=None, multi_class=False):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    if multi_class:
        metrics["precision"] = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics["recall"] = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics["f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    else:
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
        if y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

        # Add confusion matrix metrics (assuming binary 0/1)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0
        else:
            metrics["fpr"] = 0.0
            metrics["fnr"] = 0.0

    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)

    return metrics
