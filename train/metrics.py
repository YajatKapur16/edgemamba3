import numpy as np
from sklearn.metrics import (
    average_precision_score, 
    roc_auc_score, 
    mean_absolute_error,
    classification_report,
    mean_squared_error,
    r2_score
)


def compute_metric(preds: np.ndarray, labels: np.ndarray, metric: str) -> float:
    """
    Compute evaluation metric.

    Args:
        preds:  model outputs (logits for classification, values for regression)
        labels: ground truth
        metric: "ap" | "mae" | "auroc"
    """
    if metric == "ap":
        # Apply sigmoid to logits for AP
        preds_prob = 1 / (1 + np.exp(-preds))
        if labels.ndim == 1:
            # Single-label binary
            return float(average_precision_score(labels, preds_prob))
        else:
            # Multi-label (Peptides-func: 10 classes)
            # Skip columns where all labels are the same
            aps = []
            for i in range(labels.shape[1]):
                if labels[:, i].sum() > 0 and labels[:, i].sum() < len(labels):
                    aps.append(average_precision_score(labels[:, i], preds_prob[:, i]))
            return float(np.mean(aps)) if aps else 0.0

    elif metric == "auroc":
        preds_prob = 1 / (1 + np.exp(-preds))
        return float(roc_auc_score(labels, preds_prob))

    elif metric == "mae":
        return float(mean_absolute_error(labels, preds))

    else:
        raise ValueError(f"Unknown metric: {metric}")


def generate_eval_report(preds: np.ndarray, labels: np.ndarray, task_type: str, filepath: str):
    """Generates detailed reports."""
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    with open(filepath, "w") as f:
        if task_type == "classification":
            preds_prob = 1 / (1 + np.exp(-preds))
            preds_bin = (preds_prob > 0.5).astype(int)
            
            try:
                auroc = roc_auc_score(labels, preds_prob, average="macro")
            except:
                auroc = 0.0
                
            clf_rep = classification_report(labels, preds_bin, zero_division=0)
            
            f.write("=== Classification Evaluation Report ===\n\n")
            f.write(f"AUROC (Macro): {auroc:.4f}\n\n")
            f.write("Detailed Classification Report (Threshold=0.5):\n")
            f.write(clf_rep)
            
        elif task_type == "regression":
            mae = mean_absolute_error(labels, preds)
            mse = mean_squared_error(labels, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(labels, preds)
            
            f.write("=== Regression Evaluation Report ===\n\n")
            f.write(f"MAE:  {mae:.4f}\n")
            f.write(f"MSE:  {mse:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"R2:   {r2:.4f}\n")

