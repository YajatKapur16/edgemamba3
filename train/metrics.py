import numpy as np
from sklearn.metrics import (
    average_precision_score, 
    roc_auc_score, 
    mean_absolute_error,
    classification_report,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    median_absolute_error,
    precision_recall_curve,
    auc
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


def compute_all_metrics(preds: np.ndarray, labels: np.ndarray, task_type: str) -> dict:
    """
    Compute a comprehensive set of evaluation metrics based on task type.
    
    Args:
        preds:  model outputs (logits for classification, values for regression)
        labels: ground truth
        task_type: "classification" or "regression"
        
    Returns:
        Dictionary mapping metric names to their computed values.
    """
    metrics_dict = {}
    
    if task_type == "classification":
        preds_prob = 1 / (1 + np.exp(-preds))
        preds_bin = (preds_prob > 0.5).astype(int)
        
        # AP (AUPRC)
        if labels.ndim == 1:
            metrics_dict["ap"] = average_precision_score(labels, preds_prob)
            precision, recall, _ = precision_recall_curve(labels, preds_prob)
            metrics_dict["auprc"] = auc(recall, precision)
            metrics_dict["auroc"] = roc_auc_score(labels, preds_prob)
            metrics_dict["accuracy"] = accuracy_score(labels, preds_bin)
            metrics_dict["f1"] = f1_score(labels, preds_bin, average="macro")
        else:
            aps, auprcs, aurocs, f1s = [], [], [], []
            for i in range(labels.shape[1]):
                if labels[:, i].sum() > 0 and labels[:, i].sum() < len(labels):
                    aps.append(average_precision_score(labels[:, i], preds_prob[:, i]))
                    prec, rec, _ = precision_recall_curve(labels[:, i], preds_prob[:, i])
                    auprcs.append(auc(rec, prec))
                    try:
                        aurocs.append(roc_auc_score(labels[:, i], preds_prob[:, i]))
                    except ValueError:
                        pass
                    f1s.append(f1_score(labels[:, i], preds_bin[:, i], zero_division=0))
            metrics_dict["ap"] = np.mean(aps) if aps else 0.0
            metrics_dict["auprc"] = np.mean(auprcs) if auprcs else 0.0
            metrics_dict["auroc"] = np.mean(aurocs) if aurocs else 0.0
            metrics_dict["f1"] = np.mean(f1s) if f1s else 0.0
            # multi-label exact match accuracy can be harsh, but we provide it
            metrics_dict["accuracy"] = accuracy_score(labels, preds_bin)
            
    elif task_type == "regression":
        metrics_dict["mae"] = mean_absolute_error(labels, preds)
        metrics_dict["mse"] = mean_squared_error(labels, preds)
        metrics_dict["rmse"] = np.sqrt(metrics_dict["mse"])
        metrics_dict["r2"] = r2_score(labels, preds)
        metrics_dict["medae"] = median_absolute_error(labels, preds)
        
    return {k: float(v) for k, v in metrics_dict.items()}


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

