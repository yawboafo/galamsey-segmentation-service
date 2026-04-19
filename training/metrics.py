import evaluate
import numpy as np
import torch
from typing import Dict

# Load the mIoU metric from the evaluate library
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    """
    Computes IoU, Precision, Recall, and F1-score for binary segmentation.
    """
    logits, labels = eval_pred
    
    # Pre-processing logits to get predicted class
    # logits shape: [batch_size, num_classes, height, width]
    # We take argmax over the class dimension
    predictions = np.argmax(logits, axis=1)
    
    # compute mIoU using evaluate library
    # labels shape: [batch_size, height, width]
    results = metric.compute(
        predictions=predictions,
        references=labels,
        num_labels=2, # Binary: 0=Background, 1=Galamsey
        ignore_index=255, # Standard ignore index
        reduce_labels=False,
    )
    
    # Extract specific metrics for the 'Galamsey' class (index 1)
    # per_category_iou is usually a list/array [bg_iou, galamsey_iou]
    per_category_iou = results.get("per_category_iou", [0, 0])
    galamsey_iou = per_category_iou[1] if len(per_category_iou) > 1 else 0
    
    # Precision, Recall, F1 for the Galamsey class
    # We can compute these from the confusion matrix or manually
    # Note: evaluate mean_iou provides some but not all.
    # Alternatively, we can use simple numpy if needed.
    
    # Flatten inputs for simple precision/recall calculation
    y_true = labels.flatten()
    y_pred = predictions.flatten()
    
    # Mask out ignore index if any
    mask = (y_true != 255)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "iou": galamsey_iou,
        "mean_iou": results.get("mean_iou", 0),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": results.get("overall_accuracy", 0)
    }
