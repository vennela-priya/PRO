"""segmentation/metrics.py — Segmentation evaluation metrics."""
import torch


def compute_all_metrics(logits, targets, threshold=0.45):
    """
    Compute Dice, IoU, Precision, Recall from raw logits.

    Parameters
    ----------
    logits   : (B,1,H,W) raw model output — sigmoid applied internally
    targets  : (B,1,H,W) binary float {0,1}
    threshold: probability cut-off

    Returns
    -------
    dict — dice, iou, precision, recall (floats in [0,1])
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits.float())
        pred  = (probs >= threshold).float().view(-1)
        targ  = targets.float().view(-1)

        TP = (pred * targ).sum().item()
        FP = (pred * (1 - targ)).sum().item()
        FN = ((1 - pred) * targ).sum().item()

        s         = 1e-6
        dice      = (2 * TP + s) / (2 * TP + FP + FN + s)
        iou       = (TP + s)     / (TP + FP + FN + s)
        precision = (TP + s)     / (TP + FP + s)
        recall    = (TP + s)     / (TP + FN + s)

    return {
        'dice':      float(dice),
        'iou':       float(iou),
        'precision': float(precision),
        'recall':    float(recall),
    }


# Aliases kept for backward compatibility
def dice_score(pred, target, threshold=0.45, smooth=1.0):
    return compute_all_metrics(pred, target, threshold)['dice']


def iou_score(pred, target, threshold=0.45, smooth=1.0):
    return compute_all_metrics(pred, target, threshold)['iou']
