"""Non-maximum suppression for object detections."""

import numpy as np


def non_max_suppression(boxes, probs=None, overlap_thresh=0.3, **legacy_kwargs):
    """Suppress lower-scoring boxes whose IoU exceeds ``overlap_thresh``.

    Boxes use ``[x1, y1, x2, y2]`` with exclusive bottom-right coordinates.
    ``overlapThresh`` remains accepted for compatibility with earlier code.
    """
    if "overlapThresh" in legacy_kwargs:
        overlap_thresh = legacy_kwargs.pop("overlapThresh")
    if legacy_kwargs:
        raise TypeError(f"Unexpected arguments: {', '.join(legacy_kwargs)}")
    if not 0 <= overlap_thresh <= 1:
        raise ValueError("overlap_thresh must be between 0 and 1")

    boxes_array = np.asarray(boxes, dtype=float)
    if boxes_array.size == 0:
        return np.empty((0, 4), dtype=int)
    if boxes_array.ndim != 2 or boxes_array.shape[1] != 4:
        raise ValueError("boxes must have shape (n, 4)")

    scores = (
        np.asarray(probs, dtype=float)
        if probs is not None
        else np.arange(len(boxes_array), dtype=float)
    )
    if scores.shape != (len(boxes_array),):
        raise ValueError("probs must contain one score per box")

    x1, y1, x2, y2 = boxes_array.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = np.argsort(scores)
    keep = []

    while order.size:
        current = order[-1]
        keep.append(current)
        remaining = order[:-1]
        if not remaining.size:
            break

        inter_x1 = np.maximum(x1[current], x1[remaining])
        inter_y1 = np.maximum(y1[current], y1[remaining])
        inter_x2 = np.minimum(x2[current], x2[remaining])
        inter_y2 = np.minimum(y2[current], y2[remaining])
        intersection = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= overlap_thresh]

    return boxes_array[keep].astype(int)
