"""
Object Detection Concepts — The Building Blocks

This module implements the foundational components of modern object detectors
(YOLO, Faster R-CNN, SSD) from scratch. Understanding these concepts is essential
before working with full detection frameworks.

Key Concepts:
    1. Anchor Boxes — Pre-defined boxes tiled across the image
    2. IoU (Intersection over Union) — Measuring box overlap
    3. NMS (Non-Maximum Suppression) — Removing duplicate detections

Why Anchors?
    Instead of sliding a classifier across every pixel at every scale (slow!),
    modern detectors place a fixed grid of "anchor boxes" and predict:
    - Is there an object here? (classification)
    - How should this anchor shift to fit the object? (regression)

    YOLO: 3 anchors per grid cell at 3 scales = efficient!
    Faster R-CNN: 9 anchors per position (3 scales × 3 aspect ratios)

Training Pipeline (conceptual — not implemented here):
    1. Generate anchors across the image
    2. Match each ground truth box to its best anchor (by IoU)
    3. For matched anchors: predict box offsets (Δx, Δy, Δw, Δh) + class
    4. Loss = localization_loss + classification_loss
    5. At inference: apply offsets to anchors, then NMS to remove duplicates

Robotics Connection:
    Detection → Localization → Manipulation
    "What objects?" → "Where are they?" → "How do I grab them?"
"""

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Tuple, Optional

# Type alias for bounding boxes: (x_min, y_min, x_max, y_max)
Box = Tuple[float, float, float, float]


# =============================================================================
# ANCHOR BOX GENERATION
# =============================================================================


def generate_anchors(
    center_x: float,
    center_y: float,
    scales: List[float],
    aspect_ratios: List[float],
) -> np.ndarray:
    """
    Generate anchor boxes centered at (center_x, center_y) with various scales and ratios.

    Anchors are the "starting guesses" for where objects might be. The detector
    then predicts small adjustments (Δx, Δy, Δw, Δh) to refine each anchor.

    Args:
        center_x: X coordinate of anchor center
        center_y: Y coordinate of anchor center
        scales: List of anchor sizes (e.g., [32, 64, 128] pixels)
        aspect_ratios: List of width/height ratios (e.g., [0.5, 1.0, 2.0])

    Returns:
        anchors: Array of shape (num_anchors, 4) where each row is
                 (x_min, y_min, x_max, y_max)

    Example:
        For scales=[64, 128] and aspect_ratios=[0.5, 1.0, 2.0]:
        - 64×0.5 → tall thin box (w=45, h=90)
        - 64×1.0 → square box (w=64, h=64)
        - 64×2.0 → wide flat box (w=90, h=45)
        - 128×0.5, 128×1.0, 128×2.0 → same shapes but bigger

    Math:
        For scale s and aspect ratio r (width/height):
        - width = s * sqrt(r)
        - height = s / sqrt(r)
        This ensures area ≈ s² regardless of aspect ratio.
    """
    # TODO(human): Generate anchor boxes
    #
    # Your task: Create anchors for all scale/ratio combinations (~12 lines)
    #
    # anchors = []
    #
    # for scale in scales:
    #     for ratio in aspect_ratios:
    #         # Compute width and height
    #         # width = scale * sqrt(ratio)
    #         # height = scale / sqrt(ratio)
    #         #
    #         # Compute box corners from center
    #         # x_min = center_x - width / 2
    #         # y_min = center_y - height / 2
    #         # x_max = center_x + width / 2
    #         # y_max = center_y + height / 2
    #         #
    #         # anchors.append([x_min, y_min, x_max, y_max])
    #
    # return np.array(anchors)
    #

    anchors = []

    for scale in scales:
        for ratio in aspect_ratios:
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)

            x_min = center_x - width / 2
            y_min = center_y - height / 2
            x_max = center_x + width / 2
            y_max = center_y + height / 2  # Bug fix: was width, should be height

            anchors.append([x_min, y_min, x_max, y_max])

    return np.array(anchors)


def generate_anchor_grid(
    image_size: Tuple[int, int],
    grid_size: Tuple[int, int],
    scales: List[float],
    aspect_ratios: List[float],
) -> np.ndarray:
    """
    Generate a full grid of anchors across an image.

    This is what YOLO/Faster R-CNN actually do: place anchors at regular intervals.

    Args:
        image_size: (width, height) of the image
        grid_size: (cols, rows) number of anchor positions
        scales: Anchor scales
        aspect_ratios: Anchor aspect ratios

    Returns:
        All anchors, shape (grid_rows * grid_cols * num_anchors_per_cell, 4)
    """
    img_w, img_h = image_size
    grid_cols, grid_rows = grid_size

    # Compute spacing between anchor centers
    stride_x = img_w / grid_cols
    stride_y = img_h / grid_rows

    all_anchors = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            # Center of this grid cell
            cx = (col + 0.5) * stride_x  # !Q why the 0.5 addition here?
            cy = (row + 0.5) * stride_y

            # Generate anchors at this position
            cell_anchors = generate_anchors(cx, cy, scales, aspect_ratios)
            if cell_anchors is not None:
                all_anchors.append(cell_anchors)

    if not all_anchors:
        return np.array([])

    return np.vstack(all_anchors)


# =============================================================================
# IoU (INTERSECTION OVER UNION)
# =============================================================================


def compute_iou(box_a: Box, box_b: Box) -> float:
    """
    Compute IoU (Intersection over Union) between two bounding boxes.

    IoU is THE metric for measuring how well a predicted box matches ground truth.
    - IoU = 1.0: Perfect overlap
    - IoU = 0.0: No overlap
    - IoU > 0.5: Typically considered a "correct" detection

    Args:
        box_a: First box as (x_min, y_min, x_max, y_max)
        box_b: Second box as (x_min, y_min, x_max, y_max)

    Returns:
        IoU value between 0.0 and 1.0

    Formula:
        IoU = Area(A ∩ B) / Area(A ∪ B)
            = Area(intersection) / (Area(A) + Area(B) - Area(intersection))

    Visual:
        ┌───────┐
        │   A   │
        │   ┌───┼───┐
        │   │ ∩ │   │    intersection = overlap region
        └───┼───┘   │    union = A + B - intersection
            │   B   │
            └───────┘
    """
    # TODO(human): Compute IoU between two boxes
    #
    # Your task: Calculate intersection and union areas (~10 lines)
    #
    # Step 1: Compute intersection coordinates
    #   x_min_inter = max(box_a[0], box_b[0])
    #   y_min_inter = max(box_a[1], box_b[1])
    #   x_max_inter = min(box_a[2], box_b[2])
    #   y_max_inter = min(box_a[3], box_b[3])
    #
    # Step 2: Compute intersection area (0 if no overlap)
    #   inter_width = max(0, x_max_inter - x_min_inter)
    #   inter_height = max(0, y_max_inter - y_min_inter)
    #   intersection = inter_width * inter_height
    #
    # Step 3: Compute areas of each box
    #   area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    #   area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    #
    # Step 4: Compute union and IoU
    #   union = area_a + area_b - intersection
    #   iou = intersection / union if union > 0 else 0.0
    #
    # return iou
    #

    x_min_inter = max(box_a[0], box_b[0])
    y_min_inter = max(box_a[1], box_b[1])
    x_max_inter = min(
        box_a[2], box_b[2]
    )  # Bug fix: min not max (intersection boundary)
    y_max_inter = min(box_a[3], box_b[3])  # Bug fix: min not max

    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    intersection = inter_width * inter_height

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union = area_a + area_b - intersection
    iou = 0.0 if union <= 0 else intersection / union

    return iou


def compute_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute IoU between all pairs of boxes from two sets (vectorized).

    This is the efficient version used in actual detection code. Instead of
    nested loops, we use NumPy broadcasting to compute all IoUs at once.

    Args:
        boxes_a: Array of shape (N, 4), each row is (x_min, y_min, x_max, y_max)
        boxes_b: Array of shape (M, 4), each row is (x_min, y_min, x_max, y_max)

    Returns:
        IoU matrix of shape (N, M) where iou[i, j] = IoU(boxes_a[i], boxes_b[j])

    Broadcasting Strategy:
        - Expand boxes_a to shape (N, 1, 4)
        - Expand boxes_b to shape (1, M, 4)
        - Operations broadcast to (N, M, 4) then reduce to (N, M)

    Why vectorized?
        NMS on 1000 boxes would need 1000×1000 = 1M IoU computations.
        Loops in Python: ~10 seconds. NumPy broadcasting: ~10ms.
    """
    # TODO(human): Compute IoU matrix using broadcasting
    #
    # Your task: Vectorized IoU computation (~15 lines)
    #
    # Step 1: Get number of boxes
    #   N = boxes_a.shape[0]
    #   M = boxes_b.shape[0]
    #
    # Step 2: Reshape for broadcasting # !Q why are these shapes different?
    #   # boxes_a: (N, 4) → (N, 1, 4)
    #   # boxes_b: (M, 4) → (1, M, 4)
    #   a = boxes_a[:, np.newaxis, :]  # Shape: (N, 1, 4) # !Q What does np.newaxis do? and explain the syntax in the []
    #   b = boxes_b[np.newaxis, :, :]  # Shape: (1, M, 4)
    #
    # Step 3: Compute intersection coordinates (broadcasts to (N, M))
    #   x_min_inter = np.maximum(a[..., 0], b[..., 0])
    #   y_min_inter = np.maximum(a[..., 1], b[..., 1])
    #   x_max_inter = np.minimum(a[..., 2], b[..., 2])
    #   y_max_inter = np.minimum(a[..., 3], b[..., 3])
    #
    # Step 4: Compute intersection area
    #   inter_width = np.maximum(0, x_max_inter - x_min_inter)
    #   inter_height = np.maximum(0, y_max_inter - y_min_inter)
    #   intersection = inter_width * inter_height
    #
    # Step 5: Compute areas
    #   area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    #   area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    #
    # Step 6: Compute union (broadcast area_a to (N, 1), area_b to (1, M))
    #   union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - intersection
    #
    # Step 7: Compute IoU (avoid division by zero)
    #   iou = np.where(union > 0, intersection / union, 0.0)
    #
    # return iou
    #

    a = boxes_a[:, np.newaxis, :]
    b = boxes_b[np.newaxis, :, :]

    # ... (ellipsis) means "all remaining dimensions"
    # a[..., 0] = a[:, :, 0] = "column 0 (x_min) for all boxes"
    x_min_inter = np.maximum(a[..., 0], b[..., 0])
    y_min_inter = np.maximum(a[..., 1], b[..., 1])
    x_max_inter = np.minimum(a[..., 2], b[..., 2])  # Bug fix: minimum not maximum
    y_max_inter = np.minimum(a[..., 3], b[..., 3])  # Bug fix: minimum not maximum

    inter_width = np.maximum(0, x_max_inter - x_min_inter)
    inter_height = np.maximum(0, y_max_inter - y_min_inter)
    intersection = inter_width * inter_height

    # Bug fix: area calculation was wrong, need proper column indexing
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - intersection

    iou = np.where(union > 0, intersection / union, 0.0)

    return iou


# =============================================================================
# NON-MAXIMUM SUPPRESSION (NMS)
# =============================================================================


def non_maximum_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> List[int]:
    """
    Apply Non-Maximum Suppression to remove duplicate detections.

    When a detector runs, it often produces multiple overlapping boxes for
    the same object. NMS keeps only the highest-confidence detection and
    removes boxes that overlap too much with it.

    Algorithm:
        1. Sort boxes by confidence score (highest first)
        2. Take the highest-scoring box, add to output
        3. Remove all boxes with IoU > threshold with this box
        4. Repeat until no boxes remain

    Args:
        boxes: Array of shape (N, 4), each row is (x_min, y_min, x_max, y_max)
        scores: Array of shape (N,), confidence score for each box
        iou_threshold: Boxes with IoU > this are considered duplicates

    Returns:
        List of indices of boxes to keep

    Example:
        If boxes 0, 3, 7 overlap significantly and box 3 has highest score,
        NMS keeps box 3 and removes 0 and 7.

    Why 0.5 threshold?
        - Too low (0.3): Might remove valid nearby detections
        - Too high (0.7): Might keep too many duplicates
        - 0.5 is the standard "good enough" value
    """
    # TODO(human): Implement NMS
    #
    # Your task: Greedy NMS loop (~15 lines)
    #
    # Step 1: Handle edge cases
    #   if len(boxes) == 0:
    #       return []
    #
    # Step 2: Sort indices by score (descending)
    #   order = np.argsort(-scores)  # Negative for descending order
    #
    # Step 3: Greedy selection
    #   keep = []
    #
    #   while len(order) > 0:
    #       # Take the highest scoring box
    #       i = order[0]
    #       keep.append(i)
    #
    #       # If only one box left, we're done
    #       if len(order) == 1:
    #           break
    #
    #       # Compute IoU of this box with all remaining boxes
    #       # (Use your compute_iou or compute_iou_batch here)
    #       remaining_boxes = boxes[order[1:]]
    #       ious = np.array([compute_iou(boxes[i], boxes[j]) for j in order[1:]])
    #
    #       # Keep only boxes with IoU <= threshold
    #       mask = ious <= iou_threshold
    #       order = order[1:][mask]
    #
    # return keep
    #

    if len(boxes) == 0:
        return []

    order = np.argsort(-scores)

    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        remaining_boxes = boxes[order[1:]]
        # boxes[i:i+1] keeps 2D shape (1, 4) needed for batch IoU
        # [0] flattens result from (1, M) to (M,) for masking
        ious = compute_iou_batch(boxes[i : i + 1], remaining_boxes)[0]

        mask = ious <= iou_threshold
        order = order[1:][mask]

    return keep


# =============================================================================
# VISUALIZATION
# =============================================================================


def visualize_anchors(
    image_size: Tuple[int, int] = (416, 416),
    grid_size: Tuple[int, int] = (4, 4),
    scales: List[float] = [32, 64],
    aspect_ratios: List[float] = [0.5, 1.0, 2.0],
) -> None:
    """Visualize anchor boxes on a grid."""
    anchors = generate_anchor_grid(image_size, grid_size, scales, aspect_ratios)

    if anchors is None or len(anchors) == 0:
        print("generate_anchors() not implemented yet!")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(0, image_size[0])
    ax.set_ylim(image_size[1], 0)  # Flip Y axis (image coordinates)
    ax.set_aspect("equal")
    ax.set_title(
        f"Anchor Grid: {grid_size[0]}×{grid_size[1]} cells, "
        f"{len(scales)} scales × {len(aspect_ratios)} ratios"
    )

    # Color by aspect ratio
    colors = plt.cm.Set1(np.linspace(0, 1, len(aspect_ratios)))
    anchors_per_cell = len(scales) * len(aspect_ratios)

    for i, (x_min, y_min, x_max, y_max) in enumerate(anchors):
        ratio_idx = i % len(aspect_ratios)
        color = colors[ratio_idx]

        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=1,
            edgecolor=color,
            facecolor="none",
            alpha=0.7,
        )
        ax.add_patch(rect)

    # Add legend
    legend_patches = [
        patches.Patch(color=colors[i], label=f"Ratio {aspect_ratios[i]:.1f}")
        for i in range(len(aspect_ratios))
    ]
    ax.legend(handles=legend_patches, loc="upper right")

    # Draw grid lines
    img_w, img_h = image_size
    grid_cols, grid_rows = grid_size
    for i in range(grid_cols + 1):
        ax.axvline(i * img_w / grid_cols, color="gray", linestyle="--", alpha=0.3)
    for i in range(grid_rows + 1):
        ax.axhline(i * img_h / grid_rows, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()


def visualize_iou(
    box_a: Box = (50, 50, 150, 150),
    box_b: Box = (100, 80, 200, 180),
) -> None:
    """Visualize IoU between two boxes with highlighted intersection."""
    iou = compute_iou(box_a, box_b)

    if iou is None:
        print("compute_iou() not implemented yet!")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(0, 300)
    ax.set_ylim(300, 0)
    ax.set_aspect("equal")

    # Draw box A
    rect_a = patches.Rectangle(
        (box_a[0], box_a[1]),
        box_a[2] - box_a[0],
        box_a[3] - box_a[1],
        linewidth=2,
        edgecolor="blue",
        facecolor="blue",
        alpha=0.3,
        label="Box A",
    )
    ax.add_patch(rect_a)

    # Draw box B
    rect_b = patches.Rectangle(
        (box_b[0], box_b[1]),
        box_b[2] - box_b[0],
        box_b[3] - box_b[1],
        linewidth=2,
        edgecolor="red",
        facecolor="red",
        alpha=0.3,
        label="Box B",
    )
    ax.add_patch(rect_b)

    # Draw intersection
    x_min_inter = max(box_a[0], box_b[0])
    y_min_inter = max(box_a[1], box_b[1])
    x_max_inter = min(box_a[2], box_b[2])
    y_max_inter = min(box_a[3], box_b[3])

    if x_max_inter > x_min_inter and y_max_inter > y_min_inter:
        rect_inter = patches.Rectangle(
            (x_min_inter, y_min_inter),
            x_max_inter - x_min_inter,
            y_max_inter - y_min_inter,
            linewidth=2,
            edgecolor="green",
            facecolor="green",
            alpha=0.5,
            label="Intersection",
        )
        ax.add_patch(rect_inter)

    ax.set_title(f"IoU = {iou:.3f}", fontsize=16)
    ax.legend(loc="upper right")

    # Add formula
    ax.text(
        150,
        280,
        "IoU = Intersection / Union\n    = Intersection / (A + B - Intersection)",
        fontsize=10,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()


def animate_nms(
    n_boxes: int = 15,
    iou_threshold: float = 0.5,
    delay: float = 0.5,
) -> None:
    """Animate the NMS process step by step."""
    np.random.seed(42)

    # Generate random boxes clustered around a few points
    centers = [(100, 100), (250, 150), (150, 280)]
    boxes = []
    scores = []

    for cx, cy in centers:
        for _ in range(n_boxes // 3 + 1):
            # Random offset from center
            dx, dy = np.random.normal(0, 20, 2)
            w, h = np.random.uniform(40, 80, 2)
            x_min = cx + dx - w / 2
            y_min = cy + dy - h / 2
            x_max = x_min + w
            y_max = y_min + h
            boxes.append([x_min, y_min, x_max, y_max])
            scores.append(np.random.uniform(0.3, 0.95))

    boxes = np.array(boxes[:n_boxes])
    scores = np.array(scores[:n_boxes])

    # Check if NMS is implemented
    keep = non_maximum_suppression(boxes, scores, iou_threshold)
    if keep is None:
        print("non_maximum_suppression() not implemented yet!")
        return

    # Now animate the process
    print("=" * 60)
    print("NMS ANIMATION")
    print("=" * 60)
    print(f"Starting with {len(boxes)} boxes")
    print(f"IoU threshold: {iou_threshold}")
    print()

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Sort by score for visualization
    order = np.argsort(-scores)
    remaining = set(range(len(boxes)))
    kept = []

    def draw_state(title: str):
        ax1.clear()
        ax2.clear()

        ax1.set_xlim(0, 350)
        ax1.set_ylim(350, 0)
        ax1.set_title("All Boxes (opacity = confidence)")
        ax1.set_aspect("equal")

        ax2.set_xlim(0, 350)
        ax2.set_ylim(350, 0)
        ax2.set_title(f"After NMS: {len(kept)} kept")
        ax2.set_aspect("equal")

        # Draw all boxes on left
        for i in range(len(boxes)):
            color = "green" if i in kept else ("gray" if i not in remaining else "blue")
            alpha = scores[i] if i in remaining else 0.2
            rect = patches.Rectangle(
                (boxes[i, 0], boxes[i, 1]),
                boxes[i, 2] - boxes[i, 0],
                boxes[i, 3] - boxes[i, 1],
                linewidth=1,
                edgecolor=color,
                facecolor=color,
                alpha=alpha * 0.5,
            )
            ax1.add_patch(rect)

        # Draw kept boxes on right
        for i in kept:
            rect = patches.Rectangle(
                (boxes[i, 0], boxes[i, 1]),
                boxes[i, 2] - boxes[i, 0],
                boxes[i, 3] - boxes[i, 1],
                linewidth=2,
                edgecolor="green",
                facecolor="green",
                alpha=0.5,
            )
            ax2.add_patch(rect)
            ax2.text(
                (boxes[i, 0] + boxes[i, 2]) / 2,
                (boxes[i, 1] + boxes[i, 3]) / 2,
                f"{scores[i]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                weight="bold",
            )

        fig.suptitle(title, fontsize=12)
        fig.canvas.draw()
        fig.canvas.flush_events()

    draw_state("Initial state")
    plt.pause(delay)

    # Simulate NMS step by step
    remaining_order = list(order)

    while remaining_order:
        i = remaining_order[0]
        kept.append(i)
        remaining.discard(i)

        draw_state(f"Keep box {i} (score={scores[i]:.2f})")
        plt.pause(delay)

        # Remove overlapping boxes
        removed = []
        for j in remaining_order[1:]:
            iou = compute_iou(tuple(boxes[i]), tuple(boxes[j]))
            if iou is not None and iou > iou_threshold:
                remaining.discard(j)
                removed.append(j)

        if removed:
            draw_state(
                f"Remove {len(removed)} overlapping boxes (IoU > {iou_threshold})"
            )
            plt.pause(delay)

        remaining_order = [j for j in remaining_order[1:] if j in remaining]

    draw_state(f"NMS Complete: {len(kept)} boxes kept from {len(boxes)}")
    plt.ioff()
    plt.show()


def full_pipeline_demo() -> None:
    """Demonstrate the full detection post-processing pipeline."""
    print("=" * 60)
    print("FULL DETECTION PIPELINE DEMO")
    print("=" * 60)
    print()

    # Check implementations
    test_anchors = generate_anchors(100, 100, [32], [1.0])
    test_iou = compute_iou((0, 0, 10, 10), (5, 5, 15, 15))
    test_nms = non_maximum_suppression(
        np.array([[0, 0, 10, 10], [5, 5, 15, 15]]), np.array([0.9, 0.8]), 0.5
    )

    missing = []
    if test_anchors is None:
        missing.append("generate_anchors()")
    if test_iou is None:
        missing.append("compute_iou()")
    if test_nms is None:
        missing.append("non_maximum_suppression()")

    if missing:
        print("!" * 60)
        print(f"Not implemented yet: {', '.join(missing)}")
        print("Complete the TODO(human) sections first.")
        print("!" * 60)
        return

    # Simulate detector output
    np.random.seed(123)
    n_detections = 30

    # Generate clustered boxes (simulating multiple objects)
    object_centers = [(80, 100), (250, 120), (160, 280)]
    boxes = []
    scores = []
    classes = []

    for obj_idx, (cx, cy) in enumerate(object_centers):
        # Each object generates multiple overlapping detections
        for _ in range(n_detections // len(object_centers)):
            dx, dy = np.random.normal(0, 15, 2)
            w = np.random.uniform(50, 90)
            h = np.random.uniform(50, 90)
            boxes.append(
                [cx + dx - w / 2, cy + dy - h / 2, cx + dx + w / 2, cy + dy + h / 2]
            )
            scores.append(np.random.uniform(0.4, 0.95))
            classes.append(obj_idx)

    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)

    print(f"Simulated detector output: {len(boxes)} raw detections")
    print()

    # Apply NMS per class
    final_boxes = []
    final_scores = []
    final_classes = []

    for cls in range(len(object_centers)):
        cls_mask = classes == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        keep = non_maximum_suppression(cls_boxes, cls_scores, iou_threshold=0.5)
        for k in keep:
            final_boxes.append(cls_boxes[k])
            final_scores.append(cls_scores[k])
            final_classes.append(cls)

    print(f"After NMS: {len(final_boxes)} final detections")
    print()

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Before NMS
    ax1.set_xlim(0, 350)
    ax1.set_ylim(350, 0)
    ax1.set_title(f"Before NMS: {len(boxes)} detections")
    ax1.set_aspect("equal")

    colors = ["blue", "red", "green"]
    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor=colors[cls],
            facecolor=colors[cls],
            alpha=score * 0.3,
        )
        ax1.add_patch(rect)

    # After NMS
    ax2.set_xlim(0, 350)
    ax2.set_ylim(350, 0)
    ax2.set_title(f"After NMS: {len(final_boxes)} detections")
    ax2.set_aspect("equal")

    for box, score, cls in zip(final_boxes, final_scores, final_classes):
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor=colors[cls],
            facecolor=colors[cls],
            alpha=0.4,
        )
        ax2.add_patch(rect)
        ax2.text(
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2,
            f"{score:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            weight="bold",
        )

    fig.suptitle("Object Detection Post-Processing Pipeline", fontsize=14)
    plt.tight_layout()
    plt.savefig("output/detection_pipeline.png", dpi=150, bbox_inches="tight")
    print("Saved: output/detection_pipeline.png")
    plt.show()


# =============================================================================
# TRAINING CONCEPTS (Documentation Only)
# =============================================================================

"""
HOW OBJECT DETECTOR TRAINING WORKS (Conceptual Overview)
=========================================================

You've already trained YOLO, so this is just a refresher of the key concepts
that connect to what you implemented here.

1. ANCHOR MATCHING
------------------
During training, each ground truth box is assigned to the anchor with highest IoU.

    for gt_box in ground_truth_boxes:
        ious = compute_iou_batch(anchors, gt_box)
        best_anchor = argmax(ious)
        anchor_assignments[best_anchor] = gt_box

This is why you need efficient IoU computation — matching 1000s of anchors
to dozens of ground truth boxes per image.


2. BOX REGRESSION
-----------------
Instead of predicting absolute coordinates, detectors predict OFFSETS from anchors:

    Δx = (gt_center_x - anchor_center_x) / anchor_width
    Δy = (gt_center_y - anchor_center_y) / anchor_height
    Δw = log(gt_width / anchor_width)
    Δh = log(gt_height / anchor_height)

Why log for width/height? It makes the optimization symmetric:
- Predicting 0.5 (halving) and 2.0 (doubling) are equally "wrong"


3. LOSS FUNCTIONS
-----------------
Detection loss = Classification Loss + Localization Loss

Classification (what object?):
    - Cross-entropy for multi-class
    - Focal loss to handle class imbalance (most anchors are background)

Localization (where exactly?):
    - L1 loss: |predicted - target|
    - Smooth L1: L1 when large, L2 when small (more stable gradients)
    - IoU loss: 1 - IoU(predicted_box, gt_box) — directly optimizes what we measure!


4. TRAINING LOOP
----------------
    for images, gt_boxes, gt_classes in dataloader:
        # Forward pass
        pred_offsets, pred_classes = model(images)

        # Decode predictions: apply offsets to anchors
        pred_boxes = decode_boxes(anchors, pred_offsets)

        # Match predictions to ground truth
        matches = match_anchors_to_gt(anchors, gt_boxes)

        # Compute losses
        loc_loss = smooth_l1(pred_offsets[matched], target_offsets)
        cls_loss = focal_loss(pred_classes, target_classes)

        loss = loc_loss + λ * cls_loss
        loss.backward()
        optimizer.step()


5. INFERENCE
------------
    # Forward pass
    pred_offsets, pred_classes = model(image)

    # Decode predictions
    pred_boxes = decode_boxes(anchors, pred_offsets)
    pred_scores = softmax(pred_classes)

    # Post-processing (WHAT YOU IMPLEMENTED!)
    for cls in range(num_classes):
        keep = NMS(pred_boxes, pred_scores[:, cls], threshold=0.5)
        final_detections.extend(pred_boxes[keep])

This is where your NMS implementation fits — it's the last step before
detections are used for downstream tasks (manipulation, tracking, etc.).
"""


# =============================================================================
# MAIN DEMO
# =============================================================================


def run_detection_concepts_demo() -> None:
    """Run all object detection concept demos."""
    print("\n" + "=" * 60)
    print("OBJECT DETECTION CONCEPTS — Phase 3 Final")
    print("=" * 60 + "\n")

    # Part 1: Anchors
    print("PART 1: Anchor Box Visualization")
    print("-" * 40)
    visualize_anchors()
    plt.savefig("output/anchor_grid.png", dpi=150, bbox_inches="tight")
    print("Saved: output/anchor_grid.png")
    plt.show()

    input("\nPress Enter to continue to IoU visualization...")

    # Part 2: IoU
    print("\nPART 2: IoU Visualization")
    print("-" * 40)
    visualize_iou()
    plt.savefig("output/iou_visualization.png", dpi=150, bbox_inches="tight")
    print("Saved: output/iou_visualization.png")
    plt.show()

    input("\nPress Enter to continue to NMS animation...")

    # Part 3: NMS
    print("\nPART 3: NMS Animation")
    print("-" * 40)
    animate_nms(n_boxes=12, iou_threshold=0.5, delay=0.8)

    input("\nPress Enter to continue to full pipeline demo...")

    # Part 4: Full Pipeline
    print("\nPART 4: Full Detection Pipeline")
    print("-" * 40)
    full_pipeline_demo()

    print()
    print("=" * 60)
    print("OBJECT DETECTION CONCEPTS COMPLETE")
    print("=" * 60)
    print("""
    What you implemented:
    1. generate_anchors() — Creating candidate regions
    2. compute_iou() — Measuring box overlap
    3. compute_iou_batch() — Efficient vectorized IoU
    4. non_maximum_suppression() — Removing duplicates

    These are the building blocks used in:
    - YOLO (all versions)
    - Faster R-CNN / Mask R-CNN
    - SSD, RetinaNet, EfficientDet

    Next up in Phase 4:
    - Deep RL (DQN) for manipulation
    - SLAM concepts (visual odometry)
    """)


if __name__ == "__main__":
    run_detection_concepts_demo()
