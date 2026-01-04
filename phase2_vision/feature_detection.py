"""
Feature Detection — Finding Interest Points in Images

This module implements Harris Corner Detection from scratch and demonstrates
SIFT/ORB feature extraction using OpenCV. Feature detection is the foundation
for visual odometry, SLAM, and image matching in robotics.

Why corners matter:
- Edges are ambiguous (you can slide along them)
- Corners are unique (distinct in all directions)
- Corners are stable under small viewpoint changes

The Harris detector finds corners by analyzing how the image changes
when you shift a small window in different directions.

Key insight: At a corner, shifting the window in ANY direction
causes a large change. At an edge, only perpendicular shifts matter.
At a flat region, no direction causes much change.

Connection to SLAM (Phase 4):
    Frame t → [Detect corners] → Match to Frame t+1 → Estimate motion
"""

import matplotlib

matplotlib.use("TkAgg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

# Reuse our convolution and Sobel kernels from image_convolution
from phase2_vision.image_convolution import convolve2d, gaussian_kernel, SOBEL_X, SOBEL_Y


# =============================================================================
# HARRIS CORNER DETECTION
# =============================================================================


def harris_response(image: np.ndarray, k: float = 0.04) -> np.ndarray:
    """
    Compute Harris corner response for each pixel.

    The Harris detector uses the "structure tensor" (also called the
    second moment matrix) to measure how the image changes in a local window.

    Structure Tensor M at each pixel:
        M = [ Σ(Ix²)    Σ(Ix·Iy) ]
            [ Σ(Ix·Iy)  Σ(Iy²)   ]

    Where:
        - Ix, Iy are image gradients (from Sobel)
        - Σ sums over a Gaussian-weighted local window

    Corner Response R:
        R = det(M) - k * trace(M)²
        R = (Ix2 * Iy2 - Ixy²) - k * (Ix2 + Iy2)²

    Interpretation:
        - R > 0 (large positive): Corner (both eigenvalues large)
        - R < 0 (large negative): Edge (one eigenvalue large)
        - R ≈ 0: Flat region (both eigenvalues small)

    Args:
        image: Grayscale image, shape (H, W), values 0-255
        k: Harris detector free parameter (typically 0.04-0.06)

    Returns:
        Corner response image R, shape slightly smaller than input
        (due to convolution border effects)
    """
    # TODO(human): Implement Harris corner response
    #
    # Your task: Compute the corner response R (~12-15 lines)
    #
    # Step 1: Compute image gradients using Sobel
    #   Ix = convolve2d(image, SOBEL_X)
    #   Iy = convolve2d(image, SOBEL_Y)
    #
    # Step 2: Compute products of gradients (element-wise)
    #   Ix2 = Ix * Ix
    #   Iy2 = Iy * Iy
    #   Ixy = Ix * Iy
    #
    # Step 3: Apply Gaussian smoothing to each product
    #   This creates the "weighted sum" in the structure tensor
    #   gauss = gaussian_kernel(size=5, sigma=1.0)
    #   Sx2 = convolve2d(Ix2, gauss)   # Smoothed Ix²
    #   Sy2 = convolve2d(Iy2, gauss)   # Smoothed Iy²
    #   Sxy = convolve2d(Ixy, gauss)   # Smoothed Ix·Iy
    #
    # Step 4: Compute Harris response
    #   det_M = Sx2 * Sy2 - Sxy * Sxy     # determinant
    #   trace_M = Sx2 + Sy2               # trace
    #   R = det_M - k * (trace_M ** 2)
    #
    # return R

    Ix = convolve2d(image, SOBEL_X)
    Iy = convolve2d(image, SOBEL_Y)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    gauss = gaussian_kernel(size=5, sigma=1.0)
    Sx2 = convolve2d(Ix2, gauss)
    Sy2 = convolve2d(Iy2, gauss)
    Sxy = convolve2d(Ixy, gauss)

    det_M = Sx2 * Sy2 - Sxy * Sxy
    trace_M = Sx2 + Sy2
    R = det_M - k * (trace_M**2)

    return R


def detect_corners(
    response: np.ndarray,
    threshold_ratio: float = 0.01,
    min_distance: int = 10,
    offset: int = 3,
) -> List[Tuple[int, int]]:
    """
    Extract corner locations from Harris response using non-maximum suppression.

    Non-maximum suppression ensures we only keep the strongest corner
    in each local region, avoiding clusters of detections.

    Args:
        response: Harris corner response image (from harris_response)
        threshold_ratio: Keep corners with R > threshold_ratio * max(R)
        min_distance: Minimum pixels between detected corners
        offset: Pixel offset to convert from response coords to original image coords.
                Default 3 accounts for Sobel 3x3 (1px) + Gaussian 5x5 (2px).

    Returns:
        List of (row, col) tuples for each detected corner (in original image coordinates)
    """
    if response is None:
        return []

    # Threshold: keep only strong responses
    threshold = threshold_ratio * response.max()
    corner_mask = response > threshold

    # Find all points above threshold
    candidates = np.argwhere(corner_mask)

    if len(candidates) == 0:
        return []

    # Sort by response strength (strongest first)
    strengths = response[corner_mask]
    sorted_indices = np.argsort(-strengths)
    candidates = candidates[sorted_indices]

    # Non-maximum suppression: keep corners that aren't too close to stronger ones
    corners = []
    for row, col in candidates:
        # Check if too close to an already-selected corner
        too_close = False
        for prev_row, prev_col in corners:
            dist = np.sqrt((row - prev_row) ** 2 + (col - prev_col) ** 2)
            if dist < min_distance:
                too_close = True
                break

        if not too_close:
            # Add offset to convert from response coords to original image coords
            corners.append((row + offset, col + offset))

    return corners


# =============================================================================
# SIFT / ORB WITH OPENCV
# =============================================================================


def extract_sift_features(
    image: np.ndarray,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Extract SIFT keypoints and descriptors using OpenCV.

    SIFT (Scale-Invariant Feature Transform) finds keypoints that are:
    - Scale invariant (detected at multiple scales)
    - Rotation invariant (descriptor normalized by dominant orientation)
    - Distinctive (128-dimensional descriptor)

    Args:
        image: Grayscale image

    Returns:
        keypoints: List of cv2.KeyPoint objects
        descriptors: Array of shape (N, 128) with SIFT descriptors
    """
    # Ensure uint8 for OpenCV
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors


def extract_orb_features(
    image: np.ndarray,
    n_features: int = 500,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Extract ORB keypoints and descriptors using OpenCV.

    ORB (Oriented FAST and Rotated BRIEF) is a fast alternative to SIFT:
    - Uses FAST corner detector (very fast)
    - Binary descriptor (32 bytes vs SIFT's 128 floats)
    - Rotation invariant
    - Good for real-time robotics applications

    Args:
        image: Grayscale image
        n_features: Maximum number of features to detect

    Returns:
        keypoints: List of cv2.KeyPoint objects
        descriptors: Array of shape (N, 32) with binary ORB descriptors
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors


def match_features(
    desc1: np.ndarray,
    desc2: np.ndarray,
    method: str = "sift",
    ratio_threshold: float = 0.75,
) -> List[cv2.DMatch]:
    """
    Match features between two images using descriptor matching.

    Uses Lowe's ratio test: a match is only accepted if the best match
    is significantly better than the second-best match.

    Args:
        desc1: Descriptors from image 1
        desc2: Descriptors from image 2
        method: "sift" (L2 norm) or "orb" (Hamming distance)
        ratio_threshold: Lowe's ratio test threshold

    Returns:
        List of good matches (cv2.DMatch objects)
    """
    if desc1 is None or desc2 is None:
        return []

    # Choose distance metric based on descriptor type
    if method == "orb":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Find 2 best matches for ratio test
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    return good_matches


# =============================================================================
# TEST IMAGE GENERATION
# =============================================================================


def generate_corner_test_image(size: int = 128) -> np.ndarray:
    """Generate a test image with clear corners for Harris detection."""
    image = np.ones((size, size), dtype=np.float64) * 128

    # Add rectangles (4 corners each)
    image[20:60, 20:60] = 200
    image[70:110, 30:90] = 50
    image[30:50, 80:110] = 220

    # Add a triangle (3 corners)
    for i in range(30):
        image[90 + i, 90 - i : 90 + i] = 180

    # Add some noise
    noise = np.random.normal(0, 5, image.shape)
    image = np.clip(image + noise, 0, 255)

    return image


def generate_shifted_image(image: np.ndarray, dx: int = 5, dy: int = 3) -> np.ndarray:
    """Create a slightly shifted version of an image (for matching demo)."""
    shifted = np.zeros_like(image)
    h, w = image.shape

    # Shift and copy
    src_y1, src_y2 = max(0, dy), min(h, h + dy)
    src_x1, src_x2 = max(0, dx), min(w, w + dx)
    dst_y1, dst_y2 = max(0, -dy), min(h, h - dy)
    dst_x1, dst_x2 = max(0, -dx), min(w, w - dx)

    shifted[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    return shifted


# =============================================================================
# VISUALIZATION
# =============================================================================


def visualize_harris_detection(
    image: np.ndarray, corners: List[Tuple[int, int]]
) -> None:
    """Show detected Harris corners on the image."""
    plt.figure(figsize=(10, 5))

    # Original with corners
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    if corners:
        rows, cols = zip(*corners)
        plt.scatter(cols, rows, c="red", s=50, marker="x", linewidths=2)
    plt.title(f"Harris Corners Detected: {len(corners)}")
    plt.axis("off")

    # Response heatmap
    plt.subplot(1, 2, 2)
    response = harris_response(image)
    if response is not None:
        plt.imshow(response, cmap="hot")
        plt.colorbar(label="Corner Response R")
        plt.title("Harris Response (red = corners)")
    else:
        plt.text(
            0.5,
            0.5,
            "harris_response() not implemented",
            ha="center",
            va="center",
            fontsize=12,
        )
    plt.axis("off")

    plt.tight_layout()


def visualize_response_interpretation(image: np.ndarray) -> None:
    """Show how Harris response classifies corners, edges, and flat regions."""
    response = harris_response(image)

    if response is None:
        print("harris_response() not implemented yet!")
        return

    # Offset due to convolutions: Sobel 3x3 (1px) + Gaussian 5x5 (2px) = 3px
    offset = 3

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Harris Response Interpretation", fontsize=14)

    # Original
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Full response
    im = axes[0, 1].imshow(
        response, cmap="RdBu", vmin=-response.max(), vmax=response.max()
    )
    axes[0, 1].set_title("Harris Response R")
    plt.colorbar(im, ax=axes[0, 1])
    axes[0, 1].axis("off")

    # Crop image to match response coordinates (accounting for offset)
    rh, rw = response.shape
    image_cropped = image[offset : offset + rh, offset : offset + rw]

    # Corners (R > 0)
    corners_mask = response > 0.01 * response.max()
    axes[1, 0].imshow(image_cropped, cmap="gray")
    axes[1, 0].imshow(corners_mask, cmap="Reds", alpha=0.5)
    axes[1, 0].set_title("Corners (R > 0, red overlay)")
    axes[1, 0].axis("off")

    # Edges (R < 0)
    edges_mask = response < -0.01 * abs(response.min())
    axes[1, 1].imshow(image_cropped, cmap="gray")
    axes[1, 1].imshow(edges_mask, cmap="Blues", alpha=0.5)
    axes[1, 1].set_title("Edges (R < 0, blue overlay)")
    axes[1, 1].axis("off")

    plt.tight_layout()


def visualize_feature_comparison(image: np.ndarray) -> None:
    """Compare Harris, SIFT, and ORB on the same image."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Feature Detection Comparison", fontsize=14)

    # Ensure uint8 for OpenCV
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)

    # Harris
    response = harris_response(image)
    if response is not None:
        corners = detect_corners(response)
        axes[0].imshow(image, cmap="gray")
        if corners:
            rows, cols = zip(*corners)
            axes[0].scatter(cols, rows, c="red", s=30, marker="x")
        axes[0].set_title(f"Harris Corners: {len(corners)}")
    else:
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Harris (not implemented)")
    axes[0].axis("off")

    # SIFT
    sift_kp, _ = extract_sift_features(image)
    sift_img = cv2.drawKeypoints(
        image_uint8, sift_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    axes[1].imshow(sift_img)
    axes[1].set_title(f"SIFT Keypoints: {len(sift_kp)}")
    axes[1].axis("off")

    # ORB
    orb_kp, _ = extract_orb_features(image)
    orb_img = cv2.drawKeypoints(
        image_uint8, orb_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    axes[2].imshow(orb_img)
    axes[2].set_title(f"ORB Keypoints: {len(orb_kp)}")
    axes[2].axis("off")

    plt.tight_layout()


def visualize_feature_matching(
    image1: np.ndarray,
    image2: np.ndarray,
    method: str = "orb",
) -> None:
    """Show feature matching between two images."""
    img1_uint8 = np.clip(image1, 0, 255).astype(np.uint8)
    img2_uint8 = np.clip(image2, 0, 255).astype(np.uint8)

    if method == "sift":
        kp1, desc1 = extract_sift_features(image1)
        kp2, desc2 = extract_sift_features(image2)
    else:
        kp1, desc1 = extract_orb_features(image1)
        kp2, desc2 = extract_orb_features(image2)

    matches = match_features(desc1, desc2, method=method)

    # Draw matches
    match_img = cv2.drawMatches(
        img1_uint8,
        kp1,
        img2_uint8,
        kp2,
        matches[:50],  # Show top 50 matches
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    plt.figure(figsize=(14, 6))
    plt.imshow(match_img)
    plt.title(f"{method.upper()} Feature Matching: {len(matches)} matches")
    plt.axis("off")
    plt.tight_layout()


# =============================================================================
# MAIN DEMO
# =============================================================================


def run_harris_demo() -> None:
    """Run Harris corner detection demo."""
    print("=" * 60)
    print("HARRIS CORNER DETECTION")
    print("=" * 60)
    print()

    # Generate test image
    image = generate_corner_test_image()
    print(f"Generated {image.shape[0]}x{image.shape[1]} test image")

    # Compute Harris response
    print("Computing Harris response...")
    response = harris_response(image)

    if response is None:
        print()
        print("!" * 60)
        print("harris_response() not implemented yet!")
        print("Complete the TODO(human) section first.")
        print("!" * 60)

        # Still show the test image
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap="gray")
        plt.title("Test Image (implement harris_response to detect corners)")
        plt.axis("off")
        plt.show()
        return

    # Detect corners
    corners = detect_corners(response)
    print(f"Detected {len(corners)} corners")

    # Visualizations
    plt.ion()

    visualize_harris_detection(image, corners)
    plt.savefig("output/harris_corners.png", dpi=150, bbox_inches="tight")

    visualize_response_interpretation(image)
    plt.savefig("output/harris_interpretation.png", dpi=150, bbox_inches="tight")

    print()
    print("Figures saved:")
    print("  - output/harris_corners.png")
    print("  - output/harris_interpretation.png")

    plt.ioff()
    plt.show()


def run_feature_comparison_demo() -> None:
    """Compare Harris, SIFT, and ORB feature detection."""
    print()
    print("=" * 60)
    print("FEATURE DETECTION COMPARISON")
    print("=" * 60)
    print()

    image = generate_corner_test_image()

    plt.ion()
    visualize_feature_comparison(image)
    plt.savefig("output/feature_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: output/feature_comparison.png")
    plt.ioff()
    plt.show()


def run_matching_demo() -> None:
    """Demonstrate feature matching between two images."""
    print()
    print("=" * 60)
    print("FEATURE MATCHING DEMO")
    print("=" * 60)
    print()

    # Create two related images
    image1 = generate_corner_test_image()
    image2 = generate_shifted_image(image1, dx=8, dy=5)

    # Add some rotation/noise to image2 to make it more challenging
    noise = np.random.normal(0, 10, image2.shape)
    image2 = np.clip(image2 + noise, 0, 255)

    print("Matching features between original and shifted image...")

    plt.ion()
    visualize_feature_matching(image1, image2, method="orb")
    plt.savefig("output/feature_matching.png", dpi=150, bbox_inches="tight")
    print("Saved: output/feature_matching.png")

    plt.ioff()
    plt.show()


def main():
    """Run all feature detection demos."""
    print("\n" + "=" * 60)
    print("FEATURE DETECTION — Phase 2 Final Module")
    print("=" * 60 + "\n")

    # Part 1: Harris Corner Detection
    run_harris_demo()

    input("\nPress Enter to continue to feature comparison...")

    # Part 2: Compare Harris vs SIFT vs ORB
    run_feature_comparison_demo()

    input("\nPress Enter to continue to matching demo...")

    # Part 3: Feature Matching (robotics application)
    run_matching_demo()

    print()
    print("=" * 60)
    print("ROBOTICS CONNECTION")
    print("=" * 60)
    print("""
    What you just saw is the foundation of visual odometry:

    1. Detect features in frame t      (Harris, SIFT, or ORB)
    2. Detect features in frame t+1
    3. Match features between frames
    4. Estimate camera motion from matches

    In Phase 4 (SLAM), you'll use this to build maps and localize!
    """)


if __name__ == "__main__":
    main()
