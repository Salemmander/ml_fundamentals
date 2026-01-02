"""
Image Convolution & Filters — Building Blocks of Computer Vision

This module implements 2D convolution from scratch, demonstrating the fundamental
operation behind edge detection, blurring, and CNN feature extraction.

Why this matters for robotics:
- Edge detection: lane boundaries, obstacle outlines, feature extraction
- Blurring: noise reduction for noisy sensor images
- Convolution: the core operation in CNNs for visual perception

Key concept:
    Convolution slides a small kernel (filter) across an image, computing a
    weighted sum at each position. Different kernels detect different features:
    - Sobel kernels → edges (gradients)
    - Gaussian kernel → blur (smoothing)
    - Sharpen kernel → enhance details

The same operation powers the first layers of every CNN — just learned kernels
instead of hand-designed ones.
"""

import matplotlib

matplotlib.use("TkAgg")  # Must come before pyplot import for Linux

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Tuple


# =============================================================================
# KERNELS (Filters)
# =============================================================================

# Sobel kernels for edge detection (gradient approximation)
SOBEL_X = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ],
    dtype=np.float64,
)

SOBEL_Y = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ],
    dtype=np.float64,
)

# Sharpening kernel (enhances edges)
SHARPEN = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ],
    dtype=np.float64,
)

# Simple box blur (averaging)
BOX_BLUR_3x3 = np.ones((3, 3), dtype=np.float64) / 9


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel for blurring.

    The Gaussian kernel weights center pixels more heavily than edge pixels,
    producing a smoother blur than a simple box filter.

    Args:
        size: Kernel size (should be odd, e.g., 3, 5, 7)
        sigma: Standard deviation of the Gaussian

    Returns:
        Normalized Gaussian kernel, shape (size, size)
    """
    # Create coordinate grids centered at 0
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)

    # 2D Gaussian formula: exp(-(x² + y²) / (2σ²))
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # Normalize so kernel sums to 1 (preserves image brightness)
    return kernel / kernel.sum()


# =============================================================================
# CORE CONVOLUTION OPERATION
# =============================================================================


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply 2D convolution to an image using a kernel (filter).

    Convolution slides the kernel across every position in the image,
    computing the element-wise product and summing the result. This is
    the fundamental operation behind edge detection, blurring, and CNNs.

    Args:
        image: Input image, shape (height, width), values typically 0-255
        kernel: Convolution kernel, shape (kh, kw), typically 3x3 or 5x5

    Returns:
        Filtered image, shape (height - kh + 1, width - kw + 1)
        Note: Output is smaller due to "valid" convolution (no padding)

    Example:
        For a 3x3 kernel at position (i, j):
        output[i, j] = sum(image[i:i+3, j:j+3] * kernel)
    """
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # Output size for "valid" convolution (no padding)
    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1
    output = np.zeros((output_h, output_w), dtype=np.float64)

    # TODO(human): Implement the convolution operation
    #
    # Your task: Fill in the nested loops to slide the kernel across the image
    #
    # For each output position (i, j):
    # 1. Extract the region of the image under the kernel
    #    region = image[i:i+kernel_h, j:j+kernel_w]
    #
    # 2. Compute element-wise product with kernel and sum
    #    output[i, j] = np.sum(region * kernel)
    #
    # Structure:
    # for i in range(output_h):
    #     for j in range(output_w):
    #         ... extract region and compute weighted sum ...
    #

    for i in range(output_h):
        for j in range(output_w):
            region = image[i : i + kernel_h, j : j + kernel_w]
            output[i, j] = np.sum((region * kernel))

    return output


# =============================================================================
# EDGE DETECTION
# =============================================================================


def detect_edges(image: np.ndarray) -> np.ndarray:
    """
    Detect edges using Sobel operators.

    Applies Sobel X and Y kernels to compute horizontal and vertical
    gradients, then combines them into gradient magnitude.

    Args:
        image: Grayscale image, shape (height, width)

    Returns:
        Edge magnitude image, normalized to 0-255 range

    Math:
        Gx = convolve(image, SOBEL_X)  # horizontal edges
        Gy = convolve(image, SOBEL_Y)  # vertical edges
        magnitude = sqrt(Gx² + Gy²)
    """
    # TODO(human): Implement edge detection
    #
    # Your task: Apply Sobel kernels and compute gradient magnitude (~5-8 lines)
    #
    # Steps:
    # 1. Apply SOBEL_X kernel to get horizontal gradient: gx = convolve2d(image, SOBEL_X)
    # 2. Apply SOBEL_Y kernel to get vertical gradient: gy = convolve2d(image, SOBEL_Y)
    # 3. Compute magnitude: magnitude = np.sqrt(gx**2 + gy**2)
    # 4. Normalize to 0-255: magnitude = (magnitude / magnitude.max()) * 255
    #
    # Return the normalized magnitude image
    #

    gx = convolve2d(image, SOBEL_X)
    gy = convolve2d(image, SOBEL_Y)
    magnitude = np.sqrt(gx**2 + gy**2)  # Pythagorean theorem
    magnitude = (magnitude / magnitude.max()) * 255
    return magnitude


# =============================================================================
# TEST IMAGE GENERATION
# =============================================================================


def generate_test_image(size: int = 64) -> np.ndarray:
    """
    Generate a synthetic test image with clear geometric features.

    Creates an image with rectangles and lines that have distinct edges
    for testing edge detection and filter effects.

    Args:
        size: Image dimensions (size x size)

    Returns:
        Grayscale image with values 0-255
    """
    image = np.zeros((size, size), dtype=np.float64)

    # Add a white rectangle
    image[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 200

    # Add a smaller dark rectangle inside
    image[3 * size // 8 : 5 * size // 8, 3 * size // 8 : 5 * size // 8] = 50

    # Add diagonal line
    for i in range(size):
        if 0 <= i < size:
            image[i, min(i, size - 1)] = 255

    # Add some noise for realism
    noise = np.random.normal(0, 10, (size, size))
    image = np.clip(image + noise, 0, 255)

    return image


def generate_simple_image(size: int = 12) -> np.ndarray:
    """
    Generate a tiny test image for animation demo.

    Small enough to animate kernel sliding pixel-by-pixel.
    """
    image = np.zeros((size, size), dtype=np.float64)

    # Simple pattern: bright square in center
    image[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = 200

    # Add border
    image[2, 2:-2] = 150
    image[-3, 2:-2] = 150
    image[2:-2, 2] = 150
    image[2:-2, -3] = 150

    return image


# =============================================================================
# ANIMATED VISUALIZATION
# =============================================================================


def animate_convolution(
    image: np.ndarray,
    kernel: np.ndarray,
    delay: float = 0.1,
    skip: int = 1,
) -> np.ndarray:
    """
    Animate the convolution operation, showing kernel sliding across image.

    This visualization shows exactly what convolution does:
    - Yellow box: current kernel position on input image
    - Red dot: output pixel being computed
    - Updates in real-time as kernel slides

    Args:
        image: Input image (use small image, e.g., 12x12 for visibility)
        kernel: Convolution kernel
        delay: Seconds between frames
        skip: Process every Nth pixel (1 = all pixels, 2 = every other, etc.)

    Returns:
        The convolved output image
    """
    print("=" * 60)
    print("CONVOLUTION ANIMATION")
    print("=" * 60)
    print("Watch the yellow kernel slide across the input image.")
    print("Each position computes one output pixel (red dot).")
    print()

    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1
    output = np.zeros((output_h, output_w), dtype=np.float64)

    # Set up the figure
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Convolution: Sliding Kernel Operation", fontsize=14)

    ax_input, ax_kernel, ax_output = axes

    # Input image display
    im_input = ax_input.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax_input.set_title("Input Image")
    ax_input.set_xlabel("Column (j)")
    ax_input.set_ylabel("Row (i)")

    # Kernel position rectangle (will be updated)
    rect = patches.Rectangle(
        (0, 0), kernel_w, kernel_h, linewidth=2, edgecolor="yellow", facecolor="none"
    )
    ax_input.add_patch(rect)

    # Kernel display
    im_kernel = ax_kernel.imshow(kernel, cmap="RdBu", vmin=-2, vmax=2)
    ax_kernel.set_title(f"Kernel ({kernel_h}x{kernel_w})")
    for i in range(kernel_h):
        for j in range(kernel_w):
            ax_kernel.text(
                j,
                i,
                f"{kernel[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )
    plt.colorbar(im_kernel, ax=ax_kernel, fraction=0.046)

    # Output image display (starts empty)
    im_output = ax_output.imshow(output, cmap="gray", vmin=0, vmax=255)
    ax_output.set_title("Output (being computed...)")
    ax_output.set_xlabel("Column")
    ax_output.set_ylabel("Row")

    # Marker for current output pixel
    (output_marker,) = ax_output.plot([], [], "ro", markersize=8)

    plt.tight_layout()

    # Animate the convolution
    for i in range(0, output_h, skip):
        for j in range(0, output_w, skip):
            # Extract region under kernel
            region = image[i : i + kernel_h, j : j + kernel_w]

            # Compute convolution at this position
            value = np.sum(region * kernel)
            output[i, j] = value

            # For skipped pixels, fill them too (approximate)
            if skip > 1:
                for di in range(skip):
                    for dj in range(skip):
                        if i + di < output_h and j + dj < output_w:
                            r = image[
                                i + di : i + di + kernel_h, j + dj : j + dj + kernel_w
                            ]
                            if r.shape == kernel.shape:
                                output[i + di, j + dj] = np.sum(r * kernel)

            # Update kernel position rectangle
            rect.set_xy((j - 0.5, i - 0.5))

            # Update output marker
            output_marker.set_data([j], [i])

            # Update output image
            # Normalize for display
            display_output = np.clip(output, 0, 255)
            im_output.set_array(display_output)

            # Refresh display
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(delay)

    print("Convolution complete!")
    ax_output.set_title("Output (complete)")

    plt.ioff()
    return output


def run_convolution_demo(image_size: int = 64) -> None:
    """
    Run comprehensive convolution demo with multiple filters.

    Shows side-by-side comparison of original image with various filters:
    - Gaussian blur
    - Edge detection (Sobel)
    - Sharpening

    Args:
        image_size: Size of test image
    """
    print("=" * 60)
    print("IMAGE CONVOLUTION & FILTERS DEMO")
    print("=" * 60)
    print()

    # Generate test image
    image = generate_test_image(image_size)
    print(f"Generated {image_size}x{image_size} test image")

    # Check if convolution is implemented
    test_output = convolve2d(image, BOX_BLUR_3x3)
    if test_output is None or np.all(test_output == 0):
        print("\n" + "!" * 60)
        print("NOTE: convolve2d() not yet implemented!")
        print("Complete the TODO(human) section in convolve2d(),")
        print("then run this demo again.")
        print("!" * 60)

        # Still show the test image
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap="gray")
        plt.title("Test Image (implement convolve2d to see filters)")
        plt.colorbar()
        plt.show()
        return

    print("Applying filters...")

    # Apply various filters
    gaussian = gaussian_kernel(5, sigma=1.5)
    blurred = convolve2d(image, gaussian)
    sharpened = convolve2d(image, SHARPEN)

    # Edge detection
    edges = detect_edges(image)
    if edges is None:
        print("Note: detect_edges() not implemented yet, skipping edge display")
        edges = np.zeros_like(blurred)

    # Normalize outputs for display
    blurred_display = np.clip(blurred, 0, 255)
    sharpened_display = np.clip(sharpened, 0, 255)

    # Create visualization
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Convolution Filter Effects", fontsize=14)

    # Original
    axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Blurred
    axes[0, 1].imshow(blurred_display, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Gaussian Blur (5x5, σ=1.5)")
    axes[0, 1].axis("off")

    # Sharpened
    axes[1, 0].imshow(sharpened_display, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Sharpened")
    axes[1, 0].axis("off")

    # Edges
    axes[1, 1].imshow(edges, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Edge Detection (Sobel)")
    axes[1, 1].axis("off")

    plt.tight_layout()

    # Print summary
    print()
    print("=" * 60)
    print("FILTER SUMMARY")
    print("=" * 60)
    print("• Gaussian Blur: Smooths image, reduces noise")
    print("  - Kernel weights decrease with distance from center")
    print("  - Larger σ = more blur")
    print()
    print("• Sharpening: Enhances edges and details")
    print("  - Center weight > 1, neighbors negative")
    print("  - Amplifies differences between pixels")
    print()
    print("• Sobel Edge Detection: Finds intensity gradients")
    print("  - Sobel X detects vertical edges")
    print("  - Sobel Y detects horizontal edges")
    print("  - Magnitude combines both directions")
    print("=" * 60)

    plt.ioff()
    plt.savefig("convolution_filters.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved to convolution_filters.png")
    plt.show()


def main():
    """Main entry point — runs both animation and filter demo."""
    print("\n" + "=" * 60)
    print("PART 1: Animated Convolution (small image)")
    print("=" * 60 + "\n")

    # First, run the animation on a small image
    small_image = generate_simple_image(12)

    # Check if implemented before animating
    test = convolve2d(small_image, SOBEL_X)
    if test is not None and not np.all(test == 0):
        animate_convolution(small_image, SOBEL_X, delay=0.05, skip=1)
        plt.show()

        input("\nPress Enter to continue to filter comparison demo...")

    print("\n" + "=" * 60)
    print("PART 2: Filter Comparison (larger image)")
    print("=" * 60 + "\n")

    # Then show filter comparison
    run_convolution_demo(image_size=64)


if __name__ == "__main__":
    main()
