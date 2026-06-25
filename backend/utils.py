import cv2
import numpy as np

def preprocess_frame(
    img: np.ndarray, 
    blur: bool = False, 
    equalize: bool = False, 
    edges: bool = False
) -> np.ndarray:
    """
    Preprocess a video frame (convert to grayscale, apply blur, histogram
    equalization, or Canny edge detection) for tracking.

    Args:
        img: Input BGR image frame.
        blur: Whether to apply Gaussian Blur.
        equalize: Whether to apply Histogram Equalization.
        edges: Whether to apply Canny Edge Detection.

    Returns:
        Processed single-channel (grayscale) or binary image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if equalize:
        gray = cv2.equalizeHist(gray)
    if edges:
        gray = cv2.Canny(gray, 50, 150)
    return gray

def draw_box(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Draws a bounding box representing the tracked ROI on the image.

    Args:
        img: Image on which to draw.
        x: Top-left x coordinate of ROI.
        y: Top-left y coordinate of ROI.
        w: Width of ROI.
        h: Height of ROI.

    Returns:
        Annotated copy of the input image.
    """
    return cv2.rectangle(img.copy(), (x, y), (x + w, x + h), (0, 0, 255), 2)
