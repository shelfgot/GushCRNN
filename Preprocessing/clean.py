import numpy as np

import cv2

def adaptive_threshold(image):
    """Apply the adaptive threshhold algo from OpenCV for binarization

    Args:
        image (np.array): a numpy array of the image

    Returns:
        np.array: binarized image
    """
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )


def median_filter(image, kernel_size=3):
    """Applies a median filter to remove noise from image

    Args:
        image (np.array): image to filter
        kernel_size (int, optional): kernel size for the median filter. Defaults to 3.

    Returns:
        image: filtered image
    """
    padded = np.pad(image, kernel_size // 2, mode='edge')
    filtered = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered[i, j] = np.median(padded[i:i+kernel_size, j:j+kernel_size])
    
    return filtered

def normalize_image(image):
    """Normalize image to [0,1] range

    Args:
        image (np.array): image

    Returns:
        image: normalized image
    """
    return image / 255.0
