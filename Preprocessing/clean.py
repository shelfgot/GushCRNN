import numpy as np

import cv2


'''def otsu_threshold(image):
    """Apply Otsu's algorithm for binarization

    Args:
        image (np.array): array, N x 1

    Returns:
        image (np.array): array, N x 1 (binarized)
    """
    #DEBUG!
    #print("Min pixel value:", np.min(image))
    #print("Max pixel value:", np.max(image))

    histogram, _ = np.histogram(image.flatten(), bins=256, range=[0, 1])
    total_pixels = image.size
    sum_total = np.sum(np.arange(256) * histogram)

    sum_background, weight_background, max_variance, threshold = 0, 0, 0, 0
    for t in range(256):
        weight_background += histogram[t]
        if weight_background == 0: 
            continue
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0: 
            break

        sum_background += t * histogram[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > max_variance:
            max_variance, threshold = variance, t


    return (image > threshold).astype(np.uint8)'''

def adaptive_threshold(image):
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
