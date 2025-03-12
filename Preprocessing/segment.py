import numpy as np
def segment_characters(image):
    """Find character bounding boxes and extract individual characters

    Args:
        image (np.array): N x 1

    Returns:
        characters (np.array): N x 1 
    """
    rows, cols = image.shape
    projections = np.sum(image, axis=0)  # Vertical projection profile
    
    threshold = 0.02 * rows  # Threshold for separating characters
    segments, start = [], None
    
    for i, value in enumerate(projections):
        if value > threshold and start is None:
            start = i
        elif value <= threshold and start is not None:
            segments.append((start, i))
            start = None
    
    # Extract character images
    characters = [image[:, start:end] for start, end in segments]
    return characters
