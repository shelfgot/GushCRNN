import numpy as np
from PIL import Image


def load_image(filename):
    """Loads our image as a NumPy array in grayscale.

    Args:
        filename (string): filename for image

    Returns:
        np.array: grayscaled np-array (yes, breaks the one-function one-purpose rule, but PIL beckons)
    """
    image = Image.open(filename).convert("L")  
    return np.array(image) 