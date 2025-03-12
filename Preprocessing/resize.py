import numpy as np

def resize_character(image, target_size=(32, 32)):
    """Resize char to fixed size; keep aspect ratio w/o skew

    Args:
        image (np.array): array of pixels, N x 1
        target_size (tuple, optional): (width, height). Defaults to (32, 32)

    Returns:
        _type_: _description_
    """
    from scipy.ndimage import zoom
    
    aspect_ratio = image.shape[1] / image.shape[0]

    #destructure tuple
    new_height, new_width = target_size
    
    if aspect_ratio > 1:  # then wider than tall
        scale = new_width / image.shape[1]
    else:  # then taller than wide
        scale = new_height / image.shape[0]

    resized = zoom(image, scale)
    padded = np.zeros(target_size)
    # center the padding
    padded[:resized.shape[0], :resized.shape[1]] = resized  
    
    return padded
