import numpy as np

def random_rotation(image, max_angle=15):
    """Rotate image by small random angle

    Args:
        image (_type_): _description_
        max_angle (int, optional): _description_. Defaults to 15.

    Returns:
        _type_: _description_
    """
    angle = np.random.uniform(-max_angle, max_angle)
    center = np.array(image.shape) // 2
    rotated = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x, y = i - center[0], j - center[1]
            new_x = int(center[0] + x * np.cos(np.radians(angle)) - y * np.sin(np.radians(angle)))
            new_y = int(center[1] + x * np.sin(np.radians(angle)) + y * np.cos(np.radians(angle)))
            if 0 <= new_x < image.shape[0] and 0 <= new_y < image.shape[1]:
                rotated[new_x, new_y] = image[i, j]
    return rotated

def random_shift(image, max_shift=2):
    """Shift image randomly in x or y direction

    Args:
        image (_type_): _description_
        max_shift (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    shift_x, shift_y = np.random.randint(-max_shift, max_shift + 1, size=2)
    shifted = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_x, new_y = i + shift_x, j + shift_y
            if 0 <= new_x < image.shape[0] and 0 <= new_y < image.shape[1]:
                shifted[new_x, new_y] = image[i, j]
    return shifted

def add_random_noise(image, noise_level=0.05):
    """Add random noise to the image

    Args:
        image (_type_): _description_
        noise_level (float, optional): _description_. Defaults to 0.05.

    Returns:
        _type_: _description_
    """
    noise = np.random.normal(0, noise_level, image.shape)
    return np.clip(image + noise, 0, 1)

def augment_image(image):
    """Apply a random combination of random transformations

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    if np.random.rand() < 0.5:
        image = random_rotation(image)
    if np.random.rand() < 0.5:
        image = random_shift(image)
    if np.random.rand() < 0.5:
        image = add_random_noise(image)
    return image