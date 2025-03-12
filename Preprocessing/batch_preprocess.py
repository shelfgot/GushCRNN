from clean import *
from resize import *
from segment import *

def preprocess_image(image):
    image = adaptive_threshold(image)
    plt.imshow(image, cmap="gray")
    plt.show()
    image = median_filter(image)
    image = normalize_image(image)
    return image

def process_handwritten_text(image):
    """Preprocessing for handwritten text in an image

    Args:
        image (np.array): the image as an array of pixel values

    Returns:
        array (of images): the extracted character image arrays
    """
    image = preprocess_image(image)  
    characters = segment_characters(image)  
    characters = [resize_character(char) for char in characters] 
    return np.array(characters)  


if __name__ == "__main__":
    """For testing purposes!!
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    text_image = np.array(Image.open("test/test.jpeg").convert("L"))  
    
    # test
    processed_chars = process_handwritten_text(text_image)
    
    # show extracted characters
    fig, axes = plt.subplots(1, len(processed_chars), figsize=(10, 2))
    for i, char_img in enumerate(processed_chars):
        axes[i].imshow(char_img, cmap="gray")
        axes[i].axis("off")
    plt.show()

