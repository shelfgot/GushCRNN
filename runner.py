import numpy as np
from Training import CRNN as crnn
from PIL import Image


def load_image(filename):
    """Loads our image as a NumPy array in grayscale."""
    image = Image.open(filename).convert("L")  
    return np.array(image) 

num_classes = 26
images = [np.random.rand(8, 8) for _ in range(10)]
labels = [np.eye(num_classes)[np.random.randint(0, num_classes)] for _ in range(10)]

model = crnn.CRNN(input_shape=(8, 8), num_classes=num_classes)
crnn.train(model, images, labels, batch_size=4, epochs=10)