import numpy as np
from data_augment import augment_image
from Preprocessing import batch_preprocess

#utf-8 encoding shouldn't be a problem...

def load_and_preprocess_data(images, labels, num_classes, ALEF='א'):
    """Load images, preprocess them, segment characters, and prepare labels.

    Args:
        images (np.array): |X| =
        labels (np.array): |X| 
        num_classes (int, optional): classes

    Returns:
        images: original image list, prepocessed
    """
    processed_data, processed_labels = [], []

    for img, label in zip(images, labels):
        
        characters = batch_preprocess.process_handwritten_text(img)  

        # convert to one-hot labels
        one_hot_labels = [np.eye(num_classes)[ord(ch) - ord(ALEF)] for ch in label]  
        
        processed_data.extend(characters)
        processed_labels.extend(one_hot_labels)

    return np.array(processed_data), np.array(processed_labels)


# LOSS FUNCTIONS
# ;; using cross entropy loss function; see Bishop ;;
def cross_entropy_loss(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

# TRAINING LOOP

def train_crnn(model, images, labels, batch_size=4, epochs=100):
    """Train CRNN on preprocessed and segmented text images.

    Args:
        model (): [see methods]
        images (np.array): |X|
        labels (np.array): also |X|
        batch_size (int, optional): batch size for sequential character training. Defaults to 4.
        epochs (int, optional): epochs for training. Defaults to 100.
    """
    X_train, y_train = load_and_preprocess_data(images, labels)

    for epoch in range(epochs):
        loss = 0
        for i in range(0, len(X_train), batch_size):
            #set up X and y
            batch_imgs = np.array([augment_image(img) for img in images[i:i+batch_size]])
            batch_labels = y_train[i:i+batch_size]
            #forward pass through neural net
            output = model.forward(batch_imgs) 
            #use cross entropy loss. TODO: add other loss functions? 
            loss += cross_entropy_loss(output, batch_labels)  
            d_loss = output - batch_labels
            model.backward(d_loss)  
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss/len(X_train):.4f}")