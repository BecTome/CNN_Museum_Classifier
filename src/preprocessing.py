from tensorflow import keras
import numpy as np
import keras.backend as K
# Generador de datos personalizado basado en Sequence
class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_data = self.data[start:end]
        batch_labels = self.labels[start:end]
        return batch_data, batch_labels

    def on_epoch_end(self):
        # Clear the Keras session to release GPU memory
        K.clear_session()

def center_crop(image, target_width, target_height):
    """
    Perform a center crop on an image.

    Args:
    - image: Input image (NumPy array or OpenCV image).
    - target_width: Width of the target crop.
    - target_height: Height of the target crop.

    Returns:
    - Cropped image.
    """
    if isinstance(image, str):
        image = plt.imread(image)

    if image is None:
        raise ValueError("Image not found.")

    h, w, _ = image.shape

    # Calculate the coordinates for the center crop
    left = (w - target_width) // 2
    top = (h - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    # Perform the center crop
    cropped_image = image[top:bottom, left:right]

    return cropped_image

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Test the center crop function
    image = center_crop('input/data/raw/2279953-45075.jpg', 50, 50)
    image = image.reshape()
    plt.imshow(image)
    plt.show()