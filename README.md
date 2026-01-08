import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os

# Parameters
IMAGE_SIZE = 128  # Use 128x128 images for simplicity
BATCH_SIZE = 16
EPOCHS = 50

def load_images_from_folder(folder, img_size=(IMAGE_SIZE, IMAGE_SIZE)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
    return np.array(images)

# Load your dataset
# X = load_images_from_folder('path/to/your/images')  # Provide your own image path here

# For demo purposes, create random data (delete in production)
X = np.random.rand(100, IMAGE_SIZE, IMAGE_SIZE, 3)

# Convert to LAB color space
X_lab = np.array([cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB) for img in X])
X_lab = X_lab / 255.

# Use L as input, AB as target
X_l = X_lab[..., 0:1]
X_ab = X_lab[..., 1:]

# Build autoencoder
input_l = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_l)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(2, (3, 3), activation='tanh', padding='same')(x)

autoencoder = keras.Model(input_l, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(X_l, X_ab, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_split=0.2)

# Example for colorization
def colorize(grayscale_img):
    gray_lab = cv2.cvtColor(grayscale_img, cv2.COLOR_BGR2LAB) / 255.
    gray_l = gray_lab[..., 0:1].reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
    pred_ab = autoencoder.predict(gray_l)
    output_lab = np.concatenate([gray_l[0], pred_ab[0]], axis=-1)
    output_lab = (output_lab * 255).astype(np.uint8)
    colorized = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    return colorized

# Save your model
# autoencoder.save('image_colorization_autoencoder.h5')# image-project
colorisation
