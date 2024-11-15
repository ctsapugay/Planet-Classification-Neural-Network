import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES  # Import drag-and-drop
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
# ImageDataGenerator to rescale the images
IDG = ImageDataGenerator(rescale=1/255)

# Directory paths for training, testing, and validation data
test_path = '/Users/gregchu/Downloads/planets_data/testing'
train_path = '/Users/gregchu/Downloads/planets_data/training'
valid_path = '/Users/gregchu/Downloads/planets_data/validating'

# Image generators for training, testing, and validation
train = IDG.flow_from_directory(directory=train_path, target_size=(224, 224), class_mode='categorical', batch_size=10)
test = IDG.flow_from_directory(directory=test_path, target_size=(224, 224), class_mode='categorical', batch_size=10)
valid = IDG.flow_from_directory(directory=valid_path, target_size=(224, 224), class_mode='categorical', batch_size=10)

# Build the model
model = Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.2),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    keras.layers.Dropout(0.25),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    keras.layers.Dropout(0.25),
    Flatten(),
    Dense(units=11, activation='softmax')  # Assuming 11 classes in the dataset
])


model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(x=train, validation_data=test, epochs=10, verbose=2)

# Saves the model after training
model.save('/Users/gregchu/Downloads/v2_image_model/my_model.h5')  # Or use a path without .h5 for SavedModel format


def load_and_preprocess_image(image_path):
    # Load the image with the target size
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale the image as done for training
    return img_array

