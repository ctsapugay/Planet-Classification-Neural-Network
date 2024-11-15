
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Loads already trained model
model = load_model('/Users/gregchu/Downloads/v2_image_model/my_model.h5')
# Preprocessing and tests on given images

def load_and_preprocess_images(image_paths):
    images = []
    for image_path in image_paths:
        img = load_img(image_path, target_size=(224, 224))  # Adjust target_size to match your model
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        images.append(img_array)
    return np.array(images)  # Stack all images into a single NumPy array

test_image_paths = [
    '/Users/gregchu/Downloads/moon.jpeg', 
    '/Users/gregchu/Downloads/jupiter.jpeg',
    '/Users/gregchu/Downloads/pluto.jpeg', 
    '/Users/gregchu/Downloads/makemake.jpeg', 
    '/Users/gregchu/Downloads/mercury.jpeg', 
    '/Users/gregchu/Downloads/venus.jpeg', 
    '/Users/gregchu/Downloads/earth.jpg', 
    '/Users/gregchu/Downloads/neptune.jpeg', 
    '/Users/gregchu/Downloads/uranus.jpeg',
    '/Users/gregchu/Downloads/saturn.jpeg', 
    '/Users/gregchu/Downloads/mars.jpeg'
]


# Preprocess the batch of images
img_batch = load_and_preprocess_images(test_image_paths)

predictions = model.predict(img_batch)
class_names = ['Earth', 'Jupiter', 'Makemake', 'Mars', 'Mercury', 
               'Moon', 'Neptune', 'Pluto', 'Saturn', 'Uranus', 'Venus']

# Create the index_to_class dictionary
index_to_class = {i: class_names[i] for i in range(len(class_names))}

#predicted_class = np.argmax(prediction)  # Get the class with the highest probability
for i, prediction in enumerate(predictions):
    predicted_index = np.argmax(prediction)
    predicted_class = index_to_class[predicted_index]
    plt.imshow(img_batch[i])
    plt.title(f"Predicted Planet: {predicted_class}")
    plt.axis('off')
    plt.show()

