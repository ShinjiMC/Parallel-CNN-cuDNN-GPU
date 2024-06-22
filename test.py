import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from PIL import Image

def prediction(model, test_image):
    result = model.predict(test_image)
    flower_categories = ['aster', 'daffodil', 'dahlia', 'daisy', 'dandelion', 'iris', 'orchid', 'rose', 'sunflower', 'tulip']
    for i in range(len(result[0])):
        if result[0][i] >= 0.42:
            return flower_categories[i]
    return "noprediction"
# Load the saved model
model = tf.keras.models.load_model('flowers.h5')

# Directory where test images are stored
test_dir = "./Test"

# List all files in the test directory
test_files = os.listdir(test_dir)

# Set a seed for reproducibility
seed_value = np.random.randint(0, 100)
rng = np.random.default_rng(seed_value)

random_file_filename = rng.choice(test_files)
file_dir = os.path.join(test_dir, random_file_filename)

test_img_files = os.listdir(file_dir)
random_img_filename = rng.choice(test_img_files)
img_dir = os.path.join(file_dir, random_img_filename)

test_image = Image.open(img_dir)
test_image = test_image.resize((224, 224))  # Resize to match the model's input size
test_image = np.array(test_image)
test_image = test_image / 255.0  # Normalize the pixel values to be in the range [0, 1]

test_image = np.expand_dims(test_image, axis=0)

# Make prediction using the defined function
predicted_flower = prediction(model, test_image)

# Print the prediction result
print(f"The flower in the image '{img_dir}' is predicted to be a {predicted_flower}.")