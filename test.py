import os
import numpy as np
import tensorflow as tf
from PIL import Image

def prediction(model, test_image):
    result = model.predict(test_image)
    flower_categories = ['aster', 'daffodil', 'dahlia', 'daisy', 'dandelion', 'iris', 'orchid', 'rose', 'sunflower', 'tulip']
    max_prob_index = np.argmax(result[0])
    max_prob = result[0][max_prob_index]
    if max_prob >= 0.40:
        return flower_categories[max_prob_index]
    else:
        return "noprediction"

# Load the saved model
model = tf.keras.models.load_model('flowers.h5')

validate_dir = "./Validate"
total_images = 0
correct_predictions = 0
flower_categories = os.listdir(validate_dir)

for category in flower_categories:
    category_path = os.path.join(validate_dir, category)
    if os.path.isdir(category_path):
        # List all image files in the category directory
        image_files = os.listdir(category_path)
        
        for image_file in image_files:
            image_path = os.path.join(category_path, image_file)
            # Load and preprocess the image
            test_image = Image.open(image_path)
            test_image = test_image.resize((224, 224))  # Resize to match the model's input size
            test_image = np.array(test_image)
            test_image = test_image / 255.0  # Normalize the pixel values to be in the range [0, 1]
            test_image = np.expand_dims(test_image, axis=0)
            # Make prediction
            predicted_flower = prediction(model, test_image)
            # Increment the total image counter
            total_images += 1
            # Check if the prediction is correct
            if predicted_flower == category:
                correct_predictions += 1

# Calculate the accuracy as a percentage
accuracy = (correct_predictions / total_images) * 100
# Print the result
print(f"The model's accuracy on the validation set is {accuracy:.2f}%")
# Print the number of correct and incorrect predictions
print(f"Correct predictions: {correct_predictions}/{total_images}")
print(f"Incorrect predictions: {total_images - correct_predictions}/{total_images}")
