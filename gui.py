import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from PIL import Image
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget

class FlowerPredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        # Load the saved model
        self.model = tf.keras.models.load_model('flowers.h5')
        self.flower_categories = ['aster', 'daffodil', 'dahlia', 'daisy', 'dandelion', 'iris', 'orchid', 'rose', 'sunflower', 'tulip']
        # Directory where test images are stored
        self.test_dir = "./Test"

    def initUI(self):
        self.setWindowTitle('Flower Prediction')
        self.setFixedSize(1280, 720)
        
        self.main_layout = QHBoxLayout()

        # Left side layout (70%)
        self.left_layout = QVBoxLayout()
        
        self.title_label = QLabel('Classification of Flowers', self)
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setFont(QtGui.QFont('Arial', 20))
        self.left_layout.addWidget(self.title_label)
        
        self.image_container = QVBoxLayout()
        self.image_container.setAlignment(QtCore.Qt.AlignCenter)
        
        self.image_label = QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.image_container.addWidget(self.image_label)
        
        self.left_layout.addLayout(self.image_container)
        self.left_layout.setStretch(0, 1)
        self.left_layout.setStretch(1, 9)
        self.main_layout.addLayout(self.left_layout, 7)
        
        # Right side layout (30%)
        self.right_layout = QVBoxLayout()
        
        self.prediction_label = QLabel('Prediction: ', self)
        self.right_layout.addWidget(self.prediction_label)
        
        self.random_test_button = QPushButton('Test Random Image', self)
        self.random_test_button.clicked.connect(self.test_random_image)
        self.right_layout.addWidget(self.random_test_button)
        
        self.upload_button = QPushButton('Upload Image', self)
        self.upload_button.clicked.connect(self.upload_image)
        self.right_layout.addWidget(self.upload_button)
        
        self.main_layout.addLayout(self.right_layout, 3)
        
        self.setLayout(self.main_layout)
    
    def prediction(self, test_image):
        result = self.model.predict(test_image)
        max_prob_index = np.argmax(result[0])
        max_prob = result[0][max_prob_index]
        if max_prob >= 0.40:
            return self.flower_categories[max_prob_index]
        else:
            return "noprediction"

    def load_and_predict(self, img_path):
        test_image = Image.open(img_path)
        test_image = test_image.resize((224, 224))  # Resize to match the model's input size
        test_image = np.array(test_image)
        test_image = test_image / 255.0  # Normalize the pixel values to be in the range [0, 1]
        test_image = np.expand_dims(test_image, axis=0)
        predicted_flower = self.prediction(test_image)
        return predicted_flower

    def test_random_image(self):
        # List all files in the test directory
        test_files = os.listdir(self.test_dir)
        seed_value = np.random.randint(0, 100)
        rng = np.random.default_rng(seed_value)
        random_file_filename = rng.choice(test_files)
        file_dir = os.path.join(self.test_dir, random_file_filename)
        test_img_files = os.listdir(file_dir)
        random_img_filename = rng.choice(test_img_files)
        img_dir = os.path.join(file_dir, random_img_filename)
        predicted_flower = self.load_and_predict(img_dir)
        self.display_image(img_dir)
        self.prediction_label.setText(f'Prediction: {predicted_flower}\nFile: {img_dir}')

    def upload_image(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Images (*.png *.xpm *.jpg *.jpeg)", options=options)
        if filename:
            predicted_flower = self.load_and_predict(filename)
            self.display_image(filename)
            self.prediction_label.setText(f'Prediction: {predicted_flower}')
    
    def display_image(self, img_path):
        pixmap = QtGui.QPixmap(img_path)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    ex = FlowerPredictionApp()
    ex.show()
    sys.exit(app.exec_())
