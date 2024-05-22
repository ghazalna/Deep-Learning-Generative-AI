# Deep-Learning-Generative-AI: Computer Vision (Train a Neural Network with Images)

## Overview

This repository contains code for training a neural network to classify images from the Fashion MNIST dataset using TensorFlow and Keras. The project demonstrates fundamental concepts in computer vision and deep learning, including data preprocessing, model building, training, and evaluation.

## Contents

- `Computer_Vision(Train a Neural Network with images).ipynb`: Jupyter notebook containing the complete workflow from loading the dataset to evaluating the model.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Jupyter Notebook
- TensorFlow and Keras

Install the necessary packages using pip:

```bash
pip install tensorflow numpy matplotlib jupyter
```

### Running the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Deep-Learning-Generative-AI.git
   cd Deep-Learning-Generative-AI
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open `Computer_Vision(Train a Neural Network with images).ipynb` and run the cells to train and evaluate the neural network.

## Code Explanation

### 1. Load the Fashion MNIST Dataset

The dataset is loaded and split into training and test sets.

```python
import tensorflow as tf

fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()
```

### 2. Normalize the Images

Normalize the pixel values of the images to a range of 0 to 1.

```python
training_images = training_images / 255.0
test_images = test_images / 255.0
```

### 3. Build the Classification Model

A sequential neural network model is built with three dense layers.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

### 4. Train the Model

Train the model using the training data.

```python
model.fit(training_images, training_labels, epochs=5)
```

### 5. Evaluate the Model

Evaluate the model's performance on the test data.

```python
model.evaluate(test_images, test_labels)
```

### 6. Visualize and Predict

Visualize an image from the dataset and predict its label.

```python
import numpy as np
import matplotlib.pyplot as plt

index = 42006
np.set_printoptions(linewidth=320)
print(f'LABEL: {training_labels[index]}')
print(f'\nIMAGE PIXEL ARRAY: \n {training_images[index]}')
plt.imshow(training_images[index])

# Predict the label for a sample image
first_sample = training_images[:1]
predicted_probabilities = model.predict(first_sample)
predicted_label = tf.argmax(predicted_probabilities, axis=1).numpy()
print("Predicted Label for the first sample: ", predicted_label)
print("Probabilities: ", predicted_probabilities)
```

## Results

The model achieves an accuracy of approximately 88% on the test dataset after 5 epochs of training. The predicted labels for sample images are displayed along with their probabilities.


