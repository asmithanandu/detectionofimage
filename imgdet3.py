import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab import files  # For uploading files

# CIFAR-10 class names
class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# Upload an image
uploaded = files.upload()  # Opens file picker

# Load and preprocess the image
for filename in uploaded.keys():
    img = cv2.imread(filename)  # Read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (32, 32))  # Resize to 32x32
    img = img / 255.0  # Normalize (same as training images)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Display the image with the predicted label
    plt.imshow(cv2.imread(filename))
    plt.title(f"Predicted: {class_names[predicted_class]}")
    plt.axis("off")
    plt.show()
