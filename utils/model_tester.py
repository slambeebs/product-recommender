import tensorflow as tf
import numpy as np
import os

# Load the model
model = tf.keras.models.load_model("models/cnn_model.h5")

# Get class names from your folder structure
class_names = sorted(os.listdir("cnn_data"))
print("Class names:", class_names)

# Load one known training image
img_path = "cnn_data/20726/000005.jpg"  # change this to an actual file
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = tf.expand_dims(img_array, 0)  # shape (1, 128, 128, 3)

# Predict
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]

print("✅ Predicted class:", predicted_class)