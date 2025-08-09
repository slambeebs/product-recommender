import tensorflow as tf
import numpy as np
from typing import Tuple, List

def load_trained_model(model_path: str) -> tf.keras.Model:
    """
    Loads a trained CNN model from file.
    """
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path: str, img_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Loads and preprocesses an image for prediction.
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_image(model: tf.keras.Model, image_path: str, class_names: List[str]) -> Tuple[str, float]:
    """
    Predicts the class of an image.

    Args:
        model (tf.keras.Model): The trained CNN model.
        image_path (str): Path to the image file.
        class_names (List[str]): List of class labels.

    Returns:
        Tuple[str, float]: Predicted class label and confidence score.
    """
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions[0])
    return class_names[predicted_idx], float(np.max(predictions[0]))