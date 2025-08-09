import logging
import tensorflow as tf
from typing import Optional
from .data_loader import validate_images_with_tf, load_datasets
from .model_builder import build_model

logging.basicConfig(level=logging.INFO)
AUTOTUNE = tf.data.AUTOTUNE

def train_model(data_dir: str, model_path: str, epochs: int = 10) -> tf.keras.Model:
    """
    Train a CNN model on the given dataset.

    Args:
        data_dir (str): Path to dataset directory.
        model_path (str): File path to save the trained model.
        epochs (int): Number of training epochs.

    Returns:
        tf.keras.Model: The trained model.
    """
    logging.info("Validating images...")
    validate_images_with_tf(data_dir)

    logging.info("Loading datasets...")
    train_ds, val_ds, class_names = load_datasets(
        data_dir,
        img_size=(128, 128),
        batch_size=32
    )

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    logging.info("Building model...")
    model = build_model(input_shape=(128, 128, 3), num_classes=len(class_names))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    logging.info("Starting training...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    logging.info(f"Saving model to {model_path}...")
    model.save(model_path)

    logging.info("Training complete!")
    return model