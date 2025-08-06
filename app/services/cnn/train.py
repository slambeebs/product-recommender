import tensorflow as tf
import logging
from config.config import CONFIG
from .data_loader import validate_images_with_tf, load_datasets
from .model_builder import build_model

AUTOTUNE = tf.data.AUTOTUNE
logging.basicConfig(level=logging.INFO)

def train_model():
    """
    Main training pipeline entry point.
    """
    logging.info("Starting training pipeline...")

    validate_images_with_tf(CONFIG["DATA_DIR"])

    train_ds, val_ds, class_names = load_datasets(
        CONFIG["DATA_DIR"],
        CONFIG["IMG_SIZE"],
        CONFIG["BATCH_SIZE"]
    )

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = build_model(num_classes=len(class_names))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    logging.info("Training model...")
    model.fit(train_ds, validation_data=val_ds, epochs=CONFIG["EPOCHS"])

    model.save(CONFIG["MODEL_PATH"])
    logging.info(f"Model saved to {CONFIG['MODEL_PATH']}")

    return model