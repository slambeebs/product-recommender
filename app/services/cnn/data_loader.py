import tensorflow as tf
import pathlib
import logging
from typing import Tuple, List

logging.basicConfig(level=logging.INFO)

def validate_images_with_tf(directory: str) -> None:
    """
    Removes unreadable images from the given directory.
    """
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    directory = pathlib.Path(directory)
    removed = 0

    for image_path in directory.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in supported_formats:
            try:
                raw = tf.io.read_file(str(image_path))
                _ = tf.io.decode_image(raw)
            except Exception:
                logging.warning(f"Removing invalid image: {image_path.name}")
                image_path.unlink()
                removed += 1

    logging.info(f"TF cleanup done. {removed} bad images removed.")

def load_datasets(data_dir: str, img_size: Tuple[int, int], batch_size: int):
    """
    Loads training and validation datasets from directory.
    """
    try:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=img_size,
            batch_size=batch_size
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=img_size,
            batch_size=batch_size
        )
    except Exception as e:
        logging.error(f"Dataset loading failed: {e}")
        raise

    return train_ds, val_ds, train_ds.class_names