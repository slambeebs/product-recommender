import tensorflow as tf
from tf_keras import layers, models
import os
from PIL import Image
import pathlib

# Image parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10

# ✅ TensorFlow-based image validation to remove unreadable images
def validate_images_with_tf(directory):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    directory = pathlib.Path(directory)
    removed = 0

    for image_path in directory.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in supported_formats:
            try:
                raw = tf.io.read_file(str(image_path))
                _ = tf.io.decode_image(raw)  # Let TensorFlow decode it
            except Exception:
                print(f"❌ Removing invalid image: {image_path.name}")
                image_path.unlink()
                removed += 1

    print(f"🧹 TF cleanup done. {removed} bad images removed.\n")

# ✅ Run image validation
validate_images_with_tf("cnn_data")

# ✅ Load dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'cnn_data',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'cnn_data',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("✅ Classes:", class_names)

# ✅ Normalize pixel values
normalization_layer = layers.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# ✅ Improve performance with prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ✅ Define CNN model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train the model
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ✅ Save the trained model
model.save("cnn_model.h5")
print("✅ Model saved to cnn_model.h5")
