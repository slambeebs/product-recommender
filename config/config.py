import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "IMG_SIZE": (128, 128),
    "BATCH_SIZE": int(os.getenv("BATCH_SIZE", 16)),
    "EPOCHS": int(os.getenv("EPOCHS", 10)),
    "DATA_DIR": os.getenv("DATA_DIR", "cnn_data"),
    "MODEL_PATH": os.getenv("MODEL_PATH", "models/cnn_model.h5")
}