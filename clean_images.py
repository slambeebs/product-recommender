import os
from PIL import Image

def clean_invalid_images(root_dir):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    removed_files = []

    for class_folder in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)
            ext = os.path.splitext(file_path)[-1].lower()
            if ext not in valid_extensions:
                os.remove(file_path)
                removed_files.append(file_path)
                continue
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Check if image is readable
            except Exception:
                os.remove(file_path)
                removed_files.append(file_path)

    print(f"✅ Cleaned {len(removed_files)} invalid files.")
    if removed_files:
        print("🗑 Removed files:")
        for f in removed_files:
            print(f" - {f}")

# Run cleanup before training
clean_invalid_images("cnn_data")
