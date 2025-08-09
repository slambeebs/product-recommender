from services.cnn.train import train_model

if __name__ == "__main__":
    # Path to training data directory
    data_dir = "cnn_data"
    model_path = "models/cnn_model.h5"

    # Train the model
    train_model(data_dir=data_dir, model_path=model_path, epochs=10)