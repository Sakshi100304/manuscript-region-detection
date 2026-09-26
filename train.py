from ultralytics import YOLO


def train_model():
    # Load YOLO pretrained model
    model = YOLO("yolo11n.pt")

    # Train the model
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=1024,
        batch=4,
        project="models",
        name="manuscript_detector_1024"
    )

    print("Training completed successfully.")


if __name__ == "__main__":
    train_model()