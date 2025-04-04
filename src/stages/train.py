from ultralytics import YOLO

def train_yolov8n(data_yaml, epochs=100, imgsz=640, batch=32, name="yolov8n_train"):
    """
    Train a YOLOv8n model using the specified dataset.

    Args:
        data_yaml (str): Path to the dataset YAML file.
        epochs (int): Number of training epochs. Default is 50.
        imgsz (int): Image size for training. Default is 640.
        batch (int): Batch size. Default is 16.
        name (str): Name of the training run. Default is "yolov8n_train".
    """
    # Load the YOLOv8n model
    model = YOLO("D:/Users/eniang.eniang/Desktop/coal_size-detector/yolo11n.pt")  # Load a pretrained YOLOv8n model

    # Train the model
    results = model.train(
        data=data_yaml,  # Path to the dataset YAML file
        epochs=epochs,   # Number of training epochs
        imgsz=imgsz,     # Image size
        batch=batch,     # Batch size
        name=name,       # Name of the training run
        device=0
    )

    # Print training results
    print("Training completed!")
    print(results)

def main():
    # Define the path to the dataset YAML file
    data_yaml = "D:/Users/eniang.eniang/Desktop/coal_size-detector/dataset.yaml"

    # Train the YOLOv8n model
    train_yolov8n(data_yaml, epochs=100, imgsz=640, batch=64, name="yolov8n_coal_detector")

if __name__ == "__main__":
    main()