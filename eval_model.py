from ultralytics import YOLO

def visualize_predictions(model_path, data_yaml, imgsz=640, conf=0.50):
    """
    Visualize predictions on the validation dataset.

    Args:
        model_path (str): Path to the trained model weights.
        data_yaml (str): Path to the dataset YAML file.
        imgsz (int): Image size for prediction. Default is 640.
        conf (float): Confidence threshold for predictions. Default is 0.25.
    """
    # Load the trained model
    model = YOLO(model_path)

    # Run predictions on the validation dataset
    results = model.predict(
        source=data_yaml,  # Path to the dataset YAML file
        imgsz=imgsz,       # Image size
        conf=conf,         # Confidence threshold
        save=True,         # Save prediction images
        device="0",        # Use GPU 0 (or "cpu" for CPU)
    )

    print("Predictions saved!")

def main():
    # Define the path to the trained model weights
    model_path = "D:/Users/eniang.eniang/Desktop/coal_size-detector/runs/detect/yolov8n_coal_detector10/weights/best.pt"  # Update this path to your trained model

    # Define the path to the dataset YAML file
    data_yaml = "data/split_data/images/test"

    # Visualize predictions
    visualize_predictions(model_path, data_yaml)

if __name__ == "__main__":
    main()

# from ultralytics import YOLO
# import numpy as np

# def evaluate_model(model_path, data_yaml, imgsz=640, conf=0.50):
#     """
#     Evaluate the model and compute IoU metrics.

#     Args:
#         model_path (str): Path to the trained model weights.
#         data_yaml (str): Path to the dataset YAML file.
#         imgsz (int): Image size for prediction. Default is 640.
#         conf (float): Confidence threshold for predictions. Default is 0.50.
#     """
#     # Load the trained model
#     model = YOLO(model_path)

#     # Run validation (evaluation) on the test dataset
#     metrics = model.val(
#         data=data_yaml,    # Path to the dataset YAML file
#         imgsz=imgsz,      # Image size
#         conf=conf,        # Confidence threshold
#         iou=0.50,         # IoU threshold for mAP calculation
#         device="0",       # Use GPU 0 (or "cpu" for CPU)
#         split="test",      # Evaluate on the test set
#     )

#     # Print IoU-related metrics
#     print("\nIoU and Detection Metrics:")
#     print(f"mAP@0.50 (IoU=0.50): {metrics.box.map:.4f}")
#     print(f"mAP@0.50:0.95 (IoU range [0.50, 0.95]): {metrics.box.map50:.4f} to {metrics.box.map95:.4f}")

#     # If you want per-class IoU, you can access detailed results
#     if hasattr(metrics.box, "iou"):
#         print("\nPer-class IoU:")
#         for class_id, iou in enumerate(metrics.box.iou):
#             print(f"Class {class_id}: IoU = {iou:.4f}")

# def main():
#     # Define the path to the trained model weights
#     model_path = "D:/Users/eniang.eniang/Desktop/coal_size-detector/runs/detect/yolov8n_coal_detector12/weights/best.pt"

#     # Define the path to the dataset YAML file
#     data_yaml = "D:/Users/eniang.eniang/Desktop/coal_size-detector/dataset.yaml"  # Ensure this points to your YAML file (not just the test images folder)

#     # Evaluate the model and compute IoU
#     evaluate_model(model_path, data_yaml)

# if __name__ == "__main__":
#     main()