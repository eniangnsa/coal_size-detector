import cv2
import albumentations as A
import numpy as np
from pathlib import Path

def read_yolo_label(label_path, image_width, image_height):
    """
    Read a YOLO format label file and convert bounding boxes to Albumentations format.
    YOLO format: [class_id, x_center, y_center, width, height] (normalized)
    Albumentations format: [x_min, y_min, x_max, y_max] (unnormalized)
    """
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found at {label_path}. Check the file path.")

    with open(label_path, 'r') as file:
        lines = file.readlines()
    
    boxes = []
    class_ids = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        
        # Convert YOLO format to Albumentations format
        x_min = (x_center - width / 2) * image_width
        y_min = (y_center - height / 2) * image_height
        x_max = (x_center + width / 2) * image_width
        y_max = (y_center + height / 2) * image_height
        
        boxes.append([x_min, y_min, x_max, y_max])
        class_ids.append(class_id)
    
    return boxes, class_ids

def write_yolo_label(label_path, boxes, class_ids, image_width, image_height):
    """
    Convert bounding boxes back to YOLO format and save to a label file.
    """
    with open(label_path, 'w') as file:
        for box, class_id in zip(boxes, class_ids):
            x_min, y_min, x_max, y_max = box
            
            # Convert Albumentations format to YOLO format
            x_center = ((x_min + x_max) / 2) / image_width
            y_center = ((y_min + y_max) / 2) / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height
            
            file.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def apply_augmentation(image, boxes, class_ids, augmentation_pipeline):
    """
    Apply augmentations to the image and bounding boxes.
    """
    transformed = augmentation_pipeline(image=image, bboxes=boxes, class_labels=class_ids)
    transformed_image = transformed['image']
    transformed_boxes = transformed['bboxes']
    transformed_class_ids = transformed['class_labels']
    
    return transformed_image, transformed_boxes, transformed_class_ids

def augment_data(source_images_dir, source_labels_dir, augmented_images_dir, augmented_labels_dir, prefix):
    """
    Apply augmentations to images and labels and save the augmented data.

    Args:
        source_images_dir (Path): Directory containing the source images.
        source_labels_dir (Path): Directory containing the source labels.
        augmented_images_dir (Path): Directory to save the augmented images.
        augmented_labels_dir (Path): Directory to save the augmented labels.
        prefix (str): Prefix for the augmented files (e.g., "large" or "normal").
    """
    # Create augmented directories if they don't exist
    augmented_images_dir.mkdir(parents=True, exist_ok=True)
    augmented_labels_dir.mkdir(parents=True, exist_ok=True)

    # Define individual augmentation pipelines
    augmentations = [
        A.Compose([
            A.Resize(640, 640),  # Resize to YOLOv8 input size
            A.HorizontalFlip(p=1.0),  # Always apply horizontal flip
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
        
        A.Compose([
            A.Resize(640, 640),
            A.Rotate(limit=15, p=1.0),  # Always apply rotation
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
        
        A.Compose([
            A.Resize(640, 640),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),  # Always adjust brightness/contrast
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
        
        A.Compose([
            A.Resize(640, 640),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),  # Always adjust hue/saturation
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
        
        A.Compose([
            A.Resize(640, 640),
            A.GaussNoise(var_limit=(10, 50), p=1.0),  # Always add Gaussian noise
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
    ]

    # Get all image and label files
    image_files = sorted(source_images_dir.glob("*.jpg"))
    label_files = sorted(source_labels_dir.glob("*.txt"))

    # Ensure the number of images and labels match
    if len(image_files) != len(label_files):
        raise ValueError(f"Mismatch between number of images ({len(image_files)}) and labels ({len(label_files)})")

    # Process each image and label pair
    for idx, (image_path, label_path) in enumerate(zip(image_files, label_files)):
        # Debug: Print the paths
        print(f"Processing image: {image_path}")
        print(f"Processing label: {label_path}")

        # Load the image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Unable to read image at {image_path}. Skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image_height, image_width, _ = image.shape

        # Load and convert bounding boxes
        boxes, class_ids = read_yolo_label(label_path, image_width, image_height)

        # Apply each augmentation independently
        for aug_idx, augmentation_pipeline in enumerate(augmentations):
            # Apply augmentation
            augmented_image, augmented_boxes, augmented_class_ids = apply_augmentation(image, boxes, class_ids, augmentation_pipeline)

            # Save the augmented image and labels
            output_image_path = augmented_images_dir / f"aug_{prefix}_{idx + 1:03d}_{aug_idx + 1:03d}.jpg"
            output_label_path = augmented_labels_dir / f"aug_{prefix}_{idx + 1:03d}_{aug_idx + 1:03d}.txt"

            # Ensure the augmented image is a NumPy array
            if not isinstance(augmented_image, np.ndarray):
                print(f"Warning: Augmented image is not a NumPy array. Skipping {image_path}...")
                continue

            # Save the augmented image
            cv2.imwrite(str(output_image_path), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
            write_yolo_label(output_label_path, augmented_boxes, augmented_class_ids, image_width, image_height)

            print(f"Augmented image saved to: {output_image_path}")
            print(f"Augmented label saved to: {output_label_path}")

def main():
    # Define source and destination directories for large data
    source_large_images_dir = Path("C:/Users/SCII1/Desktop/coal_size detector/data/large_dest").resolve()
    source_large_labels_dir = Path("C:/Users/SCII1/Desktop/coal_size detector/data/annotated_labels_dest").resolve()
    augmented_large_images_dir = Path("C:/Users/SCII1/Desktop/coal_size detector/data/augmented/augmented_large_image").resolve()
    augmented_large_labels_dir = Path("C:/Users/SCII1/Desktop/coal_size detector/data/augmented/augmented_large_label").resolve()

    # Define source and destination directories for normal data
    source_norm_image_dir = Path("C:/Users/SCII1/Desktop/coal_size detector/data/normal_dest").resolve()
    source_norm_label_dir = Path("C:/Users/SCII1/Desktop/coal_size detector/data/normal_label_dest").resolve()
    augmented_norm_image_dir = Path("C:/Users/SCII1/Desktop/coal_size detector/data/augmented/augmented_norm_image").resolve()
    augmented_norm_label_dir = Path("C:/Users/SCII1/Desktop/coal_size detector/data/augmented/augmented_norm_label").resolve()

    # Apply augmentations to large data
    print("Augmenting large data...")
    augment_data(source_large_images_dir, source_large_labels_dir, augmented_large_images_dir, augmented_large_labels_dir, prefix="large")

    # Apply augmentations to normal data
    print("Augmenting normal data...")
    augment_data(source_norm_image_dir, source_norm_label_dir, augmented_norm_image_dir, augmented_norm_label_dir, prefix="normal")

    print("All augmentations completed!")

if __name__ == "__main__":
    main()