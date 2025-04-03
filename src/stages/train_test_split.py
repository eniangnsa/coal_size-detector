import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

def consolidate_data(source_dirs, consolidated_dir):
    """
    Consolidate data from multiple directories into a single directory.

    Args:
        source_dirs (list of tuples): List of (source_image_dir, source_label_dir, destination_subdir) tuples.
        consolidated_dir (Path): Path to the consolidated directory.
    """
    # Create consolidated directories
    consolidated_images_dir = consolidated_dir / "images"
    consolidated_labels_dir = consolidated_dir / "labels"
    consolidated_images_dir.mkdir(parents=True, exist_ok=True)
    consolidated_labels_dir.mkdir(parents=True, exist_ok=True)

    # Copy data from source directories to consolidated directory
    for source_image_dir, source_label_dir, destination_subdir in source_dirs:
        # Create subdirectories for images and labels
        dest_image_dir = consolidated_images_dir / destination_subdir
        dest_label_dir = consolidated_labels_dir / destination_subdir
        dest_image_dir.mkdir(parents=True, exist_ok=True)
        dest_label_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        for image_path in source_image_dir.glob("*.jpg"):
            shutil.copy(image_path, dest_image_dir / image_path.name)

        # Copy labels
        for label_path in source_label_dir.glob("*.txt"):
            shutil.copy(label_path, dest_label_dir / label_path.name)

    print("Data consolidation completed!")

def split_data(consolidated_dir, output_dir, test_size=0.2, random_state=42):
    """
    Split the consolidated data into train and test sets.

    Args:
        consolidated_dir (Path): Path to the consolidated directory.
        output_dir (Path): Path to save the train and test sets.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    """
    # Create output directories
    train_images_dir = output_dir / "images" / "train"
    train_labels_dir = output_dir / "labels" / "train"
    test_images_dir = output_dir / "images" / "test"
    test_labels_dir = output_dir / "labels" / "test"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    # Get all image and label files
    image_files = sorted((consolidated_dir / "images").rglob("*.jpg"))
    label_files = sorted((consolidated_dir / "labels").rglob("*.txt"))

    # Ensure the number of images and labels match
    if len(image_files) != len(label_files):
        raise ValueError(f"Mismatch between number of images ({len(image_files)}) and labels ({len(label_files)})")

    # Split the data into train and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        image_files, label_files, test_size=test_size, random_state=random_state
    )

    # Copy train data
    for image_path, label_path in zip(train_images, train_labels):
        shutil.copy(image_path, train_images_dir / image_path.name)
        shutil.copy(label_path, train_labels_dir / label_path.name)

    # Copy test data
    for image_path, label_path in zip(test_images, test_labels):
        shutil.copy(image_path, test_images_dir / image_path.name)
        shutil.copy(label_path, test_labels_dir / label_path.name)

    print("Data splitting completed!")

def main():
    # Define source directories
    source_dirs = [
        (Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/normal_dest"), Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/normal_label_dest"), "original_normal"),
        (Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/large_dest"), Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/annotated_labels_dest"), "original_large"),
        (Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/augmented/augmented_norm_image"), Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/augmented/augmented_norm_label"), "augmented_normal"),
        (Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/augmented/augmented_large_image"), Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/augmented/augmented_large_label"), "augmented_large"),
    ]

    # Define consolidated and output directories
    consolidated_dir = Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/consolidated_data")
    output_dir = Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/split_data")

    # Consolidate data
    consolidate_data(source_dirs, consolidated_dir)

    # Split data into train and test sets
    split_data(consolidated_dir, output_dir, test_size=0.2, random_state=42)

if __name__ == "__main__":
    main()