import shutil
from pathlib import Path
import os

def consolidate_data(source_dirs):
    """Collect data from different directories into a single directory

    Args:
        source_dirs (list of tuples): List of (source_image_dir, source_label_dir, 
                        destination_image_subdir, destination_label_subdir) tuples.
    """
    #  copy data from source to desired location
    for source_image_dir, source_label_dir, dest_image_dir, dest_label_dir in source_dirs:
        # create destination directories if they don't exist
        dest_image_dir.mkdir(parents=True, exist_ok=True)
        dest_label_dir.mkdir(parents=True, exist_ok=True)

        # copy the images
        for image_path in source_image_dir.glob("*.jpg"):
            try:
                shutil.copy(image_path, dest_image_dir / image_path.name)
            except Exception as e:
                print(f"Error copying {image_path}: {e}")

        # copy the labels
        for label_path in source_label_dir.glob("*.txt"):
            try:
                shutil.copy(label_path, dest_label_dir / label_path.name)
            except Exception as e:
                print(f"Error copying {label_path}: {e}")

    print("Data Consolidation completed")

def main():
    source_dirs = [
        (Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/demo_test_large_image"), 
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/demo_test_large_label"), 
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/split_data/images/test"),
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/split_data/labels/test")),
        # normal test 
        (Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/demo_test_normal_image"),
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/demo_test_normal_label"),
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/split_data/images/test"),
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/split_data/labels/test")),
        # normal train
        (Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/demo_train_normal_image"),
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/demo_train_normal_label"),
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/split_data/images/train"),
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/split_data/labels/train")), 
        # large train
        (Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/demo_train_large_image"),
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/demo_train_large_label"),
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/split_data/images/train"),
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/split_data/labels/train")),
        # augmented large
        (Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/augmented/augmented_large_image"),
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/augmented/augmented_large_label"),
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/split_data/images/train"),
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/split_data/labels/train")),
        # augmented normal
        (Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/augmented/augmented_normal_image"),
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/augmented/augmented_normal_label"),
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/split_data/images/train"),
         Path("D:/Users/eniang.eniang/Desktop/coal_size-detector/data/split_data/labels/train"))
    ]

    consolidate_data(source_dirs)

if __name__ == "__main__":
    main()