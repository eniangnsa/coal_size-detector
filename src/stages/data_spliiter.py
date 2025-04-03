from pathlib import Path
import shutil

def train_test_split(
    large_image_source: str,
    large_label_source: str,
    normal_image_source: str,
    normal_label_source: str,
    output_base: str,
    train_size: int = 200,
    test_size: int = 100
):
    """
    Splits large and normal coal images into train and test sets.
    
    Args:
        large_image_source: Path to source directory for large coal images
        large_label_source: Path to source directory for large coal labels
        normal_image_source: Path to source directory for normal coal images
        normal_label_source: Path to source directory for normal coal labels
        output_base: Base directory where output folders will be created
        train_size: Number of samples for training set
        test_size: Number of samples for test set
    """
    # Convert paths to Path objects
    output_base = Path(output_base)
    large_image_source = Path(large_image_source)
    large_label_source = Path(large_label_source)
    normal_image_source = Path(normal_image_source)
    normal_label_source = Path(normal_label_source)
    
    # Create output directories
    output_dirs = {
        'train_large_image': output_base / "demo_train_large_image",
        'train_large_label': output_base / "demo_train_large_label",
        'train_normal_image': output_base / "demo_train_normal_image",
        'train_normal_label': output_base / "demo_train_normal_label",
        'test_large_image': output_base / "demo_test_large_image",
        'test_large_label': output_base / "demo_test_large_label",
        'test_normal_image': output_base / "demo_test_normal_image",
        'test_normal_label': output_base / "demo_test_normal_label"
    }
    
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get source files
    source_large_images = sorted(large_image_source.rglob("*.jpg"))
    source_large_labels = sorted(large_label_source.rglob("*.txt"))
    source_normal_images = sorted(normal_image_source.rglob("*.jpg"))
    source_normal_labels = sorted(normal_label_source.rglob("*.txt"))
    
    # Validate counts
    if len(source_large_images) != len(source_large_labels):
        raise ValueError(f"Mismatch: {len(source_large_images)} large images vs {len(source_large_labels)} labels")
    if len(source_normal_images) != len(source_normal_labels):
        raise ValueError(f"Mismatch: {len(source_normal_images)} normal images vs {len(source_normal_labels)} labels")
    
    total_needed = train_size + test_size
    if len(source_large_images) < total_needed or len(source_normal_images) < total_needed:
        raise ValueError(f"Not enough samples. Need {total_needed}, have {min(len(source_large_images), len(source_normal_images))}")
    
    # Copy training files
    for i in range(train_size):
        shutil.copy(source_large_images[i], output_dirs['train_large_image'])
        shutil.copy(source_large_labels[i], output_dirs['train_large_label'])
        shutil.copy(source_normal_images[i], output_dirs['train_normal_image'])
        shutil.copy(source_normal_labels[i], output_dirs['train_normal_label'])
    
    # Copy test files
    for i in range(train_size, train_size + test_size):
        shutil.copy(source_large_images[i], output_dirs['test_large_image'])
        shutil.copy(source_large_labels[i], output_dirs['test_large_label'])
        shutil.copy(source_normal_images[i], output_dirs['test_normal_image'])
        shutil.copy(source_normal_labels[i], output_dirs['test_normal_label'])
    
    print(f"Successfully split {train_size} train and {test_size} test samples for both large and normal coal.")

if __name__ == "__main__":
    # Example usage
    train_test_split(
        large_image_source="D:/Users/eniang.eniang/Desktop/coal_size-detector/data/large_dest",
        large_label_source="D:/Users/eniang.eniang/Desktop/coal_size-detector/data/annotated_labels_dest",
        normal_image_source="D:/Users/eniang.eniang/Desktop/coal_size-detector/data/normal_dest",
        normal_label_source="D:/Users/eniang.eniang/Desktop/coal_size-detector/data/normal_label_dest",
        output_base="D:/Users/eniang.eniang/Desktop/coal_size-detector/data",
        train_size=200,
        test_size=100
    )