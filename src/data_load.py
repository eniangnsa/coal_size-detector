import yaml
from pathlib import Path
import shutil

def load_params(config_path="params.yaml"):
    """Load parameters from the YAML file."""
    with open(config_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

def rename_files(source_path, destination_path, prefix="normal_size"):
    """
    Renames all .jpg and .txt files in the specified directory and saves them in the destination directory.

    Args:
        source_path (str): Path to the directory containing the files to rename.
        destination_path (str): Path to the directory where renamed files will be saved.
        prefix (str): Prefix for the new file names. Default is "normal_size".

    Returns:
        tuple: A tuple containing two lists: (renamed_images, renamed_labels).
    """
    source_path = Path(source_path)
    destination_path = Path(destination_path)

    # Create the destination directory if it doesn't exist
    destination_path.mkdir(parents=True, exist_ok=True)

    # Initialize lists to store new names
    renamed_images = []
    renamed_labels = []

    # Process .jpg files (images)
    all_image_files = sorted(source_path.glob("*.jpg"))
    for idx, file in enumerate(all_image_files):
        new_name = f"{prefix}_{idx + 1:03d}.jpg"
        new_file_path = destination_path / new_name

        # Copy the file to the destination directory
        shutil.copy(file, new_file_path)

        # Append the new name to the list
        renamed_images.append(new_name)

    # Process .txt files (labels)
    all_text_files = sorted(source_path.glob("*.txt"))
    for idx, file in enumerate(all_text_files):
        new_name = f"{prefix}_{idx + 1:03d}.txt"
        new_file_path = destination_path / new_name

        # Copy the file to the destination directory
        shutil.copy(file, new_file_path)

        # Append the new name to the list
        renamed_labels.append(new_name)

    return renamed_images, renamed_labels

def create_empty_labels(image_dir, label_dir):
    """
    Create empty .txt label files for all images in the specified directory.
    
    Args:
        image_dir (str or Path): Directory containing the images.
        label_dir (str or Path): Directory to save the empty label files.
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    # Create the destination directory if it doesn't exist
    label_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files in the image directory
    all_images = sorted(image_dir.glob("*.jpg"))

    # Create empty .txt files for each image
    for image in all_images:
        # Get the stem of the image file (filename without extension)
        stem = image.stem

        # Create the corresponding .txt file name
        new_label = f"{stem}.txt"
        new_label_path = label_dir / new_label

        # Create an empty .txt file
        with open(new_label_path, 'w') as f:
            pass  # Create an empty file

        print(f"Created empty label file: {new_label_path}")

    print("All empty label files created successfully!")

def load_data(params):
    """Load data from the specified directories and rename files."""
    source_dir = Path(params['data']['source_dir'])
    normal_coal_dir = source_dir / params['data']['normal_coal_dir']
    large_coal_dir = source_dir / params['data']['large_coal_dir']
    labels_dir = source_dir / params['data']['labels_dir']

    # Ensure that the directories exist
    if not normal_coal_dir.exists() or not large_coal_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Charly, that thing no dey there. the images and labels no dey here: {labels_dir}, {normal_coal_dir}, {large_coal_dir}")
    
    # Check if the images and the labels have the same number of records
    normal_images = sorted(list(normal_coal_dir.glob("*.jpg")))
    large_images = sorted(list(large_coal_dir.glob("*.jpg")))
    annotated_labels = sorted(list(labels_dir.glob("*.txt")))

    if len(normal_images) != len(annotated_labels):
        raise ValueError(f"Chairman, matter dey ground. the normal images: {len(normal_images)} size no match the labels: {len(annotated_labels)}")
    
    # Define the destinations to save the renamed files
    normal_dest = Path("C:/Users/SCII1/Desktop/coal_size detector/data/normal_dest")
    large_dest = Path("C:/Users/SCII1/Desktop/coal_size detector/data/large_dest")
    annotated_labels_dest = Path("C:/Users/SCII1/Desktop/coal_size detector/data/annotated_labels_dest")

    # Create directories if they don't exist
    normal_dest.mkdir(parents=True, exist_ok=True)
    large_dest.mkdir(parents=True, exist_ok=True)
    annotated_labels_dest.mkdir(parents=True, exist_ok=True)

    # Call the rename function and rename the files
    renamed_normal_images, _ = rename_files(source_path=normal_coal_dir, destination_path=normal_dest)
    renamed_large_images, _ = rename_files(source_path=large_coal_dir, destination_path=large_dest, prefix="large_size")
    _, renamed_annotated_labels = rename_files(source_path=labels_dir, destination_path=annotated_labels_dest, prefix="large_size")

    # Create empty label files for images in the normal_dest folder
    normal_label_dest = Path("C:/Users/SCII1/Desktop/coal_size detector/data/normal_label_dest")
    create_empty_labels(normal_dest, normal_label_dest)

    return renamed_normal_images, renamed_large_images, renamed_annotated_labels

def main():
    # Load the params
    params = load_params()

    # Load the data
    renamed_norm, renamed_large, renamed_annotated_label = load_data(params)

    # Print some information
    print(f"Loaded normal images: {len(renamed_norm)}, large coal: {len(renamed_large)}, and labels: {len(renamed_annotated_label)}")

if __name__ == "__main__":
    main()