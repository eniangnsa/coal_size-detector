import argparse
import yaml
from typing import Text
import os
from pathlib import Path

def load_params(config_path="params.yaml"):
    with open(config_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

def load_data(params):
    """
    Load the image data from the specified directories
    """
    source_dir = Path(params["data"]["source_dir"])
    images_dir = source_dir / params["data"]["images_dir"]
    labels_dir = source_dir / params["data"]["labels_dir"]

    # Ensure that the files exist in the directories
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"The images or labels were not found: {images_dir}, {labels_dir}")
    
    # Get a list of the images and labels
    image_files = sorted(list(images_dir.glob("*.jpg")))
    label_files = sorted(list(labels_dir.glob("*.txt")))

    # Add normal annotations to the label files
    normal_path = source_dir / params['data']['normal_coal_dir']
    normal_names = sorted(normal_path.glob("*.jpg"))
    normal_labels = []
    for label_path in normal_names:
        # Get the filename without the extension
        stem = label_path.stem
        txt_filename = stem + ".txt"
        normal_labels.append(txt_filename)

    label_files.extend(normal_labels)

    # Verify that the number of images and labels match
    if len(image_files) != len(label_files):
        raise ValueError(
            f"Mismatch in the number of images ({len(image_files)}) and labels ({len(label_files)})"
        )
    return image_files, label_files

def rename_files(image_dir, label_dir, output_image_dir, output_label_dir, prefix="normal_size"):
    """
    Rename image and label files to have English names.
    
    Args:
        image_dir (str): Path to the directory containing the original images.
        label_dir (str): Path to the directory containing the original labels.
        output_image_dir (str): Path to save the renamed images.
        output_label_dir (str): Path to save the renamed labels.
        prefix (str): Prefix for the new filenames (e.g., "normal_size").
    """
    # Convert paths to Path objects
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_image_dir = Path(output_image_dir)
    output_label_dir = Path(output_label_dir)
    
    # Create output directories if they don't exist
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of image and label files
    image_files = sorted(list(image_dir.glob("*.jpg")))
    label_files = sorted(list(label_dir.glob("*.txt")))
    
    # Ensure the number of images and labels match
    if len(image_files) != len(label_files):
        raise ValueError(f"The number of images and labels does not match.  image files: {len(image_files)} and label files: {len(label_files)}")
    
    # Rename files
    for idx, (image_path, label_path) in enumerate(zip(image_files, label_files)):
        # Generate new filenames
        new_image_name = f"{prefix}_{idx + 1:03d}.jpg"
        new_label_name = f"{prefix}_{idx + 1:03d}.txt"
        
        # Define new paths
        new_image_path = output_image_dir / new_image_name
        new_label_path = output_label_dir / new_label_name
        
        # Rename (move) the files
        image_path.rename(new_image_path)
        label_path.rename(new_label_path)
        
        print(f"Renamed {image_path.name} -> {new_image_name}")
        print(f"Renamed {label_path.name} -> {new_label_name}")

def main():
    # Load parameters
    params = load_params()
    
    # Load data
    image_files, label_files = load_data(params)
    
    # Print some information
    print(f"Loaded {len(image_files)} images and {len(label_files)} labels.")
    print(f"First image: {image_files[0]}")
    print(f"First label: {label_files[0]}")
    
    # Rename files
    source_dir = Path(params["data"]["source_dir"])
    images_dir = source_dir / params["data"]["images_dir"]
    labels_dir = source_dir / params["data"]["labels_dir"]
    output_image_dir = source_dir / "images_renamed"
    output_label_dir = source_dir / "labels_renamed"
    
    rename_files(images_dir, labels_dir, output_image_dir, output_label_dir, prefix="normal_size")

if __name__ == "__main__":
    main()