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

    # ensure that the files exist in the directories
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"the images or labels were not found{images_dir}, {labels_dir}")
    
    # get a list of the images and labels
    image_files = sorted(list(images_dir.glob("*.jpg*")))
    label_files = sorted(list(labels_dir.glob("*.*")))

    # Verify that the number of images and labels match
    if len(image_files) != len(label_files):
        raise ValueError(
            f"My guy, there's a mismatch in the number of images: {len(image_files)} and the labels: {len(label_files)}"
            )
    return image_files, label_files

def main():
    # Load parameters
    params = load_params()
    
    # Load data
    image_files, label_files = load_data(params)
    
    # Print some information
    print(f"Loaded {len(image_files)} images and {len(label_files)} labels.")
    print(f"First image: {image_files[0]}")
    print(f"First label: {label_files[0]}")

if __name__ == "__main__":
    main()


