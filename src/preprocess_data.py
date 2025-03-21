import cv2
import numpy as np
from pathlib import Path
from data_load import load_data, load_params

def preprocess_image(image, target_size):
    return cv2.resize(image, target_size)

def preprocess_data(params):
    image_files, label_files = load_data(params)
    processed_dir = Path(params['data']['processed_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    

    target_size = tuple(params['preprocessing']['resize'])

    for idx, (image_path, label_path) in enumerate(zip(image_files, label_files)):
        # load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # preprocess the image
        preprocessed_image = preprocess_image(image, target_size)

        # save the processed image
        output_image_path = processed_dir /params['processed_image']/ f"processed_{idx}.jpg"
        cv2.imwrite(str(output_image_path), cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2BGR))

        # copy the label files
        output_label_path = processed_dir /params['processed_labels'] /f"processed_{idx}.txt"
        with open(label_path, "r") as src, open(output_label_path, "w") as dest:
            dest.write(src.read())

        print(f"Saved processed data: {output_image_path}, {output_label_path}")

    def main():
        # load the parameters
        params = load_params()

        # preprocess the data
        preprocess_data(params)

    if __name__ == "__main__":
        main()