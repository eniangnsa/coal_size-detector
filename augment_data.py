import tensorflow as tf
import numpy as np
from pathlib import Path
from src.data_load import load_params, load_data
import cv2

def augment_image(image, boxes, params):
    """Apply augmentation to an image and its bounding boxes."""
    # Convert image to TensorFlow tensor
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Apply random hue, saturation, and value adjustments
    image = tf.image.random_hue(image, params['augmentation']['hsv_h'])
    image = tf.image.random_saturation(image, lower=1 - params['augmentation']['hsv_s'], upper=1 + params['augmentation']['hsv_s'])
    image = tf.image.random_brightness(image, max_delta=params['augmentation']['hsv_v'])
    
    # Apply random horizontal flip
    if np.random.rand() < params['augmentation']['flip']:
        image = tf.image.flip_left_right(image)
        boxes[:, 1] = 1.0 - boxes[:, 1]  # Flip x_center for YOLO format
    
    # Apply random translation and scaling
    translate_x = params['augmentation']['translate'] * np.random.uniform(-1, 1)
    translate_y = params['augmentation']['translate'] * np.random.uniform(-1, 1)
    scale = 1.0 + params['augmentation']['scale'] * np.random.uniform(-1, 1)
    
    # Apply transformations to boxes
    boxes[:, 1] = (boxes[:, 1] + translate_x) * scale  # x_center
    boxes[:, 2] = (boxes[:, 2] + translate_y) * scale  # y_center
    boxes[:, 3] *= scale  # width
    boxes[:, 4] *= scale  # height
    
    # Clip boxes to [0, 1] range
    boxes = np.clip(boxes, 0, 1)
    
    return image.numpy().astype(np.uint8), boxes

def augment_data(params):
    """Augment images and labels."""
    image_files, label_files = load_data(params)
    augmented_dir = Path(params['data']['augmented_dir'])
    augmented_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (image_path, label_path) in enumerate(zip(image_files, label_files)):
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Load bounding boxes
        with open(label_path, 'r') as file:
            lines = file.readlines()
        boxes = np.array([list(map(float, line.strip().split())) for line in lines])
        
        # Apply augmentation
        augmented_image, augmented_boxes = augment_image(image, boxes, params)
        
        # Save augmented image
        output_image_path = augmented_dir / f"aug_{idx}.jpg"
        cv2.imwrite(str(output_image_path), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
        
        # Save augmented labels
        output_label_path = augmented_dir / f"aug_{idx}.txt"
        with open(output_label_path, 'w') as file:
            for box in augmented_boxes:
                file.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
        
        print(f"Saved augmented data: {output_image_path}, {output_label_path}")

def main():
    # Load parameters
    params = load_params()
    
    # Augment data
    augment_data(params)

if __name__ == "__main__":
    main()