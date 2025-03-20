import cv2
import numpy as np
import os

IMG_WIDTH, IMG_HEIGHT = 128, 32  # Standard input size

def preprocess_image(img_path):
    """
    Preprocess a single image for the CRNN model.
    
    Args:
        img_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Read image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    # Resize image
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    # Normalize pixel values
    img = img.astype(np.float32) / 255.0
    
    # Add channel dimension
    img = np.expand_dims(img, axis=-1)
    
    return img

def load_dataset(image_dir, labels_path):
    """
    Load and preprocess the dataset.
    
    Args:
        image_dir (str): Directory containing the images
        labels_path (str): Path to the labels file
        
    Returns:
        tuple: (numpy.ndarray, list) Preprocessed images and their corresponding text labels
    """
    images, texts = [], []
    
    # Read labels file
    with open(labels_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each line
    for line in lines:
        try:
            img_name, text = line.strip().split(" ", 1)
            img_path = os.path.join(image_dir, img_name)
            
            # Preprocess image
            img = preprocess_image(img_path)
            images.append(img)
            texts.append(text)
        except Exception as e:
            print(f"Error processing line: {line.strip()}")
            print(f"Error: {str(e)}")
            continue

    return np.array(images), texts 