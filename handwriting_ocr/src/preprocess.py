import cv2
import numpy as np
import os
from skimage import exposure

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

def enhance_handwriting(image):
    """
    Enhanced preprocessing specifically for handwriting
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10
    )
    
    # Remove small noise
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Invert back
    cleaned = cv2.bitwise_not(cleaned)
    
    return cleaned

def deskew(image):
    """
    Deskew the image to straighten text
    """
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = 90 + angle
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, 
        M, 
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated 