import cv2
import numpy as np
import easyocr
import os

def preprocess_image(image_path):
    """
    Enhanced preprocessing for clear handwritten text
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Could not read image")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to larger size for better recognition
    scale_factor = 2.0
    enlarged = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16,16))
    enhanced = clahe.apply(enlarged)
    
    # Denoise while preserving edges
    denoised = cv2.bilateralFilter(enhanced, 15, 35, 35)
    
    # Sharpen the image
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,  # Larger block size
        15   # Higher C value
    )
    
    # Morphological operations to connect text components
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def recognize_text(image_path):
    """
    Recognize text using EasyOCR with optimized settings
    """
    try:
        # Initialize EasyOCR with better defaults
        reader = easyocr.Reader(['en'], gpu=False, recog_network='english_g2')
        
        # Preprocess image
        processed_image = preprocess_image(image_path)
        
        # Save debug image
        os.makedirs('debug', exist_ok=True)
        cv2.imwrite('debug/preprocessed.png', processed_image)
        
        # Recognize text with optimized parameters
        results = reader.readtext(
            processed_image,
            paragraph=True,
            detail=0,
            contrast_ths=0.3,
            adjust_contrast=0.7,
            text_threshold=0.6,
            width_ths=0.8,
            height_ths=0.8,
            add_margin=0.15,
            mag_ratio=2.0
        )
        
        if not results:
            return None
            
        # Join results and clean up
        text = ' '.join(results)
        text = text.strip()
        
        # Remove any non-alphanumeric characters except spaces and punctuation
        text = ''.join(char for char in text if char.isalnum() or char.isspace() or char in '.,')
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def main():
    # Create directories
    for dir_name in ['images', 'output', 'debug']:
        os.makedirs(dir_name, exist_ok=True)
    
    # Process images
    image_files = [f for f in os.listdir('images') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No image files found in the images directory.")
        return
    
    for filename in image_files:
        image_path = os.path.join('images', filename)
        print(f"\nProcessing {filename}...")
        
        text = recognize_text(image_path)
        
        if text:
            print("\nRecognized Text:")
            print("-" * 50)
            print(text)
            print("-" * 50)
            
            # Save output
            output_file = os.path.join('output', f"{os.path.splitext(filename)[0]}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"\nText saved to {output_file}")
        else:
            print(f"Failed to process {filename}")

if __name__ == "__main__":
    main()
