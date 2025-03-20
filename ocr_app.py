import easyocr
import cv2
import os
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    """Preprocess the image for better OCR results"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)
    
    # Dilation to connect text components
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    
    return dilated

def recognize_text(image_path):
    """Recognize text in the image using EasyOCR"""
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'])
    
    # Preprocess image
    processed_image = preprocess_image(image_path)
    
    # Perform OCR
    results = reader.readtext(processed_image)
    
    # Extract text from results
    text = ' '.join([result[1] for result in results])
    
    return text

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # Process all images in the images directory
    image_dir = 'images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        print(f"Created {image_dir} directory. Please add your images there.")
        return
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            print(f"\nProcessing {filename}...")
            
            try:
                # Recognize text
                text = recognize_text(image_path)
                
                # Print results
                print(f"Recognized text: {text}")
                
                # Save to file
                output_file = os.path.join('output', f"{os.path.splitext(filename)[0]}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Saved text to {output_file}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main() 