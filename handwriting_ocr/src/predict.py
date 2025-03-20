import cv2
import numpy as np
import easyocr
from PIL import Image
import os

def preprocess_image(image):
    """
    Preprocess the image for better OCR accuracy.
    """
    # Convert PIL Image to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Enhance contrast
    contrast = cv2.convertScaleAbs(denoised, alpha=1.5, beta=0)
    
    return contrast

def predict_text(image_path):
    """
    Predict text from an image using EasyOCR.
    """
    try:
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])
        
        # Read image using PIL
        image = Image.open(image_path)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Detect text
        results = reader.readtext(processed_image)
        
        # Extract and combine text
        text = '\n'.join([result[1] for result in results])
        
        return text.strip()
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def main():
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    test_dir = os.path.join(project_root, 'test_images')
    
    # Process all images in the test directory
    print("\nProcessing images...")
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, filename)
            print(f"\nProcessing {filename}...")
            
            predicted_text = predict_text(image_path)
            
            if predicted_text:
                print("\nRecognized Text:")
                print("--------------")
                print(predicted_text)
                print("--------------")
            else:
                print(f"\nFailed to process {filename}. Please check the file path and ensure all dependencies are installed correctly.")

if __name__ == "__main__":
    main() 