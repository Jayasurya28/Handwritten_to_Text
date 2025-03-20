import cv2
import numpy as np
import pytesseract
from PIL import Image

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    """
    Preprocess the image for better OCR accuracy.
    
    Args:
        image: PIL Image object
        
    Returns:
        PIL Image: Preprocessed image
    """
    # Convert PIL Image to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to preprocess the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Apply dilation to connect text components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    
    # Convert back to PIL Image
    return Image.fromarray(dilation)

def predict_text(image_path):
    """
    Predict text from an image using Tesseract OCR.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Predicted text
    """
    # Read image using PIL
    image = Image.open(image_path)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Convert to text using pytesseract with custom configuration
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(
        processed_image,
        config=custom_config,
        lang='eng'
    )
    
    return text.strip()

def main():
    # Test with the sample image
    test_image = "handwriting_ocr/dataset/images/sample.jpg"
    try:
        predicted_text = predict_text(test_image)
        print("\nPredicted Text:")
        print("--------------")
        print(predicted_text)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 