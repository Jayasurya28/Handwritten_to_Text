import cv2
import numpy as np
from PIL import Image
import easyocr
import nltk
from nltk.corpus import words
import os

# Download required NLTK data
try:
    nltk.download('words', quiet=True)
except:
    print("Warning: NLTK words not available")

# Load English dictionary
WORD_LIST = set(words.words())

def preprocess_image(image_path):
    """
    Preprocess the image for better OCR results
    """
    # Read image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to preprocess the image
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Apply dilation to connect text components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    gray = cv2.dilate(gray, kernel, iterations=1)
    
    # Apply median blur to remove noise
    gray = cv2.medianBlur(gray, 3)
    
    return gray

def correct_word(word):
    """
    Correct misspelled words using dictionary lookup
    """
    if word.lower() in WORD_LIST:
        return word
    
    # Keep numbers and short words as is
    if word.isdigit() or len(word) <= 2:
        return word
    
    # Try to find close matches
    matches = nltk.edit_distance(word.lower(), list(WORD_LIST))
    if matches:
        return matches[0]
    
    return word

def clean_text(text):
    """
    Clean and format the recognized text
    """
    # Remove special characters and extra spaces
    text = ' '.join(text.split())
    
    # Split into sentences
    sentences = text.split('.')
    cleaned_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            # Correct words in sentence
            words = sentence.split()
            corrected_words = [correct_word(word) for word in words]
            sentence = ' '.join(corrected_words)
            
            # Capitalize first letter
            sentence = sentence[0].upper() + sentence[1:].lower()
            cleaned_sentences.append(sentence)
    
    return '. '.join(cleaned_sentences) + '.'

def recognize_text(image_path):
    """
    Recognize text from image using EasyOCR
    """
    try:
        # Initialize EasyOCR
        reader = easyocr.Reader(['en'])
        
        # Preprocess image
        processed_image = preprocess_image(image_path)
        
        # Recognize text
        results = reader.readtext(processed_image)
        
        # Extract text from results
        text = ' '.join([result[1] for result in results])
        
        # Clean and format text
        cleaned_text = clean_text(text)
        
        return cleaned_text
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def main():
    # Create images directory if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')
        print("Created images directory. Please add your images to this directory.")
        return
    
    # Process all images in the images directory
    image_files = [f for f in os.listdir('images') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No image files found in the images directory.")
        print("Please add your images to this directory and run the script again.")
        return
    
    for filename in image_files:
        image_path = os.path.join('images', filename)
        print(f"\nProcessing {filename}...")
        
        # Recognize text
        text = recognize_text(image_path)
        
        if text:
            print("\nRecognized Text:")
            print("-" * 50)
            print(text)
            print("-" * 50)
            
            # Save to output file
            output_file = os.path.join('output', f"{os.path.splitext(filename)[0]}.txt")
            os.makedirs('output', exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"\nText saved to {output_file}")
        else:
            print(f"Failed to process {filename}")

if __name__ == "__main__":
    main() 