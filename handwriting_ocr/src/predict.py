import easyocr
import cv2
import numpy as np
from PIL import Image
import os
import re
from difflib import get_close_matches
import nltk
from nltk.corpus import words

# Download required NLTK data
try:
    nltk.download('words', quiet=True)
except:
    print("Warning: NLTK words not available")

# Load English dictionary
WORD_LIST = set(words.words())

def enhance_image(image):
    """
    Enhanced preprocessing for better text recognition
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Increase image size
    height, width = gray.shape
    if height < 1000 or width < 1000:
        scale = 1000 / min(height, width)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, 
                         interpolation=cv2.INTER_CUBIC)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        11
    )
    
    # Remove small noise
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def correct_word(word):
    """
    Attempt to correct misspelled words using dictionary
    """
    if word.lower() in WORD_LIST:
        return word
    
    # Keep numbers and short words as is
    if word.isdigit() or len(word) <= 2:
        return word
    
    # Try to find close matches
    matches = get_close_matches(word.lower(), WORD_LIST, n=1, cutoff=0.8)
    
    if matches:
        # Preserve original capitalization
        if word[0].isupper():
            return matches[0].capitalize()
        return matches[0]
    
    return word

def clean_text(text):
    """
    Improved text cleaning and formatting
    """
    # Remove special characters and extra spaces
    text = re.sub(r'[^a-zA-Z0-9.,!? ]', '', text)
    text = ' '.join(text.split())
    
    # Fix common OCR errors
    replacements = {
        '0': 'o',
        '1': 'l',
        '|': 'l',
        '+': 't',
        '@': 'a',
        '#': 'h',
        '$': 's',
        '&': 'and',
        '_': ' ',
        '~': ' ',
        'z': 's',
        'y': 'y',
        'x': 'x',
        'w': 'w',
        'v': 'v',
        'u': 'u',
        't': 't',
        's': 's',
        'r': 'r',
        'q': 'q',
        'p': 'p',
        'o': 'o',
        'n': 'n',
        'm': 'm',
        'l': 'l',
        'k': 'k',
        'j': 'j',
        'i': 'i',
        'h': 'h',
        'g': 'g',
        'f': 'f',
        'e': 'e',
        'd': 'd',
        'c': 'c',
        'b': 'b',
        'a': 'a'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Fix spacing after punctuation
    text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)
    
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

def predict_text(image_path):
    """
    Improved text prediction with better settings
    """
    try:
        # Initialize EasyOCR with correct parameters
        reader = easyocr.Reader(
            ['en'],
            gpu=False  # Set to True if you have GPU support
        )
        
        # Read and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Enhance image
        processed_image = enhance_image(image)
        
        # Detect text with optimized parameters
        results = reader.readtext(
            processed_image,
            paragraph=True,
            contrast_ths=0.2,
            adjust_contrast=0.5,
            text_threshold=0.7,
            width_ths=0.7,
            height_ths=0.7,
            y_ths=0.5,
            x_ths=1.0,
            slope_ths=0.5
        )
        
        # Extract text
        text = ' '.join([result[1] for result in results])
        
        # Clean and format text
        cleaned_text = clean_text(text)
        
        return cleaned_text
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    test_dir = os.path.join(project_root, 'test_images')
    
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
                print(f"\nFailed to process {filename}")

if __name__ == "__main__":
    main()