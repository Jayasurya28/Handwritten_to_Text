import re
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    print("Warning: NLTK punkt not available")

def clean_text(text):
    """
    Clean and normalize text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Fix common OCR errors
    text = text.replace(' i ', ' I ')  # Fix single 'i' to 'I'
    text = re.sub(r'\bi\b', 'I', text)  # Fix single 'i' at word boundaries
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.!?,])([A-Za-z])', r'\1 \2', text)
    
    return text

def format_sentences(text):
    """
    Format text into proper sentences
    """
    sentences = sent_tokenize(text)
    formatted_sentences = []
    
    for sentence in sentences:
        # Capitalize first letter
        sentence = sentence.strip()
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
        
        # Add period if sentence doesn't end with punctuation
        if sentence and not sentence[-1] in '.!?':
            sentence += '.'
            
        if sentence:
            formatted_sentences.append(sentence)
    
    return ' '.join(formatted_sentences) 