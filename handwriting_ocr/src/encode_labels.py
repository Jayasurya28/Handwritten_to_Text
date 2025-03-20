# Define the character set
characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
char_to_num = {c: i for i, c in enumerate(characters)}
num_to_char = {i: c for c, i in char_to_num.items()}

def encode_text(text):
    """
    Convert text to sequence of numbers.
    
    Args:
        text (str): Input text to encode
        
    Returns:
        list: List of encoded characters
    """
    return [char_to_num[c] for c in text]

def decode_text(encoded):
    """
    Convert sequence of numbers back to text.
    
    Args:
        encoded (list): List of encoded characters
        
    Returns:
        str: Decoded text
    """
    return "".join([num_to_char[i] for i in encoded])

def create_character_maps():
    """
    Create character mapping dictionaries.
    
    Returns:
        tuple: (dict, dict) Character to number and number to character mappings
    """
    return char_to_num, num_to_char 