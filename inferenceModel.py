import os
import tensorflow as tf
import numpy as np
from mltu.preprocessors.image_preprocessor import ImagePreprocessor

class HandwritingRecognizer:
    def __init__(self, model_path, image_width=128, image_height=32):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.preprocessor = ImagePreprocessor(
            target_size=(image_width, image_height),
            grayscale=True
        )
        
    def decode_predictions(self, pred):
        """Convert model predictions to text"""
        # Get character predictions
        pred = tf.argmax(pred, axis=-1)
        pred = pred.numpy()[0]
        
        # Convert indices to characters
        text = ""
        for p in pred:
            if p == 0:  # CTC blank
                continue
            text += chr(p + 31)  # Convert back to ASCII (starting from space character)
        
        return text.strip()
        
    def predict(self, image_path):
        """Predict text in an image"""
        # Preprocess image
        image = self.preprocessor(image_path)
        image = np.expand_dims(image, axis=0)
        
        # Get predictions
        predictions = self.model.predict(image)
        
        # Decode predictions to text
        text = self.decode_predictions(predictions)
        
        return text

def main():
    # Paths
    model_path = os.path.join("Models", "04_sentence_recognition", "final_model.h5")
    test_image_path = os.path.join("test_images")  # Directory containing test images
    
    # Create recognizer
    recognizer = HandwritingRecognizer(model_path)
    
    # Process all images in test directory
    if os.path.exists(test_image_path):
        for image_file in os.listdir(test_image_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(test_image_path, image_file)
                
                # Get prediction
                predicted_text = recognizer.predict(image_path)
                
                print(f"Image: {image_file}")
                print(f"Predicted Text: {predicted_text}")
                print("-" * 50)
    else:
        print(f"Test image directory not found: {test_image_path}")
        print("Please create the directory and add test images.")

if __name__ == "__main__":
    main() 