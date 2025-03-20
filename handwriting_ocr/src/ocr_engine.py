from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

class HandwritingOCR:
    def __init__(self):
        # Load TrOCR model and processor
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        
        # Use GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("Using GPU for inference")
        else:
            print("Using CPU for inference")

    def recognize_text(self, image_path):
        """
        Recognize text from image using TrOCR
        """
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare image for model
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            if torch.cuda.is_available():
                pixel_values = pixel_values.to("cuda")
            
            # Generate text
            generated_ids = self.model.generate(pixel_values, max_length=128)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None 