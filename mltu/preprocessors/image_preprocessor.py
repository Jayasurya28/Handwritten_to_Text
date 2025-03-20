import cv2
import numpy as np
from typing import Tuple, Union

class ImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (128, 32), grayscale: bool = True):
        self.target_size = target_size
        self.grayscale = grayscale
        
    def __call__(self, image_path: str) -> np.ndarray:
        """Process the image"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Convert to grayscale if needed
        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Add channel dimension if grayscale
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            
        # Resize image
        image = self._resize_maintain_aspect(image, self.target_size)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        return image
        
    def _resize_maintain_aspect(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image maintaining aspect ratio and padding if necessary"""
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        # Calculate scaling factor
        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height))
        
        # Create target array
        if len(image.shape) == 3:
            target = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
        else:
            target = np.zeros((target_height, target_width), dtype=image.dtype)
            
        # Calculate padding
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Place resized image in center
        if len(image.shape) == 3:
            target[y_offset:y_offset+new_height, x_offset:x_offset+new_width, :] = resized
        else:
            target[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
        return target 