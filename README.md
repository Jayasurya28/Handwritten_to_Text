# Handwritten Text Recognition (HTR) System

This project implements a Handwritten Text Recognition system using a Convolutional Recurrent Neural Network (CRNN) with TensorFlow and Keras. The system can recognize handwritten text from images and convert it to digital text.

## Project Structure

```
handwriting_ocr/
│── dataset/              # Stores handwritten images and labels
│   ├── images/           # Handwritten text images
│   ├── labels.txt        # Corresponding text labels for images
│── model/                # Trained model storage
│   ├── crnn_model.h5     # Saved model
│── src/                  # Main Python scripts
│   ├── preprocess.py     # Image preprocessing
│   ├── train.py          # Model training
│   ├── predict.py        # Predict text from images
│── notebooks/            # Jupyter Notebooks (for testing)
│── requirements.txt      # Required dependencies
│── README.md             # Documentation
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Place your handwritten text images in the `dataset/images/` directory
2. Create a `labels.txt` file in the `dataset/` directory with the following format:
```
image1.jpg This is the text in image 1
image2.jpg This is the text in image 2
...
```

## Training the Model

To train the model on your dataset:

```bash
python src/train.py
```

The training script will:
- Load and preprocess the images
- Train the CRNN model
- Save the best model to `model/crnn_model.h5`

## Making Predictions

To predict text from images:

```bash
python src/predict.py
```

You can also use the `HandwritingRecognizer` class in your own code:

```python
from src.predict import HandwritingRecognizer

# Initialize the recognizer
recognizer = HandwritingRecognizer()

# Predict text from a single image
text = recognizer.predict_text("path/to/image.jpg")
print(text)

# Predict text from multiple images
texts = recognizer.predict_batch(["image1.jpg", "image2.jpg"])
for text in texts:
    print(text)
```

## Model Architecture

The CRNN model consists of:
- CNN layers for feature extraction
- Bidirectional LSTM layers for sequence processing
- Dense output layer for character prediction

## Requirements

- Python 3.8+
- TensorFlow 2.13.0
- Keras 2.13.1
- OpenCV
- NumPy
- Other dependencies listed in requirements.txt

## Notes

- The model expects grayscale images of size 128x32 pixels
- Supported characters include lowercase and uppercase letters, numbers, and spaces
- The model uses CTC (Connectionist Temporal Classification) loss for training
- Early stopping and model checkpointing are implemented to prevent overfitting 