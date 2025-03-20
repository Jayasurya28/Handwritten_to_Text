# Handwritten Text Recognition

A deep learning model for recognizing handwritten text in images using a CRNN (CNN + RNN) architecture with CTC loss.

## Project Structure

```
.
├── Datasets/
│   └── IAM_Sentences/
│       ├── ascii/
│       └── sentences/
├── Models/
│   └── 04_sentence_recognition/
├── mltu/
│   ├── augmentors/
│   ├── dataProvider/
│   ├── inferenceModel/
│   ├── preprocessors/
│   ├── tensorflow/
│   ├── torch/
│   └── utils/
├── test_images/
├── configs.py
├── inferenceModel.py
├── model.py
├── train.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix/MacOS:
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the IAM Handwriting Database:
   - Visit [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
   - Register for an account
   - Download the "sentences.tgz" file
   - Extract to `Datasets/IAM_Sentences/`

## Training

1. Prepare your dataset:
   - Place training images in `Datasets/IAM_Sentences/sentences/`
   - Place corresponding labels in `Datasets/IAM_Sentences/ascii/sentences.txt`

2. Train the model:
   ```bash
   python train.py
   ```

The model will be saved in `Models/04_sentence_recognition/`.

## Inference

1. Place test images in the `test_images/` directory

2. Run inference:
   ```bash
   python inferenceModel.py
   ```

## Model Architecture

- CNN layers for feature extraction
- Bidirectional LSTM layers for sequence processing
- CTC loss for text recognition
- Input shape: (32, 128, 1) - (height, width, channels)
- Output: ASCII text

## Requirements

- Python 3.8+
- TensorFlow 2.12+
- OpenCV 4.8+
- NumPy 1.24+
- Additional dependencies in requirements.txt

## License

This project is licensed under the MIT License - see the LICENSE file for details. 