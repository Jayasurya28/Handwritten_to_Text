# Handwritten Text Recognition

This is a simple and efficient tool to convert handwritten text from images into digital text format.

## Features

- Converts handwritten text from images to digital text
- Supports multiple image formats (PNG, JPG, JPEG)
- Preprocesses images for better recognition
- Corrects common OCR errors
- Formats output text with proper capitalization and punctuation
- Saves results to text files

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows (PowerShell):
```bash
.\venv\Scripts\activate
```
- Windows (Command Prompt):
```bash
venv\Scripts\activate.bat
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Create an `images` folder in the project directory
2. Place your handwritten text images in the `images` folder
3. Run the script:
```bash
python ocr_app.py
```
4. The recognized text will be:
   - Displayed in the console
   - Saved in the `output` folder as text files

## Project Structure

```
.
├── images/           # Place your images here
├── output/          # Recognized text files will be saved here
├── venv/            # Virtual environment
├── ocr_app.py       # Main script
├── requirements.txt # Required packages
└── README.md        # This file
```

## Notes

- Supported image formats: PNG, JPG, JPEG
- For best results, use clear, well-lit images
- The script will automatically create the `images` and `output` folders if they don't exist 