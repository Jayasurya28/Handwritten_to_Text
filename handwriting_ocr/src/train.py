import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, LSTM, Bidirectional, Input, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
from preprocess import load_dataset, IMG_WIDTH, IMG_HEIGHT
from encode_labels import encode_text, characters

def build_crnn():
    """
    Build the CRNN model architecture.
    
    Returns:
        tensorflow.keras.Model: The compiled CRNN model
    """
    # Input layer
    input_layer = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1))

    # CNN layers
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    # Reshape for LSTM
    x = Reshape(target_shape=(32, 512))(x)

    # LSTM layers
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.2)(x)

    # Output layer
    output = Dense(len(characters) + 1, activation='softmax')(x)

    # Create and compile model
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def prepare_training_data(images, texts):
    """
    Prepare training data by encoding text labels.
    
    Args:
        images (numpy.ndarray): Preprocessed images
        texts (list): List of text labels
        
    Returns:
        tuple: (numpy.ndarray, numpy.ndarray) Encoded images and labels
    """
    encoded_texts = []
    max_length = max(len(text) for text in texts)
    
    for text in texts:
        # Encode text
        encoded = encode_text(text)
        # Pad with zeros if necessary
        if len(encoded) < max_length:
            encoded.extend([0] * (max_length - len(encoded)))
        encoded_texts.append(encoded)
    
    return images, np.array(encoded_texts)

def main():
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    image_dir = "dataset/images"
    labels_path = "dataset/labels.txt"
    images, texts = load_dataset(image_dir, labels_path)
    
    # Prepare training data
    print("Preparing training data...")
    X, y = prepare_training_data(images, texts)
    
    # Build and train model
    print("Building model...")
    model = build_crnn()
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        "model/crnn_model.h5",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X, y,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[checkpoint, early_stopping]
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main() 