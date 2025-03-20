import tensorflow as tf
from tensorflow.keras import layers, Model

def create_crnn_model(input_shape, num_classes, lstm_units=128):
    """Create a CRNN (CNN + RNN) model for handwritten text recognition"""
    
    # Input layer
    inputs = layers.Input(shape=input_shape, name='input')
    
    # CNN Feature Extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((1, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Prepare feature sequence for RNN
    shape = x.get_shape()
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)
    
    # RNN layers
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    x = layers.Dropout(0.25)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes + 1, activation='softmax')(x)
    
    # Create and return model
    model = Model(inputs=inputs, outputs=outputs)
    return model

def ctc_loss(y_true, y_pred):
    """CTC loss function"""
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss 