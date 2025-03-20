import os
import tensorflow as tf
from mltu.dataProvider.data_provider import DataProvider
from mltu.preprocessors.image_preprocessor import ImagePreprocessor
from mltu.tensorflow.model import create_crnn_model, ctc_loss

# Configuration
BATCH_SIZE = 32
EPOCHS = 100
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32
VOCAB_SIZE = 95  # ASCII printable characters

# Paths
train_images_path = os.path.join("Datasets", "IAM_Sentences", "sentences")
train_labels_path = os.path.join("Datasets", "IAM_Sentences", "ascii", "sentences.txt")
model_save_path = os.path.join("Models", "04_sentence_recognition")

def main():
    # Create data provider
    data_provider = DataProvider(
        data_path=train_images_path,
        labels_path=train_labels_path,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        grayscale=True
    )
    
    # Create model
    model = create_crnn_model(
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1),
        num_classes=VOCAB_SIZE,
        lstm_units=128
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=ctc_loss,
        metrics=['accuracy']
    )
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_save_path, "model_{epoch:02d}_{val_loss:.2f}.h5"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
    ]
    
    # Create training data generator
    def train_generator():
        while True:
            for batch_idx in range(len(data_provider) // BATCH_SIZE):
                batch_images, batch_labels = data_provider.get_batch(batch_idx)
                processed_images = [preprocessor(img_path) for img_path in batch_images]
                yield tf.stack(processed_images), tf.stack(batch_labels)
    
    # Train model
    model.fit(
        train_generator(),
        steps_per_epoch=len(data_provider) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_split=0.2
    )
    
    # Save final model
    model.save(os.path.join(model_save_path, "final_model.h5"))

if __name__ == "__main__":
    main() 