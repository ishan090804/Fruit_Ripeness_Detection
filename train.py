import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Configuration
IMG_SIZE = 640
BATCH_SIZE = 4 
EPOCHS = 20
NUM_CLASSES = 4
CLASS_NAMES = ['overripe', 'ripe', 'rotten', 'unripe']

# Paths
TRAIN_DIR = 'banana_data/data/train'
VAL_DIR = 'data/test'
MODEL_DIR = 'models'

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def create_datasets():
    """Load and prepare training and validation datasets using ImageDataGenerator (TF 2.20 compatible)"""
    
    # Data augmentation for training - SAME AS YOUR ORIGINAL
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Only rescaling for validation - SAME AS YOUR ORIGINAL
    val_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    
    train_dataset = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_dataset = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_dataset, val_dataset

def build_model():
    """Build MobileNetV2-based transfer learning model - SAME AS YOUR ORIGINAL"""
    
    # Load pre-trained MobileNetV2 (without top classification layer)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build complete model - SAME AS YOUR ORIGINAL
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile model - SAME AS YOUR ORIGINAL
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Plot training and validation metrics"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    plt.show()
    
    print(f"\n✓ Training history plot saved to {MODEL_DIR}/training_history.png")

def main():
    """Main training pipeline"""
    
    print("="*60)
    print("Banana Ripeness Classifier - Training")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE} | Batch Size: {BATCH_SIZE}")
    print("="*60)
    
    # Load datasets
    print("\n1. Loading datasets...")
    train_ds, val_ds = create_datasets()
    
    print(f"   Training samples: {train_ds.samples}")
    print(f"   Validation samples: {val_ds.samples}")
    print(f"   Classes: {train_ds.class_indices}")
    
    # Build model
    print("\n2. Building MobileNetV2 model...")
    model = build_model()
    model.summary()
    
    # Setup callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n3. Training model for {EPOCHS} epochs...")
    print("-"*60)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n4. Evaluating model...")
    val_loss, val_accuracy = model.evaluate(val_ds)
    print(f"\n   Final Validation Accuracy: {val_accuracy*100:.2f}%")
    print(f"   Final Validation Loss: {val_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(MODEL_DIR, 'banana_classifier_final.keras')
    model.save(final_model_path)
    print(f"\n✓ Final model saved to {final_model_path}")
    
    # Plot training history
    print("\n5. Generating training plots...")
    plot_training_history(history)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModels saved in: {MODEL_DIR}/")
    print("- best_model.keras (best validation accuracy)")
    print("- banana_classifier_final.keras (final model)")

if __name__ == "__main__":
    main()