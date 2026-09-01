import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import sys


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Configuration
IMG_SIZE = 640
CLASS_NAMES = ['overripe', 'ripe', 'rotten', 'unripe']
MODEL_PATH = 'models/best_model.keras'

def load_and_preprocess_image(image_path):
    """Load and preprocess image using OpenCV"""

    # Load image using OpenCV
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image to model input size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize pixel values from 0-255 to 0-1
    img_array = img.astype(np.float32) / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def is_banana_image(image_path):
    """Check whether the uploaded image contains a banana"""

    img = cv2.imread(image_path)

    if img is None:
        return False

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    img = np.expand_dims(img.astype(np.float32), axis=0)
    img = preprocess_input(img)

    validator = MobileNetV2(weights='imagenet')

    predictions = validator.predict(img, verbose=0)

    decoded = decode_predictions(predictions, top=5)[0]

    for _, label, confidence in decoded:
        if label.lower() == "banana":
            return True

    return False

def predict_ripeness(image_path, model_path=MODEL_PATH):
    """Predict banana ripeness from image"""

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py")
        return

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Check if the image is a banana
    print("\nChecking whether the image is a banana...")

    if not is_banana_image(image_path):
        print("\n" + "="*60)
        print("INVALID IMAGE")
        print("="*60)
        print("\nPlease upload an image of a banana.")
        print("The ripeness model only supports banana images.")
        print("="*60)
        return

    print("Image is a banana. Proceeding with prediction...")

    print(f"\nLoading model from {model_path}...")
    model = keras.models.load_model(model_path)

    print(f"Processing image: {image_path}")
    img_array = load_and_preprocess_image(image_path)

    print("Making prediction...")
    predictions = model.predict(img_array, verbose=0)

    # Get predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = CLASS_NAMES[predicted_class_idx]

    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\nPredicted Class: {predicted_class.upper()}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("\nAll Class Probabilities:")
    print("-"*40)

    for i, class_name in enumerate(CLASS_NAMES):
        prob = predictions[0][i] * 100
        bar = "█" * int(prob / 2)
        print(f"{class_name:12s}: {prob:5.2f}% {bar}")

    print("="*60)

    # Interpretation
    if confidence > 0.8:
        print("\n✓ High confidence prediction")
    elif confidence > 0.6:
        print("\n⚠ Moderate confidence - consider getting more training data")
    else:
        print("\n⚠ Low confidence - model uncertain")

    return predicted_class, confidence, predictions[0]


def main():
    """Main function"""

    if len(sys.argv) < 2:
        print("\nUsage: python predict.py <path_to_image>")
        print("\nExample:")
        print("  python predict.py test_banana.jpg")
        print("  python predict.py data/testing/ripe/banana_01.jpg")
        return

    image_path = sys.argv[1]
    predict_ripeness(image_path)


if __name__ == "__main__":
    main()