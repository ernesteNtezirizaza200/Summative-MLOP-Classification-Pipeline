"""
Prediction module for brain tumor classification.
Handles single image predictions and batch predictions.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import pickle
import os


class BrainTumorPredictor:
    """
    Predictor class for brain tumor classification.
    """
    
    def __init__(self, model_path='models/brain_tumor_model.h5', class_names_path='models/class_names.pkl'):
        """
        Initialize the predictor with a trained model.
        """
        self.model = None
        self.class_names = None
        self.img_size = (224, 224)
        
        if os.path.exists(model_path):
            self.load_model(model_path, class_names_path)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def load_model(self, model_path='models/brain_tumor_model.h5', class_names_path='models/class_names.pkl'):
        """
        Load the trained model and class names.
        """
        self.model = keras.models.load_model(model_path)
        
        if os.path.exists(class_names_path):
            with open(class_names_path, 'rb') as f:
                self.class_names = pickle.load(f)
        else:
            # Default class names if file doesn't exist
            self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        print(f"Model loaded successfully. Classes: {self.class_names}")
    
    def preprocess_image(self, image_path_or_array):
        """
        Preprocess a single image for prediction.
        """
        # Load image
        if isinstance(image_path_or_array, str):
            # Load from file path
            img = cv2.imread(image_path_or_array)
            if img is None:
                raise ValueError(f"Could not load image from {image_path_or_array}")
        else:
            # Assume it's already a numpy array
            img = image_path_or_array
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img = cv2.resize(img, self.img_size)
        
        # Normalize pixel values to [0, 1]
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image_path_or_array, return_probabilities=False):
        """
        Predict the class of a single image.
        
        Args:
            image_path_or_array: Path to image file or numpy array
            return_probabilities: If True, return all class probabilities
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        processed_img = self.preprocess_image(image_path_or_array)
        
        # Make prediction
        predictions = self.model.predict(processed_img, verbose=0)
        
        # Get predicted class
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Prepare result
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_index': int(predicted_class_idx)
        }
        
        # Add all probabilities if requested
        if return_probabilities:
            probabilities = {}
            for idx, class_name in enumerate(self.class_names):
                probabilities[class_name] = float(predictions[0][idx])
            result['probabilities'] = probabilities
        
        return result
    
    def predict_batch(self, image_paths, return_probabilities=False):
        """
        Predict classes for multiple images.
        
        Args:
            image_paths: List of image file paths
            return_probabilities: If True, return all class probabilities
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for img_path in image_paths:
            try:
                result = self.predict(img_path, return_probabilities)
                result['image_path'] = img_path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })
        
        return results


def load_predictor(model_path='models/brain_tumor_model.h5'):
    """
    Convenience function to load a predictor instance.
    """
    return BrainTumorPredictor(model_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prediction.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Load predictor
    predictor = load_predictor()
    
    # Make prediction
    result = predictor.predict(image_path, return_probabilities=True)
    
    print(f"\nPrediction for {image_path}:")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"\nAll Probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob:.4f}")

