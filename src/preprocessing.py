"""
Data preprocessing module for brain tumor MRI classification.
Includes feature extraction and data preparation.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle


def extract_image_features(image_path):
    """
    Extract features from a single image.
    Returns a dictionary of features including:
    - Histogram features
    - Statistical features
    - Texture features (using GLCM)
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        img_resized = cv2.resize(gray, (224, 224))
        
        features = {}
        
        # 1. Histogram features
        hist = cv2.calcHist([img_resized], [0], None, [256], [0, 256])
        features['hist_mean'] = np.mean(hist)
        features['hist_std'] = np.std(hist)
        features['hist_median'] = np.median(hist)
        features['hist_max'] = np.max(hist)
        features['hist_min'] = np.min(hist)
        
        # 2. Statistical features
        features['mean_intensity'] = np.mean(img_resized)
        features['std_intensity'] = np.std(img_resized)
        features['median_intensity'] = np.median(img_resized)
        features['min_intensity'] = np.min(img_resized)
        features['max_intensity'] = np.max(img_resized)
        features['variance'] = np.var(img_resized)
        
        # 3. Texture features (simplified)
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(img_resized, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_resized, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        
        # 4. Image dimensions (already known, but for consistency)
        features['width'] = img_resized.shape[1]
        features['height'] = img_resized.shape[0]
        
        # 5. Aspect ratio
        features['aspect_ratio'] = features['width'] / features['height']
        
        # 6. Image path for reference
        features['image_path'] = image_path
        
        return features
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def extract_features_from_directory(data_dir, output_csv='image_features.csv'):
    """
    Extract features from all images in a directory structure.
    Expected structure: data_dir/class_name/image.jpg
    """
    all_features = []
    labels = []
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"Found {len(class_dirs)} classes: {class_dirs}")
    
    for class_name in class_dirs:
        class_path = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Processing {len(image_files)} images from {class_name}...")
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            features = extract_image_features(img_path)
            
            if features is not None:
                features['class'] = class_name
                all_features.append(features)
                labels.append(class_name)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    # Save to CSV
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Features saved to {output_csv}")
        print(f"Total features extracted: {len(df)}")
        print(f"Feature columns: {df.columns.tolist()}")
    
    return df


def prepare_data_for_training(data_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Prepare data generators for training using Keras ImageDataGenerator.
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator


def get_class_names(data_dir):
    """
    Get class names from directory structure.
    """
    class_dirs = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d))]
    return sorted(class_dirs)


if __name__ == "__main__":
    # Create processed directory if it doesn't exist
    import os
    os.makedirs('data/processed', exist_ok=True)
    
    # Extract features from training data
    print("Extracting features from training data...")
    train_df = extract_features_from_directory('data/train', 'data/processed/image_features_train.csv')
    
    print("\nExtracting features from test data...")
    test_df = extract_features_from_directory('data/test', 'data/processed/image_features_test.csv')
    
    print("\nFeature extraction complete!")

