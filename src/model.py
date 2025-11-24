"""
Model creation and training module for brain tumor classification.
Uses transfer learning with pre-trained models and includes comprehensive evaluation.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
try:
    from src.preprocessing import prepare_data_for_training, get_class_names
except ImportError:
    from preprocessing import prepare_data_for_training, get_class_names


class BrainTumorClassifier:
    """
    Brain Tumor Classification Model using Transfer Learning.
    """
    
    def __init__(self, img_size=(224, 224), num_classes=4, base_model_name='MobileNetV2', models_dir=None):
        self.img_size = img_size
        self.num_classes = num_classes
        self.base_model_name = base_model_name
        self.model = None
        self.history = None
        self.class_names = None
        
        # Determine project root and models directory
        if models_dir is None:
            # Try to find project root (look for src/ or notebook/ directories)
            current_dir = os.path.abspath(os.getcwd())
            if os.path.basename(current_dir) == 'notebook':
                # Running from notebook folder
                self.models_dir = os.path.join(os.path.dirname(current_dir), 'models')
            elif os.path.exists(os.path.join(current_dir, 'src')):
                # Running from project root
                self.models_dir = os.path.join(current_dir, 'models')
            else:
                # Fallback: use current directory
                self.models_dir = os.path.join(current_dir, 'models')
        else:
            self.models_dir = os.path.abspath(models_dir)
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
    def build_model(self):
        """
        Build a transfer learning model using pre-trained MobileNetV2 or ResNet50.
        """
        # Load pre-trained base model
        if self.base_model_name == 'MobileNetV2':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3),
                alpha=1.0  # Width multiplier, 1.0 is the default
            )
        elif self.base_model_name == 'ResNet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        else:
            raise ValueError(f"Unknown base model: {self.base_model_name}")
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Create the model
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Preprocessing - use both GlobalAveragePooling and GlobalMaxPooling for richer features
        x = base_model(inputs, training=False)
        
        # Concatenate both pooling strategies for better feature representation
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)
        x = layers.Concatenate()([avg_pool, max_pool])
        
        # Add dropout for regularization
        x = layers.Dropout(0.3)(x)
        
        # Enhanced classifier head with more capacity
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile with optimizer and learning rate (optimized for high accuracy)
        self.model.compile(
            optimizer=Adam(learning_rate=0.0005),  # Balanced learning rate for stable training
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        return self.model
    
    def train(self, train_generator, val_generator, epochs=50, fine_tune_epochs=10):
        """
        Train the model with early stopping and learning rate reduction.
        Includes fine-tuning phase.
        """
        # Get model path in project root's models folder
        model_path = os.path.join(self.models_dir, 'brain_tumor_model.h5')
        
        # Callbacks optimized for high accuracy
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',  # Monitor accuracy instead of loss
                patience=15,  # More patience for better convergence
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',  # Monitor accuracy
                factor=0.3,  # More aggressive LR reduction
                patience=5,
                min_lr=1e-8,
                mode='max',
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        # Phase 1: Train with frozen base model
        print("Phase 1: Training with frozen base model...")
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tuning (unfreeze more layers for better performance)
        print("\nPhase 2: Fine-tuning...")
        if self.base_model_name == 'MobileNetV2':
            # Unfreeze more layers of MobileNetV2 for better fine-tuning
            # MobileNetV2 has ~155 layers, unfreeze last 60 layers (more aggressive fine-tuning)
            base_model = self.model.layers[1]
            total_layers = len(base_model.layers)
            # Unfreeze last 60 layers (approximately last 40% of the model)
            trainable_count = 0
            for i, layer in enumerate(base_model.layers):
                # Unfreeze layers from index ~95 onwards (last 60 layers)
                if i >= (total_layers - 60):
                    layer.trainable = True
                    trainable_count += 1
            print(f"Unfroze {trainable_count} layers for fine-tuning (out of {total_layers} total layers)")
        elif self.base_model_name == 'ResNet50':
            # Unfreeze last 2 stages for ResNet50
            base_model = self.model.layers[1]
            for layer in base_model.layers[-30:]:
                layer.trainable = True
            print(f"Unfroze last 30 layers for ResNet50 fine-tuning")
        
        # Recompile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=0.00005),  # Slightly higher for better fine-tuning
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        # Fine-tune
        fine_tune_history = self.model.fit(
            train_generator,
            epochs=fine_tune_epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        for key in self.history.history.keys():
            self.history.history[key].extend(fine_tune_history.history[key])
        
        return self.history
    
    def evaluate(self, test_generator):
        """
        Evaluate the model and return comprehensive metrics.
        """
        # Predictions
        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Get class names
        class_names = list(test_generator.class_indices.keys())
        
        # Calculate metrics
        accuracy = accuracy_score(true_classes, predicted_classes)
        
        # Classification report
        report = classification_report(
            true_classes,
            predicted_classes,
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Per-class metrics
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_names': class_names
        }
        
        return results
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history (loss and accuracy curves).
        
        Args:
            save_path: Path to save plot (defaults to models/visualizations/training_history.png)
        """
        if save_path is None:
            viz_dir = os.path.join(self.models_dir, 'visualizations')
            save_path = os.path.join(viz_dir, 'training_history.png')
        else:
            # If relative path, make it relative to models_dir
            if not os.path.isabs(save_path):
                save_path = os.path.join(self.models_dir, save_path)
        
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training history saved to {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, cm, class_names, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            save_path: Path to save plot (defaults to models/visualizations/confusion_matrix.png)
        """
        if save_path is None:
            viz_dir = os.path.join(self.models_dir, 'visualizations')
            save_path = os.path.join(viz_dir, 'confusion_matrix.png')
        else:
            # If relative path, make it relative to models_dir
            if not os.path.isabs(save_path):
                save_path = os.path.join(self.models_dir, save_path)
        
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def save_model(self, filepath=None):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save model (defaults to models/brain_tumor_model.h5 in project root)
        """
        if filepath is None:
            filepath = os.path.join(self.models_dir, 'brain_tumor_model.h5')
        else:
            # If relative path, make it relative to models_dir
            if not os.path.isabs(filepath):
                filepath = os.path.join(self.models_dir, filepath)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """
        Load a saved model.
        
        Args:
            filepath: Path to load model from (defaults to models/brain_tumor_model.h5 in project root)
        """
        if filepath is None:
            filepath = os.path.join(self.models_dir, 'brain_tumor_model.h5')
        else:
            # If relative path, make it relative to models_dir
            if not os.path.isabs(filepath):
                filepath = os.path.join(self.models_dir, filepath)
        
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model


def train_model(data_dir='data/train', epochs=50, fine_tune_epochs=10, models_dir=None):
    """
    Main training function.
    
    Args:
        data_dir: Directory containing training data
        epochs: Number of training epochs
        fine_tune_epochs: Number of fine-tuning epochs
        models_dir: Directory to save models (defaults to project root/models)
    """
    # Determine models directory
    if models_dir is None:
        current_dir = os.path.abspath(os.getcwd())
        if os.path.basename(current_dir) == 'notebook':
            models_dir = os.path.join(os.path.dirname(current_dir), 'models')
        elif os.path.exists(os.path.join(current_dir, 'src')):
            models_dir = os.path.join(current_dir, 'models')
        else:
            models_dir = os.path.join(current_dir, 'models')
    
    models_dir = os.path.abspath(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    
    # Prepare data
    print("Preparing data...")
    train_gen, val_gen = prepare_data_for_training(data_dir)
    
    # Get class names
    class_names = list(train_gen.class_indices.keys())
    print(f"Classes: {class_names}")
    
    # Create and build model
    classifier = BrainTumorClassifier(
        img_size=(224, 224),
        num_classes=len(class_names),
        base_model_name='MobileNetV2',
        models_dir=models_dir
    )
    classifier.class_names = class_names
    classifier.build_model()
    
    # Print model summary
    classifier.model.summary()
    
    # Train model
    print("\nStarting training...")
    classifier.train(train_gen, val_gen, epochs=epochs, fine_tune_epochs=fine_tune_epochs)
    
    # Evaluate on validation set
    print("\nEvaluating model...")
    results = classifier.evaluate(val_gen)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    
    # Plot results
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(
        results['confusion_matrix'],
        results['class_names']
    )
    
    # Save model
    classifier.save_model()
    
    # Save class names
    class_names_path = os.path.join(models_dir, 'class_names.pkl')
    with open(class_names_path, 'wb') as f:
        pickle.dump(class_names, f)
    print(f"Class names saved to {class_names_path}")
    
    return classifier, results


if __name__ == "__main__":
    # Train the model
    classifier, results = train_model(epochs=30, fine_tune_epochs=5)
    print("\nTraining completed!")

