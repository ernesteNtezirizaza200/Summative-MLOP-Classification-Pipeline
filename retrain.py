"""
Retraining script for brain tumor classification model.
Handles data upload, preprocessing, and model retraining.
"""

import os
import shutil
import numpy as np
from datetime import datetime
import time
import tensorflow as tf
try:
    from src.model import BrainTumorClassifier
    from src.preprocessing import extract_features_from_directory, prepare_data_for_training
    from src.database import get_database
except ImportError:
    import sys
    sys.path.insert(0, 'src')
    from model import BrainTumorClassifier
    from preprocessing import extract_features_from_directory, prepare_data_for_training
    from database import get_database
import pickle


def prepare_retrain_data(retrain_data_dir, main_data_dir='data/train'):
    """
    Prepare retraining data by organizing uploaded files into class folders.
    Moves files from retrain_uploads to appropriate class folders in training data.
    """
    print(f"Preparing retraining data from {retrain_data_dir}...")
    
    if not os.path.exists(retrain_data_dir):
        print(f"Retraining directory not found: {retrain_data_dir}")
        return False
    
    # Get class subdirectories
    class_dirs = [d for d in os.listdir(retrain_data_dir) 
                 if os.path.isdir(os.path.join(retrain_data_dir, d))]
    
    if not class_dirs:
        print("No class directories found in retrain_uploads")
        return False
    
    moved_count = 0
    
    # Move files to main training directory
    for class_name in class_dirs:
        source_dir = os.path.join(retrain_data_dir, class_name)
        target_dir = os.path.join(main_data_dir, class_name)
        
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Get image files
        image_files = [f for f in os.listdir(source_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Move files
        for img_file in image_files:
            source_path = os.path.join(source_dir, img_file)
            target_path = os.path.join(target_dir, f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{img_file}")
            
            try:
                shutil.move(source_path, target_path)
                moved_count += 1
            except Exception as e:
                print(f"Error moving {source_path}: {str(e)}")
    
    print(f"Moved {moved_count} files to training directory")
    return moved_count > 0


def retrain_model(retrain_data_dir='data/retrain_uploads', 
                  main_data_dir='data/train',
                  epochs=10,
                  fine_tune_epochs=3,
                  model_save_path='models/brain_tumor_model.h5'):
    """
    Main retraining function.
    
    Steps:
    1. Create training session in database
    2. Prepare retraining data (move uploaded files to training directory)
    3. Extract features from new data (with database logging)
    4. Prepare data generators
    5. Load existing model or create new one
    6. Retrain the model
    7. Evaluate and save
    8. Update database with results
    """
    # Initialize database
    db = get_database()
    
    # Create training session in database
    training_session_id = db.create_training_session(
        epochs=epochs,
        fine_tune_epochs=fine_tune_epochs,
        model_path=model_save_path,
        notes=f"Retraining with data from {retrain_data_dir}"
    )
    
    print("=" * 50)
    print("Starting Model Retraining Process")
    print(f"Training Session ID: {training_session_id}")
    print("=" * 50)
    
    # Update status to in_progress
    db.update_training_session(training_session_id, status='in_progress')
    
    # Step 1: Prepare retraining data
    print("\nStep 1: Preparing retraining data...")
    if not prepare_retrain_data(retrain_data_dir, main_data_dir):
        print("No new data to retrain with. Exiting.")
        db.update_training_session(training_session_id, status='failed', 
                                   notes="No new data to retrain with")
        return False
    
    # Get uploaded images from database
    uploaded_images = db.get_uploaded_images(processed=False)
    image_ids = [img['id'] for img in uploaded_images]
    
    # Step 2: Extract features from updated training data
    print("\nStep 2: Extracting features from training data...")
    preprocessing_start = time.time()
    images_processed = 0
    features_extracted = 0
    
    try:
        # Count images before processing
        for root, dirs, files in os.walk(main_data_dir):
            images_processed += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Extract features
        features_df = extract_features_from_directory(main_data_dir, 'data/processed/image_features_train.csv')
        features_extracted = len(features_df) if features_df is not None else 0
        
        preprocessing_time = time.time() - preprocessing_start
        
        # Log preprocessing to database
        db.log_preprocessing(
            training_session_id=training_session_id,
            images_processed=images_processed,
            features_extracted=features_extracted,
            processing_time=preprocessing_time,
            status='completed'
        )
        
        # Mark images as processed
        if image_ids:
            db.mark_images_processed(image_ids, training_session_id)
        
        print(f"Feature extraction completed: {images_processed} images, {features_extracted} features")
    except Exception as e:
        preprocessing_time = time.time() - preprocessing_start
        error_msg = str(e)
        print(f"Warning: Feature extraction failed: {error_msg}")
        print("Continuing with training...")
        
        # Log preprocessing failure
        db.log_preprocessing(
            training_session_id=training_session_id,
            images_processed=images_processed,
            features_extracted=0,
            processing_time=preprocessing_time,
            status='failed',
            error_message=error_msg
        )
    
    # Step 3: Prepare data generators
    print("\nStep 3: Preparing data generators...")
    try:
        train_gen, val_gen = prepare_data_for_training(main_data_dir)
        class_names = list(train_gen.class_indices.keys())
        num_classes = len(class_names)
        print(f"Found {num_classes} classes: {class_names}")
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        return False
    
    # Step 4: Load or create model
    print("\nStep 4: Loading/Creating model...")
    # Determine models directory (project root)
    current_dir = os.path.abspath(os.getcwd())
    if os.path.basename(current_dir) == 'notebook':
        models_dir = os.path.join(os.path.dirname(current_dir), 'models')
    elif os.path.exists(os.path.join(current_dir, 'src')):
        models_dir = os.path.join(current_dir, 'models')
    else:
        models_dir = os.path.join(current_dir, 'models')
    models_dir = os.path.abspath(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    
    classifier = BrainTumorClassifier(
        img_size=(224, 224),
        num_classes=num_classes,
        base_model_name='MobileNetV2',
        models_dir=models_dir
    )
    classifier.class_names = class_names
    
    # Try to load existing model, otherwise build new one
    # Normalize model path - if it's relative and starts with 'models/', use just the filename
    if model_save_path.startswith('models/'):
        model_filename = os.path.basename(model_save_path)
        model_full_path = os.path.join(models_dir, model_filename)
    else:
        model_filename = model_save_path
        model_full_path = os.path.join(models_dir, model_filename) if not os.path.isabs(model_save_path) else model_save_path
    
    if os.path.exists(model_full_path):
        try:
            print(f"Loading existing model from {model_full_path}...")
            # Load model to check number of classes
            temp_model = tf.keras.models.load_model(model_full_path)
            
            # Check if number of classes matches
            old_num_classes = temp_model.output_shape[-1]
            if old_num_classes != num_classes:
                print(f"Warning: Existing model has {old_num_classes} classes, but data has {num_classes} classes.")
                print("Rebuilding model with correct number of classes...")
                classifier.build_model()
            else:
                # Classes match, use existing model
                classifier.model = temp_model
                # Unfreeze some layers for fine-tuning
                for layer in classifier.model.layers[1].layers[-4:]:
                    layer.trainable = True
                classifier.model.compile(
                    optimizer=classifier.model.optimizer,
                    loss=classifier.model.loss,
                    metrics=classifier.model.metrics
                )
                print("Existing model loaded and prepared for fine-tuning.")
        except Exception as e:
            print(f"Could not load existing model: {str(e)}")
            print("Building new model...")
            classifier.build_model()
    else:
        print("No existing model found. Building new model...")
        classifier.build_model()
    
    # Step 5: Retrain the model
    print(f"\nStep 5: Retraining model (epochs={epochs}, fine_tune_epochs={fine_tune_epochs})...")
    try:
        classifier.train(
            train_gen,
            val_gen,
            epochs=epochs,
            fine_tune_epochs=fine_tune_epochs
        )
        print("Model retraining completed.")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False
    
    # Step 6: Evaluate and save
    print("\nStep 6: Evaluating model...")
    final_metrics = None
    try:
        results = classifier.evaluate(val_gen)
        final_metrics = {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score']
        }
        
        print(f"\n=== Retraining Evaluation Results ===")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        
        # Plot results
        classifier.plot_training_history('models/visualizations/training_history_retrain.png')
        classifier.plot_confusion_matrix(
            results['confusion_matrix'],
            results['class_names'],
            'models/visualizations/confusion_matrix_retrain.png'
        )
    except Exception as e:
        print(f"Warning: Evaluation failed: {str(e)}")
    
    # Save model
    print(f"\nStep 7: Saving model to {model_full_path}...")
    try:
        classifier.save_model(model_filename)
        
        # Save class names
        class_names_path = os.path.join(models_dir, 'class_names.pkl')
        with open(class_names_path, 'wb') as f:
            pickle.dump(class_names, f)
        print(f"Class names saved to {class_names_path}")
        
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        db.update_training_session(training_session_id, status='failed', 
                                   notes=f"Error saving model: {str(e)}")
        return False
    
    # Step 8: Update database with final results
    print("\nStep 8: Updating database with training results...")
    db.update_training_session(
        session_id=training_session_id,
        status='completed',
        final_metrics=final_metrics,
        images_used=len(image_ids) if image_ids else images_processed
    )
    print("Database updated successfully.")
    
    # Clean up retrain_uploads directory (optional)
    print("\nCleaning up retraining uploads directory...")
    try:
        for class_dir in os.listdir(retrain_data_dir):
            class_path = os.path.join(retrain_data_dir, class_dir)
            if os.path.isdir(class_path):
                shutil.rmtree(class_path)
        print("Cleanup completed.")
    except Exception as e:
        print(f"Warning: Cleanup failed: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Model Retraining Process Completed Successfully!")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    import sys
    
    # Default parameters
    epochs = 10
    fine_tune_epochs = 3
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
    if len(sys.argv) > 2:
        fine_tune_epochs = int(sys.argv[2])
    
    # Run retraining
    success = retrain_model(
        retrain_data_dir='data/retrain_uploads',
        epochs=epochs,
        fine_tune_epochs=fine_tune_epochs
    )
    
    if success:
        print("\n✅ Retraining completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Retraining failed!")
        sys.exit(1)

