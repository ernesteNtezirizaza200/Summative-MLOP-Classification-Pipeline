"""
Streamlit UI for Brain Tumor Classification.
Includes prediction, visualizations, upload, and retraining features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image as PILImage
import cv2
import os
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
try:
    from src.prediction import BrainTumorPredictor
    from src.preprocessing import extract_features_from_directory
    from src.database import get_database
except ImportError:
    from prediction import BrainTumorPredictor
    from preprocessing import extract_features_from_directory
    from database import get_database
import subprocess
import sys

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    try:
        st.session_state.predictor = BrainTumorPredictor()
        st.session_state.model_loaded = True
    except Exception as e:
        st.session_state.model_loaded = False
        st.session_state.model_error = str(e)

if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()

if 'request_count' not in st.session_state:
    st.session_state.request_count = 0


def get_uptime():
    """Calculate uptime."""
    uptime_seconds = time.time() - st.session_state.start_time
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


# Sidebar
with st.sidebar:
    st.title("🧠 Brain Tumor Classifier")
    st.markdown("---")
    
    # Model status
    if st.session_state.model_loaded:
        st.success("✅ Model Loaded")
    else:
        st.error("❌ Model Not Loaded")
        if 'model_error' in st.session_state:
            st.error(st.session_state.model_error)
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔮 Predict", "📊 Visualizations", "📤 Upload Data", "🔄 Retrain Model", "📈 Model Uptime"]
    )


# Home Page
if page == "🏠 Home":
    st.markdown('<div class="main-header">Brain Tumor MRI Classification System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to the Brain Tumor Classification System
    
    This application uses deep learning to classify brain tumor MRI images into four categories:
    - **Glioma**
    - **Meningioma**
    - **No Tumor**
    - **Pituitary Tumor**
    
    ### Features:
    - 🎯 **Single Image Prediction**: Upload an MRI image and get instant classification
    - 📊 **Data Visualizations**: Explore dataset features and insights
    - 📤 **Bulk Data Upload**: Upload multiple images for retraining
    - 🔄 **Model Retraining**: Trigger retraining with new data
    - 📈 **Uptime Monitoring**: Track model performance and availability
    
    ### How to Use:
    1. Navigate to **Predict** to classify a single image
    2. Check **Visualizations** to explore dataset features
    3. Use **Upload Data** to add new training data
    4. Trigger **Retrain Model** to update the model with new data
    5. Monitor **Model Uptime** for system health
    """)


# Prediction Page
elif page == "🔮 Predict":
    st.title("Image Prediction")
    st.markdown("Upload a brain tumor MRI image to get a classification prediction.")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a brain tumor MRI image"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = PILImage.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            if st.button("🔮 Predict", type="primary"):
                if not st.session_state.model_loaded:
                    st.error("Model not loaded. Please check model files.")
                else:
                    with st.spinner("Predicting..."):
                        try:
                            # Convert PIL to numpy array
                            img_array = np.array(image)
                            if len(img_array.shape) == 2:
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                            elif img_array.shape[2] == 4:
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                            
                            # Make prediction
                            result = st.session_state.predictor.predict(
                                img_array,
                                return_probabilities=True
                            )
                            
                            st.session_state.request_count += 1
                            
                            # Display results
                            st.subheader("Prediction Results")
                            
                            # Predicted class
                            st.metric(
                                "Predicted Class",
                                result['predicted_class'].upper(),
                                delta=f"{result['confidence']*100:.2f}% confidence"
                            )
                            
                            # Probabilities
                            st.subheader("Class Probabilities")
                            prob_df = pd.DataFrame({
                                'Class': list(result['probabilities'].keys()),
                                'Probability': [v*100 for v in result['probabilities'].values()]
                            })
                            prob_df = prob_df.sort_values('Probability', ascending=False)
                            
                            # Bar chart
                            fig = px.bar(
                                prob_df,
                                x='Class',
                                y='Probability',
                                color='Probability',
                                color_continuous_scale='Blues',
                                title="Prediction Probabilities"
                            )
                            fig.update_layout(yaxis_title="Probability (%)")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Table
                            st.dataframe(prob_df, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"Prediction error: {str(e)}")


# Visualizations Page
elif page == "📊 Visualizations":
    st.title("Data Visualizations")
    st.markdown("Explore dataset features and insights.")
    
    # Check if feature CSV exists
    feature_files = ['data/processed/image_features_train.csv', 'data/processed/image_features.csv', 'image_features_train.csv', 'image_features.csv']
    feature_file = None
    
    for f in feature_files:
        if os.path.exists(f):
            feature_file = f
            break
    
    if feature_file:
        df = pd.read_csv(feature_file)
        
        st.success(f"Loaded features from {feature_file}")
        st.info(f"Total samples: {len(df)}")
        
        # Feature selection
        st.subheader("Feature Analysis")
        
        # Feature 1: Mean Intensity by Class
        st.markdown("### Feature 1: Mean Intensity Distribution by Tumor Class")
        st.markdown("""
        **Interpretation**: This visualization shows the average pixel intensity across different tumor types.
        - **Glioma** and **Meningioma** typically show different intensity patterns due to their tissue composition
        - **No Tumor** images often have more uniform intensity distributions
        - **Pituitary** tumors may show distinct intensity signatures
        """)
        
        if 'mean_intensity' in df.columns and 'class' in df.columns:
            fig1 = px.box(
                df,
                x='class',
                y='mean_intensity',
                color='class',
                title="Mean Intensity by Tumor Class",
                labels={'mean_intensity': 'Mean Intensity', 'class': 'Tumor Class'}
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        # Feature 2: Standard Deviation by Class
        st.markdown("### Feature 2: Intensity Variability (Standard Deviation) by Class")
        st.markdown("""
        **Interpretation**: Standard deviation measures texture variability in the images.
        - Higher values indicate more texture variation (common in tumor regions)
        - Lower values suggest more uniform regions (common in healthy tissue)
        - Different tumor types exhibit distinct texture patterns
        """)
        
        if 'std_intensity' in df.columns and 'class' in df.columns:
            fig2 = px.violin(
                df,
                x='class',
                y='std_intensity',
                color='class',
                title="Intensity Standard Deviation by Tumor Class",
                labels={'std_intensity': 'Standard Deviation', 'class': 'Tumor Class'}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Feature 3: Gradient Mean by Class
        st.markdown("### Feature 3: Edge Strength (Gradient Mean) by Class")
        st.markdown("""
        **Interpretation**: Gradient magnitude indicates edge strength and boundaries in images.
        - Tumor boundaries often create strong edges
        - Different tumor types have varying edge characteristics
        - This feature helps distinguish between well-defined and diffuse tumors
        """)
        
        if 'gradient_mean' in df.columns and 'class' in df.columns:
            fig3 = px.scatter(
                df,
                x='mean_intensity',
                y='gradient_mean',
                color='class',
                title="Mean Intensity vs Gradient Mean (Edge Strength)",
                labels={
                    'mean_intensity': 'Mean Intensity',
                    'gradient_mean': 'Gradient Mean (Edge Strength)',
                    'class': 'Tumor Class'
                }
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        # Additional statistics
        st.subheader("Dataset Statistics")
        if 'class' in df.columns:
            class_counts = df['class'].value_counts()
            fig4 = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title="Class Distribution"
            )
            st.plotly_chart(fig4, use_container_width=True)
            
            st.dataframe(class_counts.reset_index().rename(columns={'index': 'Class', 'class': 'Count'}))
    
    else:
        st.warning("Feature CSV file not found. Please run feature extraction first.")
        if st.button("Extract Features"):
            with st.spinner("Extracting features from training data..."):
                try:
                    extract_features_from_directory('data/train', 'data/processed/image_features_train.csv')
                    st.success("Features extracted successfully! Please refresh the page.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error extracting features: {str(e)}")


# Upload Data Page
elif page == "📤 Upload Data":
    st.title("Upload Data for Retraining")
    st.markdown("Upload multiple brain tumor MRI images to be used for model retraining.")
    
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload multiple images for retraining"
    )
    
    if uploaded_files:
        st.info(f"Selected {len(uploaded_files)} file(s)")
        
        # Display uploaded files
        if st.checkbox("Show uploaded files"):
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size} bytes)")
        
        # Save location selection
        save_location = st.selectbox(
            "Save to class folder:",
            ["glioma", "meningioma", "notumor", "pituitary", "retrain_uploads"]
        )
        
        if st.button("💾 Save Files", type="primary"):
            save_dir = f"data/retrain_uploads/{save_location}"
            os.makedirs(save_dir, exist_ok=True)
            
            # Initialize database
            db = get_database()
            
            saved_count = 0
            saved_to_db = 0
            with st.spinner("Saving files to disk and database..."):
                for file in uploaded_files:
                    try:
                        file_path = os.path.join(save_dir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        # Save to database
                        image_id = db.save_uploaded_image(
                            filename=file.name,
                            class_name=save_location,
                            file_path=file_path,
                            file_size=file.size,
                            metadata={
                                'upload_source': 'streamlit_ui',
                                'upload_timestamp': datetime.now().isoformat()
                            }
                        )
                        saved_to_db += 1
                        saved_count += 1
                    except Exception as e:
                        st.error(f"Error saving {file.name}: {str(e)}")
            
            st.success(f"Successfully saved {saved_count} file(s) to {save_dir}")
            st.success(f"✅ {saved_to_db} file(s) saved to database")
            st.info("Files are ready for retraining. Go to 'Retrain Model' to trigger retraining.")
            
            # Show database statistics
            stats = db.get_training_statistics()
            st.info(f"📊 Database: {stats['total_uploaded_images']} total images uploaded, "
                   f"{stats['processed_images']} processed")


# Retrain Model Page
elif page == "🔄 Retrain Model":
    st.title("Retrain Model")
    st.markdown("Trigger model retraining with newly uploaded data.")
    
    # Initialize database
    db = get_database()
    
    # Show database statistics
    stats = db.get_training_statistics()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Uploaded", stats['total_uploaded_images'])
    with col2:
        st.metric("Processed", stats['processed_images'])
    with col3:
        st.metric("Training Sessions", stats['total_training_sessions'])
    with col4:
        st.metric("Completed", stats['completed_sessions'])
    
    # Show images by class
    if stats['images_by_class']:
        st.subheader("Uploaded Images by Class")
        class_df = pd.DataFrame(list(stats['images_by_class'].items()), 
                               columns=['Class', 'Count'])
        st.bar_chart(class_df.set_index('Class'))
    
    # Check for uploaded data
    retrain_dir = "data/retrain_uploads"
    has_data = False
    
    # Check database for unprocessed images
    unprocessed_images = db.get_uploaded_images(processed=False)
    
    if os.path.exists(retrain_dir):
        total_files = 0
        for root, dirs, files in os.walk(retrain_dir):
            total_files += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if total_files > 0 or len(unprocessed_images) > 0:
            has_data = True
            st.info(f"Found {total_files} image(s) in directory, {len(unprocessed_images)} unprocessed in database")
        else:
            st.warning("No images found for retraining. Please upload data first.")
    else:
        st.warning("Retraining directory not found. Please upload data first.")
    
    # Show recent training sessions
    st.subheader("Recent Training Sessions")
    recent_sessions = db.get_training_sessions(limit=5)
    if recent_sessions:
        sessions_df = pd.DataFrame(recent_sessions)
        # Format columns for display
        display_cols = ['id', 'session_timestamp', 'status', 'epochs', 
                       'final_accuracy', 'images_used']
        available_cols = [col for col in display_cols if col in sessions_df.columns]
        st.dataframe(sessions_df[available_cols], use_container_width=True)
    else:
        st.info("No training sessions yet.")
    
    if has_data:
        st.subheader("Retraining Configuration")
        
        epochs = st.slider("Training Epochs", min_value=5, max_value=50, value=10)
        fine_tune_epochs = st.slider("Fine-tuning Epochs", min_value=0, max_value=10, value=3)
        
        if st.button("🔄 Trigger Retraining", type="primary"):
            with st.spinner("Retraining model... This may take a while."):
                try:
                    # Import retraining function
                    from retrain import retrain_model
                    
                    # Run retraining
                    result = retrain_model(
                        retrain_data_dir=retrain_dir,
                        epochs=epochs,
                        fine_tune_epochs=fine_tune_epochs
                    )
                    
                    if result:
                        st.success("✅ Model retraining completed successfully!")
                        st.info("The model has been updated. New predictions will use the retrained model.")
                        
                        # Reload predictor
                        try:
                            st.session_state.predictor = BrainTumorPredictor()
                            st.session_state.model_loaded = True
                            st.success("✅ New model loaded successfully!")
                        except Exception as e:
                            st.warning(f"Model retrained but could not reload: {str(e)}")
                    else:
                        st.error("Retraining failed. Check logs for details.")
                
                except Exception as e:
                    st.error(f"Retraining error: {str(e)}")
                    st.exception(e)


# Model Uptime Page
elif page == "📈 Model Uptime":
    st.title("Model Uptime & Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        uptime = get_uptime()
        st.metric("Uptime", uptime)
    
    with col2:
        st.metric("Total Requests", st.session_state.request_count)
    
    with col3:
        status = "🟢 Online" if st.session_state.model_loaded else "🔴 Offline"
        st.metric("Model Status", status)
    
    st.markdown("---")
    
    # System information
    st.subheader("System Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("### Model Information")
        if st.session_state.model_loaded:
            st.success("✅ Model is loaded and ready")
            st.info(f"Model Path: models/brain_tumor_model.h5")
            st.info(f"Classes: {', '.join(st.session_state.predictor.class_names)}")
        else:
            st.error("❌ Model is not loaded")
    
    with info_col2:
        st.markdown("### Performance Metrics")
        st.info("Request handling: Active")
        st.info(f"Session started: {datetime.fromtimestamp(st.session_state.start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Uptime chart (simulated)
    st.subheader("Uptime History")
    st.info("Uptime tracking is active. The model has been running since session start.")

