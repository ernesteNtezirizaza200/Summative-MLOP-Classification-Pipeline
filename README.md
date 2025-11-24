# Brain Tumor MRI Classification - MLOPs Pipeline

## Project Description

This project implements an end-to-end Machine Learning Operations (MLOPs) pipeline for brain tumor classification from MRI images. The system classifies brain tumors into four categories: Glioma, Meningioma, No Tumor, and Pituitary tumor.

## Features

- **Image Classification**: Deep learning model for brain tumor classification
- **Feature Extraction**: Extracts image features and saves to CSV
- **Model Training & Evaluation**: Comprehensive evaluation with multiple metrics
- **Database System**: SQLite database for tracking:
  - Uploaded images and metadata
  - Preprocessing activities and logs
  - Training sessions with complete metrics history
- **REST API**: FastAPI-based prediction endpoints
- **Streamlit UI**: Interactive web interface with:
  - Single image prediction
  - Data visualizations (3+ feature interpretations)
  - Bulk data upload for retraining (saved to database)
  - Retraining trigger functionality
  - Model uptime monitoring
  - Database statistics and training history
- **Retraining Pipeline**: Automated retraining with data upload and database tracking
- **Load Testing**: Locust-based performance testing
- **Cloud Deployment**: Dockerized application ready for Render deployment

## Dataset

The dataset is from Kaggle: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

**Classes:**
- Glioma
- Meningioma
- No Tumor
- Pituitary

## Project Structure

```
Summative-MLOP-Classification-Pipeline/
│
├── README.md
├── DATABASE_IMPLEMENTATION.md
│
├── notebook/
│   └── brain_tumor_classification.ipynb
│
├── src/
│   ├── preprocessing.py      # Feature extraction
│   ├── model.py              # Model training
│   ├── prediction.py         # Prediction functions
│   └── database.py           # Database management
│
├── data/
│   ├── train/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   ├── test/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   ├── processed/
│   │   ├── image_features_train.csv
│   │   └── image_features_test.csv
│   ├── retrain_uploads/      # Uploaded images for retraining
│   └── retraining_database.db # SQLite database
│
├── models/
│   ├── brain_tumor_model.h5
│   ├── class_names.pkl
│   └── visualizations/       # Saved visualization images
│       ├── class_distribution.png
│       ├── sample_images.png
│       ├── feature_distributions.png
│       ├── training_history.png
│       ├── learning_curves.png
│       ├── confusion_matrix_validation.png
│       ├── confusion_matrix_test.png
│       └── sample_prediction.png
│
├── app.py                    # Streamlit UI
├── api.py                    # FastAPI endpoints
├── retrain.py                # Retraining script
├── locustfile.py             # Load testing
├── Dockerfile
├── requirements.txt
└── .dockerignore
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip
- Docker (optional, for containerization)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Summative-MLOP-Classification-Pipeline
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Download from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
   - Extract to the `data/` directory maintaining the train/test structure:
     ```
     data/
     ├── train/
     │   ├── glioma/
     │   ├── meningioma/
     │   ├── notumor/
     │   └── pituitary/
     └── test/
         ├── glioma/
         ├── meningioma/
         ├── notumor/
         └── pituitary/
     ```

5. **Extract features (Optional but Recommended)**
   ```bash
   python src/preprocessing.py
   ```
   This will create `data/processed/image_features_train.csv` and `data/processed/image_features_test.csv`.

6. **Train the model**
   
   Option A: Using Jupyter Notebook (Recommended)
   ```bash
   jupyter notebook notebook/brain_tumor_classification.ipynb
   ```
   
   Option B: Using Python Script
   ```bash
   python src/model.py
   ```

### Running the Application

#### Option 1: Streamlit UI (Recommended for local development)

```bash
streamlit run app.py
```

Access the UI at `http://localhost:8501`

#### Option 2: FastAPI Server

```bash
python api.py
```

API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`
- Uptime: `http://localhost:8000/uptime`

#### Option 3: Docker

```bash
# Build the image
docker build -t brain-tumor-classifier .

# Run the container
docker run -p 8501:8501 brain-tumor-classifier
```

## Usage

### Prediction

1. **Via Streamlit UI:**
   - Navigate to the "Predict" tab
   - Upload a single MRI image
   - Click "Predict" to get the classification result

2. **Via API:**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@path/to/image.jpg"
   ```

### Retraining

1. **Via Streamlit UI:**
   - Navigate to the "Upload Data" tab
   - Upload multiple images (bulk upload) - **automatically saved to database**
   - Navigate to "Retrain Model" tab
   - View database statistics and recent training sessions
   - Click "Trigger Retraining" button
   - Monitor the retraining process (all activities logged to database)

2. **Via API:**
   ```bash
   # Upload images (saved to database)
   curl -X POST "http://localhost:8000/retrain" \
        -F "files=@image1.jpg" \
        -F "files=@image2.jpg" \
        -F "class_name=glioma"
   
   # Check training status
   curl "http://localhost:8000/retrain/status"
   
   # Get database statistics
   curl "http://localhost:8000/database/stats"
   ```

3. **Via Command Line:**
   ```bash
   python retrain.py [epochs] [fine_tune_epochs]
   ```

### Load Testing with Locust

```bash
# Terminal 1: Start API
python api.py

# Terminal 2: Run Locust
locust -f locustfile.py --host=http://localhost:8000

# Open browser: http://localhost:8089
```

## Testing the Pipeline

### Single Prediction

Test the prediction functionality directly:
```bash
python src/prediction.py data/test/glioma/image1.jpg
```

### Retraining

1. Upload images via Streamlit UI or API (automatically saved to database)
2. Trigger retraining via UI or command line:
```bash
python retrain.py [epochs] [fine_tune_epochs]
```
3. All retraining activities are logged to the database:
   - Preprocessing logs (images processed, features extracted, timing)
   - Training session metrics (accuracy, precision, recall, F1-score)
   - Image usage tracking (which images were used in training)

## Database Implementation

The project includes a comprehensive SQLite database system that tracks the complete retraining pipeline:

### Database Features

1. **Data File Uploading + Saving to Database**
   - All uploaded images are automatically saved to the database
   - Tracks: filename, class, file path, size, upload timestamp, metadata
   - Accessible via Streamlit UI and FastAPI endpoints

2. **Data Preprocessing Logging**
   - All preprocessing activities are logged to the database
   - Tracks: images processed, features extracted, processing time, status
   - Images are marked as "processed" after preprocessing

3. **Retraining Session Tracking**
   - Complete training history with metrics
   - Tracks: epochs, accuracy, precision, recall, F1-score (before/after)
   - Links uploaded images to training sessions
   - Full audit trail for compliance

### Database Schema

- **`uploaded_images`**: Stores metadata for all uploaded images
- **`training_sessions`**: Tracks each retraining session with metrics
- **`preprocessing_logs`**: Logs preprocessing activities and timing

### Database Access

The database is automatically created at `data/retraining_database.db` on first use.

**View Statistics:**
- Streamlit UI: "Retrain Model" page shows database statistics
- API: `GET /database/stats` endpoint
- Python: `from src.database import get_database`

For detailed documentation, see [DATABASE_IMPLEMENTATION.md](DATABASE_IMPLEMENTATION.md)

## Model Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions
- **Loss**: Training and validation loss curves

All metrics are automatically saved to the database for each training session.

## Deployment on Render (Docker)

### Quick Steps

1. **Push code to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Create Render account** at https://render.com and connect GitHub

3. **Create new Web Service:**
   - Click "New +" → "Web Service"
   - Select your repository
   - **IMPORTANT**: Choose **"Docker"** as build method (not Nixpacks)
   - Render will automatically detect your Dockerfile

4. **Configure:**
   - **Name**: `brain-tumor-classifier` (or your choice)
   - **Region**: Choose closest to users
   - **Branch**: `main`
   - **Instance Type**: Starter ($7/month) recommended (Free tier may not work with TensorFlow)

5. **Environment Variables** (optional):
   - `PORT`: Automatically set by Render
   - `PYTHONUNBUFFERED`: `1` (recommended)

6. **Deploy:**
   - Click "Create Web Service"
   - Wait for build (5-15 minutes)
   - Your app will be live at: `https://your-app-name.onrender.com`

### Detailed Guide

For complete step-by-step instructions, see [RENDER_DOCKER_DEPLOYMENT.md](RENDER_DOCKER_DEPLOYMENT.md)

### Important Notes

- **Dockerfile**: Must use `${PORT}` environment variable (already configured)
- **Model File**: Ensure `models/brain_tumor_model.h5` exists in repository
- **Free Tier**: 512 MB RAM may be insufficient for TensorFlow
- **Recommended**: Use Starter plan ($7/month) or higher for better performance

## Video Demo

[YouTube Link - To be added]

## Results from Flood Request Simulation

Load testing results with different numbers of Docker containers will be documented here after running Locust tests.

## Troubleshooting

### Model Not Found
- Ensure you've trained the model first
- Check that `models/brain_tumor_model.h5` exists

### Import Errors
- Activate your virtual environment
- Install all requirements: `pip install -r requirements.txt`

### GPU Issues
- The code works on CPU, but GPU is recommended for training
- Install TensorFlow GPU version if you have CUDA

## Technologies Used

- **Deep Learning**: TensorFlow/Keras
- **Database**: SQLite (for tracking and audit trail)
- **API Framework**: FastAPI
- **UI Framework**: Streamlit
- **Load Testing**: Locust
- **Containerization**: Docker
- **Cloud Platform**: Render
- **Data Visualization**: Matplotlib, Seaborn, Plotly

## File Structure Details

```
├── src/
│   ├── preprocessing.py  # Feature extraction from images
│   ├── model.py          # Model training and evaluation
│   ├── prediction.py     # Prediction functions
│   └── database.py       # Database management and tracking
├── notebook/
│   └── brain_tumor_classification.ipynb  # Complete ML pipeline
├── data/
│   ├── train/            # Training images (by class)
│   ├── test/             # Test images (by class)
│   ├── processed/        # Extracted features (CSV files)
│   ├── retrain_uploads/  # Uploaded images for retraining
│   └── retraining_database.db  # SQLite database
├── models/
│   ├── brain_tumor_model.h5        # Trained model (best checkpoint)
│   ├── class_names.pkl             # Class labels
│   └── visualizations/             # Saved plots and charts
├── app.py                # Streamlit UI application
├── api.py                # FastAPI REST endpoints
├── retrain.py            # Retraining script with database logging
├── locustfile.py         # Load testing configuration
├── Dockerfile            # Docker container configuration
├── requirements.txt      # Python dependencies
└── DATABASE_IMPLEMENTATION.md  # Database documentation
```

## Author

[Your Name]

## License

This project is for educational purposes.

