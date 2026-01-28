# Myopia Detection System

A deep learning-based system for early detection of myopia using both tabular patient data and retinal images. The system employs multiple CNN architectures to provide accurate predictions with confidence scores.

## Overview

This project implements three distinct models for myopia detection:

- **Basic CNN**: Processes tabular patient data for quick screening
- **Hybrid CNN**: Enhanced model with regularization for improved accuracy on structured data
- **Image CNN**: Analyzes retinal images for visual diagnosis

The system includes a FastAPI backend for real-time predictions and a web interface for easy interaction.

## Features

- Multiple model architectures for comprehensive analysis
- Real-time image prediction via REST API
- Automatic optimal threshold calculation using Youden's J statistic
- Data augmentation for improved model generalization
- Comprehensive metrics and visualization outputs
- Interactive web interface for testing

## Requirements

```
tensorflow>=2.10.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
fastapi>=0.95.0
uvicorn>=0.21.0
pillow>=9.4.0
python-multipart>=0.0.6
```

## Installation

Clone the repository:

```bash
git clone https://github.com/chibuezedev/miovision_model.git
cd miovision_model
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── DATASET2/
│   ├── myopia.CSV           # Tabular patient data
│   └── IMAGEDATA/
│       ├── train/
│       ├── test/
│       └── val/
├── SAVED_MODELS/            # Trained model files
├── OUTPUTS/                 # Generated plots and metrics
├── train_model.py           # Model training script
└── api.py                   # FastAPI server
```

## Usage

### Training Models

Run the training script to train all three models:

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train Basic CNN, Hybrid CNN, and Image CNN models
- Generate performance visualizations
- Save trained models to `SAVED_MODELS/`
- Output metrics to `OUTPUTS/`

### Running the API Server

Start the FastAPI server:

```bash
python api.py
```

The server will be available at:
- Main interface: http://localhost:8000
- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Making Predictions

**Via Web Interface:**

Navigate to http://localhost:8000 and upload an image for analysis.

**Via API:**

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("eye_image.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}%")
```

**Using base64:**

```python
import requests
import base64

with open("eye_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

url = "http://localhost:8000/predict/base64"
response = requests.post(url, json={"image": image_data})
result = response.json()
```

## Model Performance

All models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Specificity
- Sensitivity
- AUC-ROC

Results are saved in `OUTPUTS/model_metrics.csv` and visualized in comparison charts.

## Configuration

Key parameters can be modified in the `Config` class:

```python
class Config:
    EPOCHS_CNN_BASIC = 10
    EPOCHS_HYBRID = 40
    EPOCHS_IMAGE = 50
    BATCH_SIZE = 16
    IMG_SIZE = (224, 224)
    LEARNING_RATE = 0.00009
    TEST_SIZE = 0.2
    PREDICTION_THRESHOLD = 0.45  # Automatically optimized during training
```

## API Endpoints

### GET /

Returns the web interface for testing.

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### POST /predict

Upload an image file for prediction.

**Parameters:**
- `file`: Image file (JPG, PNG, JPEG)

**Response:**
```json
{
  "prediction": "MYOPIA",
  "confidence": 87.45,
  "probability_myopia": 87.45,
  "probability_normal": 12.55,
  "message": "Myopia detected with 87.5% confidence. Please consult an eye care professional."
}
```

### POST /predict/base64

Send base64 encoded image for prediction.

**Request Body:**
```json
{
  "image": "base64_encoded_string"
}
```

### GET /model/info

Get information about the loaded model.

**Response:**
```json
{
  "model_name": "Image CNN",
  "input_shape": [224, 224, 3],
  "threshold": 0.45,
  "classes": ["NORMAL", "MYOPIA"],
  "model_path": "SAVED_MODELS/image_cnn_best.h5"
}
```

## Dataset Format

### Tabular Data

CSV file with patient measurements and binary MYOPIC label (0=NORMAL, 1=MYOPIA).

### Image Data

Directory structure:
```
IMAGEDATA/
├── train/
│   ├── NORMAL/
│   └── MYOPIA/
├── test/
│   ├── NORMAL/
│   └── MYOPIA/
└── val/
    ├── NORMAL/
    └── MYOPIA/
```

Images should be in JPG format. The system automatically resizes to 224x224 pixels.

## Model Architecture

### Image CNN

- 4 Convolutional layers with MaxPooling
- Dropout for regularization
- 3 Dense layers
- Binary classification with sigmoid activation

### Hybrid CNN

- Conv1D layers for sequential data processing
- L1-L2 regularization
- Batch normalization
- Multiple dense layers with dropout

### Basic CNN

- Simplified Conv1D architecture
- Batch normalization
- Dropout layers
- Efficient for quick predictions

## Training Features

- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Model checkpointing for best weights
- Comprehensive data augmentation for images
- Stratified train-test split

## Output Files

Training generates the following outputs:

**Models:**
- `cnn_basic_best.h5` / `cnn_basic_final.h5`
- `hybrid_cnn_best.h5` / `hybrid_cnn_final.h5`
- `image_cnn_best.h5` / `image_cnn_final.h5`

**Visualizations:**
- Training history plots
- Confusion matrices
- Model comparison charts
- Target distribution plots

**Metrics:**
- `model_metrics.csv` - Comprehensive performance metrics

## License

MIT License

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## Contact

For questions or support, please open an issue on GitHub.