import os
import io
import numpy as np
from PIL import Image
from typing import Optional
import tensorflow as tf
from tensorflow import keras  # type: ignore
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import base64


class Config:
    BASE_DIR = os.getcwd()
    # MODEL_DIR = os.path.join(BASE_DIR, "SAVED_MODELS")
    MODEL_PATH = os.path.join(BASE_DIR, "image_cnn_best.h5")
    IMG_SIZE = (224, 224)
    PREDICTION_THRESHOLD = 0.45


app = FastAPI(
    title="Myopia Detection API",
    description="AI-powered early detection of myopia using deep learning",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


def load_model():
    """Load the trained model"""
    global model
    try:
        if not os.path.exists(Config.MODEL_PATH):
            print(f"❌ Model not found at: {Config.MODEL_PATH}")
            print("Please train the model first by running the training script.")
            return False

        model = keras.models.load_model(Config.MODEL_PATH)
        print(f"✓ Model loaded successfully from: {Config.MODEL_PATH}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    success = load_model()
    if not success:
        print("⚠️  Server started but model is not loaded. Train the model first.")


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probability_myopia: float
    probability_normal: float
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model prediction"""
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize
    image = image.resize(Config.IMG_SIZE)

    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def make_prediction(img_array: np.ndarray) -> dict:
    """Make prediction using the loaded model"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is trained and available.",
        )

    # Predict
    prob_myopia = float(model.predict(img_array, verbose=0)[0][0])
    prob_normal = 1 - prob_myopia

    # Determine prediction
    prediction = "MYOPIA" if prob_myopia >= Config.PREDICTION_THRESHOLD else "NORMAL"
    confidence = prob_myopia if prediction == "MYOPIA" else prob_normal

    # Create message
    if prediction == "MYOPIA":
        message = f"⚠️ Myopia detected with {confidence * 100:.1f}% confidence. Please consult an eye care professional."
    else:
        message = f"✓ No myopia detected. Eyes appear normal with {confidence * 100:.1f}% confidence."

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "probability_myopia": round(prob_myopia * 100, 2),
        "probability_normal": round(prob_normal * 100, 2),
        "message": message,
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the test HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Myopia Detection - AI Diagnosis</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 600px;
                width: 100%;
                padding: 40px;
                animation: fadeIn 0.5s ease-in;
            }
            
            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: translateY(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            h1 {
                color: #667eea;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2em;
            }
            
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
                font-size: 0.95em;
            }
            
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-bottom: 20px;
                background: #f8f9ff;
            }
            
            .upload-area:hover {
                background: #eef1ff;
                border-color: #764ba2;
            }
            
            .upload-area.dragover {
                background: #e3e7ff;
                border-color: #764ba2;
                transform: scale(1.02);
            }
            
            .upload-icon {
                font-size: 3em;
                margin-bottom: 10px;
            }
            
            input[type="file"] {
                display: none;
            }
            
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 50px;
                font-size: 1.1em;
                cursor: pointer;
                width: 100%;
                transition: transform 0.2s ease;
                font-weight: 600;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }
            
            button:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
            }
            
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            
            .preview-container {
                margin: 20px 0;
                text-align: center;
                display: none;
            }
            
            .preview-container.show {
                display: block;
                animation: fadeIn 0.5s ease-in;
            }
            
            .preview-container img {
                max-width: 100%;
                max-height: 300px;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            
            .result {
                margin-top: 20px;
                padding: 20px;
                border-radius: 10px;
                display: none;
                animation: fadeIn 0.5s ease-in;
            }
            
            .result.show {
                display: block;
            }
            
            .result.myopia {
                background: #fff3cd;
                border-left: 5px solid #ffc107;
            }
            
            .result.normal {
                background: #d4edda;
                border-left: 5px solid #28a745;
            }
            
            .result h2 {
                margin-bottom: 15px;
                font-size: 1.5em;
            }
            
            .result.myopia h2 {
                color: #856404;
            }
            
            .result.normal h2 {
                color: #155724;
            }
            
            .metrics {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-top: 15px;
            }
            
            .metric {
                background: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            .metric-label {
                font-size: 0.85em;
                color: #666;
                margin-bottom: 5px;
            }
            
            .metric-value {
                font-size: 1.8em;
                font-weight: bold;
                color: #667eea;
            }
            
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            
            .loading.show {
                display: block;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .error {
                background: #f8d7da;
                color: #721c24;
                padding: 15px;
                border-radius: 10px;
                margin-top: 20px;
                display: none;
                border-left: 5px solid #dc3545;
            }
            
            .error.show {
                display: block;
                animation: fadeIn 0.5s ease-in;
            }
            
            .info-badge {
                display: inline-block;
                background: #667eea;
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.85em;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>👁️ Myopia Detection</h1>
            <p class="subtitle">AI-Powered Early Detection System</p>
            
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📤</div>
                <p style="font-size: 1.1em; color: #667eea; font-weight: 600;">Click to upload or drag & drop</p>
                <p style="color: #999; font-size: 0.9em; margin-top: 10px;">Supported: JPG, PNG, JPEG</p>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            
            <div class="preview-container" id="previewContainer">
                <img id="preview" src="" alt="Preview">
                <div class="info-badge">Image loaded successfully</div>
            </div>
            
            <button id="analyzeBtn" disabled>🔍 Analyze Image</button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 15px; color: #667eea;">Analyzing image...</p>
            </div>
            
            <div class="error" id="error"></div>
            
            <div class="result" id="result">
                <h2 id="resultTitle"></h2>
                <p id="resultMessage"></p>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value" id="confidence">-</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Myopia Probability</div>
                        <div class="metric-value" id="myopiaProb">-</div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const previewContainer = document.getElementById('previewContainer');
            const preview = document.getElementById('preview');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const error = document.getElementById('error');
            
            let selectedFile = null;
            
            // Click to upload
            uploadArea.addEventListener('click', () => fileInput.click());
            
            // Drag and drop
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            });
            
            // File input change
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFile(e.target.files[0]);
                }
            });
            
            function handleFile(file) {
                if (!file.type.startsWith('image/')) {
                    showError('Please select an image file (JPG, PNG, JPEG)');
                    return;
                }
                
                selectedFile = file;
                
                // Show preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    preview.src = e.target.result;
                    previewContainer.classList.add('show');
                    analyzeBtn.disabled = false;
                    result.classList.remove('show');
                    error.classList.remove('show');
                };
                reader.readAsDataURL(file);
            }
            
            // Analyze button
            analyzeBtn.addEventListener('click', async () => {
                if (!selectedFile) return;
                
                analyzeBtn.disabled = true;
                loading.classList.add('show');
                result.classList.remove('show');
                error.classList.remove('show');
                
                try {
                    const formData = new FormData();
                    formData.append('file', selectedFile);
                    
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || 'Prediction failed');
                    }
                    
                    const data = await response.json();
                    displayResult(data);
                    
                } catch (err) {
                    showError(err.message);
                } finally {
                    loading.classList.remove('show');
                    analyzeBtn.disabled = false;
                }
            });
            
            function displayResult(data) {
                const isMyopia = data.prediction === 'MYOPIA';
                
                result.className = 'result show ' + (isMyopia ? 'myopia' : 'normal');
                document.getElementById('resultTitle').textContent = 
                    isMyopia ? '⚠️ Myopia Detected' : '✓ Normal Vision';
                document.getElementById('resultMessage').textContent = data.message;
                document.getElementById('confidence').textContent = data.confidence.toFixed(1) + '%';
                document.getElementById('myopiaProb').textContent = data.probability_myopia.toFixed(1) + '%';
            }
            
            function showError(message) {
                error.textContent = '❌ ' + message;
                error.classList.add('show');
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None, "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict myopia from uploaded image

    Args:
        file: Image file (JPG, PNG, JPEG)

    Returns:
        Prediction results with confidence scores
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="File must be an image (JPG, PNG, JPEG)"
        )

    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess
        img_array = preprocess_image(image)

        # Make prediction
        prediction_result = make_prediction(img_array)

        return PredictionResponse(**prediction_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_base64(image_data: dict):
    """
    Predict myopia from base64 encoded image

    Args:
        image_data: Dict with 'image' key containing base64 string

    Returns:
        Prediction results with confidence scores
    """
    try:
        # Decode base64
        if "image" not in image_data:
            raise HTTPException(status_code=400, detail="Missing 'image' key")

        base64_str = image_data["image"]
        # Remove data URL prefix if present
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocess
        img_array = preprocess_image(image)

        # Make prediction
        prediction_result = make_prediction(img_array)

        return PredictionResponse(**prediction_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": "Image CNN",
        "input_shape": (224, 224, 3),
        "threshold": Config.PREDICTION_THRESHOLD,
        "classes": ["NORMAL", "MYOPIA"],
        "model_path": Config.MODEL_PATH,
    }


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 80)
    print("Starting Myopia Detection API Server")
    print("=" * 80)
    print(f"Model path: {Config.MODEL_PATH}")
    print("Server will run on: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("=" * 80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
