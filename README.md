# AuraSync Lightweight Backend API

This is a lightweight version of the AuraSync backend API designed to be deployed to Render without requiring large ML model files. It provides all the same endpoints as the full version but uses fallback implementations when models aren't available.

## Features

- 🚀 **Minimal Dependencies**: Reduced package requirements for faster deployment
- 🔄 **Progressive Enhancement**: Works without ML models, but enhances when they're available
- 💡 **Smart Fallbacks**: Provides realistic fallback responses even without models
- 📦 **Small Size**: Under 10MB total without models
- ⚙️ **Compatible API**: Same endpoints as the full version

## Deployment to Render

1. Create a new Web Service on Render
2. Connect to your GitHub repository
3. Use the following settings:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -k uvicorn.workers.UvicornWorker startup:app --log-level info`

## Environment Variables

Set the following environment variables in the Render dashboard:
- `FASTAPI_ENV`: `production`
- `CORS_ORIGINS`: `["https://auraasync.in", "https://www.auraasync.in"]`
- `SECRET_KEY`: (generate a secure random string)

## API Endpoints

- `/analyze/body`: Analyze body type from an uploaded image
- `/analyze/face`: Analyze face shape from an uploaded image
- `/analyze/skin`: Analyze skin tone from an uploaded image
- `/recommend`: Get fashion recommendations based on analysis results
- `/products/recommendations`: Get personalized product recommendations
- `/health`: Health check endpoint

## Optional: Adding ML Models

For enhanced accuracy, you can add the ML model files:

1. Create directories: `face_shape_Ml` and `body-shape-api`
2. Add model files:
   - `face_shape_Ml/face_shape_model.h5`
   - `body-shape-api/body_shape_model.pkl`

When the models are present, the API will automatically use them instead of fallbacks.

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python startup.py
```

## About Fallback Mode

When ML models aren't available, this API uses intelligent fallbacks that:
- Return realistic, plausible results
- Include appropriate confidence scores
- Maintain the same API structure
- Include a flag `using_fallback: true` to indicate fallback usage
