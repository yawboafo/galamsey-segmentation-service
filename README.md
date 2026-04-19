# Galamsey Segmentation Service

This is a standalone AI service designed to detect illegal mining (galamsey) footprints from satellite image chips using the SegFormer architecture.

## Features
- **Binary Semantic Segmentation**: Fine-tuned SegFormer for `background` (0) and `galamsey` (1) classes.
- **FastAPI Inference**: Production-ready API supporting image uploads and URL-based inference.
- **GeoJSON Output**: Returns cleaned, simplified polygons in GeoJSON format.
- **Training Pipeline**: End-to-end training, evaluation, and model pushing to Hugging Face Hub.
- **Post-Processing**: Automatic noise removal, binary thresholding, and polygon simplification.

---

## Project Structure

```text
galamsey-segmentation-service/
  app/                    # FastAPI Inference Service
    api/                  # API routes (predict, health)
    services/             # Model inference & post-processing logic
    schemas/              # Pydantic request/response models
    utils/                # Mask-to-polygon utilities
    main.py               # API Entry point
    config.py             # App configuration (environment variables)
  training/               # Model Training Pipeline
    train.py              # Fine-tuning script
    dataset.py            # HF/PyTorch dataset loader
    metrics.py            # Evaluation metrics (IoU, F1 score)
    config.py             # Training hyperparameters
  scripts/                # Standalone utilities
    prepare_dataset.py    # Data splitting (train/val/test)
    validate_dataset.py   # Data integrity checks
    test_inference.py     # Local API testing script
  models/                 # Local model storage
  data/                   # Dataset storage (raw and processed)
  Dockerfile              # Containerization for deployment
  requirements.txt        # Python dependencies
  .env.example            # Environment configuration template
```

---

## Getting Started

### 1. Installation
Clone this repository and install dependencies:
```bash
pip install -r requirements.txt
cp .env.example .env
```

### 2. Dataset Preparation
Place your raw image chips and binary masks in `data/raw/images/` and `data/raw/masks/`. Ensure matching filenames.

Run the preparation script to create splits:
```bash
python scripts/prepare_dataset.py
```

Validate your dataset:
```bash
python scripts/validate_dataset.py
```

---

## Training the Model

To start fine-tuning SegFormer on your custom dataset:
```bash
export PYTHONPATH=$PYTHONPATH:.
python training/train.py
```
Training parameters (batch size, learning rate, epochs) can be adjusted in `.env` or `training/config.py`.

---

## Running the Inference API

### Local Development
```bash
uvicorn app.main:app --reload
```

### Using Docker
```bash
docker build -t galamsey-segmentation .
docker run -p 8000:8000 galamsey-segmentation
```

---

## API Usage

### Endpoint: `POST /api/v1/predict`

**Parameters (Query or Multi-part):**
- `file`: Image file upload (multipart)
- `image_url`: URL of the image chip (optional)
- `threshold`: Binarization threshold (default: 0.5)
- `min_area`: Minimum pixel area to keep (default: 50)

**Example Response:**
```json
{
  "success": true,
  "confidence": 0.87,
  "area_geojson": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "Polygon",
          "coordinates": [[[x1, y1], [x2, y2], ...]]
        },
        "properties": {
          "area_px": 1240.5,
          "is_galamsey": true
        }
      }
    ]
  },
  "metadata": {
    "model_name": "segformer-b0-galamsey",
    "threshold": 0.5
  }
}
```

### Testing Inference
Use the provided script to test a local image against the running API:
```bash
python scripts/test_inference.py --image path/to/chip.jpg
```

---

## Deployment to Hugging Face
This service is designed to be compatible with Hugging Face Inference Endpoints. You can push your trained model directly using the training pipeline by setting `PUSH_TO_HUB=True` in your environment.
