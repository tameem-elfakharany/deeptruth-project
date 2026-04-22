# DeepTruth Backend (Milestone 1)

Production-ready FastAPI backend for binary deepfake image detection using your trained TensorFlow/Keras Xception classifier.

## Structure

```
backend/
├── app/
│   ├── db.py
│   ├── security.py
│   ├── main.py
│   ├── config.py
│   ├── api/
│   │   └── routes.py
│   ├── schemas/
│   │   └── prediction.py
│   ├── services/
│   │   ├── model_loader.py
│   │   ├── inference.py
│   │   └── explainability.py
│   └── utils/
│       ├── image_processing.py
│       └── validators.py
├── uploads/
├── outputs/
│   └── heatmaps/
├── deeptruth.db
├── requirements.txt
└── run.sh
```

`uploads/` and `outputs/heatmaps/` are created automatically at startup if missing.

## SQLite Database

The backend uses a lightweight SQLite database stored at:

```
backend/deeptruth.db
```

It is created automatically on API startup if missing.

### Tables

- `users`
  - `id`, `full_name`, `email` (unique), `password` (hashed), `created_at`
- `prediction_records`
  - One row per successful prediction (guest mode supported with `user_id = NULL`)
  - Stores fields aligned with the `/predict` response: label, raw score, probabilities, confidence, explanation, and optional heatmap path

### How Predictions Are Saved

After a successful `/predict` or `/predict-with-heatmap` inference, the backend inserts a row into `prediction_records` before returning the API response.

### Guest vs Logged-In Predictions

- If the request includes a valid `Authorization: Bearer <token>` header, the backend links the prediction to the authenticated user by saving `prediction_records.user_id = users.id`.
- If no token is provided (or the token is invalid on these optional-auth endpoints), the prediction still succeeds and is saved with `user_id = NULL` (guest mode).

### Login-Ready Support

`app/db.py` includes helper functions for registering and validating users:
- `create_user`
- `get_user_by_email`
- `verify_user_login`

Authentication/JWT is intentionally not implemented yet.

## Model Placement (Critical)

The backend loads the trained model from:

```
models/xception_deepfake_classifier.h5
```

This path is relative to the project root (`DeepTruth/`).

## Preprocessing (Matches Notebook)

1. Decode uploaded image bytes
2. Convert BGR → RGB
3. Resize to 224×224
4. Cast to float32
5. Divide by 255.0
6. Expand dims before prediction

## Run

From the project root:

```bash
pip install -r backend/requirements.txt
bash backend/run.sh
```

API docs:
- `http://localhost:8000/docs`

## Endpoints

### GET /health

Returns API status and whether the model is loaded.

### POST /predict

Upload an image and get REAL/FAKE prediction with confidence percentages.

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/image.jpg"
```

### POST /predict-with-heatmap (Optional)

Same as `/predict` but also generates a Grad-CAM overlay heatmap saved under `backend/outputs/heatmaps/`.

```bash
curl -X POST "http://localhost:8000/predict-with-heatmap" \
  -F "file=@path/to/image.jpg"
```

## Sample Response

```json
{
  "success": true,
  "filename": "image1.jpg",
  "prediction_label": "FAKE",
  "raw_prediction": 0.8732,
  "fake_probability": 87.32,
  "real_probability": 12.68,
  "confidence": 87.32,
  "threshold_used": 0.5,
  "explanation": "The model predicts this image is likely fake because the fake-class probability is above the decision threshold.",
  "heatmap_path": null
}
```
