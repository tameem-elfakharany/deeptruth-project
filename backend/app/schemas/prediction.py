from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    success: bool
    filename: str
    prediction_label: str
    raw_prediction: float = Field(ge=0.0, le=1.0)
    fake_probability: float = Field(ge=0.0, le=100.0)
    real_probability: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=100.0)
    threshold_used: float
    explanation: str
    heatmap_path: str | None = None
    # ONNX model extended fields
    fake_type: str | None = None
    fake_type_confidence: float | None = None
    type_probabilities: dict | None = None
    # Multi-face detection fields
    faces_detected: int = 1
    flagged_face_index: int | None = None

