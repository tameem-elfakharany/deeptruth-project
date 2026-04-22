export type PredictionResponse = {
  success: boolean
  filename: string
  prediction_label: "REAL" | "FAKE" | string
  raw_prediction: number
  fake_probability: number
  real_probability: number
  confidence: number
  threshold_used: number
  explanation: string
  heatmap_path: string | null
  // ONNX model extended fields (present when new model is active)
  fake_type: string | null
  fake_type_confidence: number | null
  type_probabilities: Record<string, number> | null
}

export type PredictionRecord = {
  id: number
  user_id: number | null
  original_filename: string
  prediction_label: string
  raw_prediction: number
  fake_probability: number
  real_probability: number
  confidence: number
  explanation: string | null
  heatmap_path: string | null
  created_at: string
}

