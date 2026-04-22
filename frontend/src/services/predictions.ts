import { apiFetch } from "@/services/api"
import { endpoints } from "@/services/endpoints"
import type { PredictionRecord, PredictionResponse } from "@/types/prediction"

export async function predictImage(file: File, withHeatmap: boolean): Promise<PredictionResponse> {
  const formData = new FormData()
  formData.append("file", file)
  const path = withHeatmap ? endpoints.predictions.predictWithHeatmap : endpoints.predictions.predict
  return apiFetch<PredictionResponse>(path, { method: "POST", body: formData })
}

export async function predictAudio(file: File): Promise<PredictionResponse> {
  const formData = new FormData()
  formData.append("file", file)
  return apiFetch<PredictionResponse>(endpoints.predictions.predictAudio, { method: "POST", body: formData })
}

export async function predictVideo(file: File): Promise<PredictionResponse> {
  const formData = new FormData()
  formData.append("file", file)
  return apiFetch<PredictionResponse>(endpoints.predictions.predictVideo, { method: "POST", body: formData })
}

export async function getMyPredictionHistory(): Promise<PredictionRecord[]> {
  return apiFetch<PredictionRecord[]>(endpoints.predictions.myHistory, { method: "GET" })
}

export async function getRecentPredictions(limit = 10): Promise<PredictionRecord[]> {
  const url = `${endpoints.predictions.recent}?limit=${encodeURIComponent(String(limit))}`
  return apiFetch<PredictionRecord[]>(url, { method: "GET" })
}

export async function getPredictionById(id: number | string): Promise<PredictionRecord> {
  return apiFetch<PredictionRecord>(endpoints.predictions.byId(id), { method: "GET" })
}

