import type { PredictionRecord } from "@/types/prediction"

export function summarize(records: PredictionRecord[]) {
  const total = records.length
  const fake = records.filter((r) => r.prediction_label === "FAKE").length
  const real = records.filter((r) => r.prediction_label === "REAL").length
  const avgConfidence =
    total === 0 ? 0 : records.reduce((acc, r) => acc + Number(r.confidence || 0), 0) / total
  const withHeatmap = records.filter((r) => Boolean(r.heatmap_path)).length
  return { total, fake, real, avgConfidence, withHeatmap }
}

export function dailyCounts(records: PredictionRecord[], days: number) {
  const now = new Date()
  const buckets: { date: string; count: number }[] = []
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(now)
    d.setDate(now.getDate() - i)
    const key = d.toISOString().slice(0, 10)
    buckets.push({ date: key, count: 0 })
  }
  const index = new Map(buckets.map((b, i) => [b.date, i]))
  for (const r of records) {
    const key = String(r.created_at || "").slice(0, 10)
    const i = index.get(key)
    if (i !== undefined) buckets[i].count += 1
  }
  return buckets
}

export function confidenceBuckets(records: PredictionRecord[]) {
  const ranges = [
    { range: "0-20", min: 0, max: 20 },
    { range: "20-40", min: 20, max: 40 },
    { range: "40-60", min: 40, max: 60 },
    { range: "60-80", min: 60, max: 80 },
    { range: "80-100", min: 80, max: 101 }
  ]
  const buckets = ranges.map((r) => ({ range: r.range, count: 0 }))
  for (const rec of records) {
    const c = Number(rec.confidence || 0)
    const i = ranges.findIndex((r) => c >= r.min && c < r.max)
    if (i >= 0) buckets[i].count += 1
  }
  return buckets
}

export function heatmapAvailability(records: PredictionRecord[]) {
  const withHeatmap = records.filter((r) => Boolean(r.heatmap_path)).length
  const withoutHeatmap = records.length - withHeatmap
  return [
    { name: "With heatmap", value: withHeatmap },
    { name: "No heatmap", value: withoutHeatmap }
  ]
}

