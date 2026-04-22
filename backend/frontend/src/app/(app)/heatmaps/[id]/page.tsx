"use client"

import Link from "next/link"
import { useParams } from "next/navigation"
import { useEffect, useMemo, useState } from "react"
import { ArrowLeft, Flame } from "lucide-react"

import type { PredictionRecord } from "@/types/prediction"
import { getPredictionById } from "@/services/predictions"
import { Topbar } from "@/components/layout/topbar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { resolveBackendPath } from "@/lib/asset"

export default function HeatmapViewerDetailPage() {
  const params = useParams<{ id: string }>()
  const id = params?.id
  const [record, setRecord] = useState<PredictionRecord | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!id) return
    setLoading(true)
    setError(null)
    getPredictionById(id)
      .then((r) => setRecord(r))
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load record."))
      .finally(() => setLoading(false))
  }, [id])

  const heatmapUrl = useMemo(() => {
    if (!record?.heatmap_path) return null
    return resolveBackendPath(record.heatmap_path)
  }, [record?.heatmap_path])

  return (
    <div>
      <Topbar title="Heatmap Viewer" subtitle="Interpretability view highlighting regions that influenced the model." />
      <div className="space-y-6 p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Link href="/heatmaps">
            <Button variant="outline">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to heatmaps
            </Button>
          </Link>
          {heatmapUrl ? (
            <a href={heatmapUrl} target="_blank" rel="noreferrer">
              <Button variant="secondary">
                Open raw heatmap <Flame className="ml-2 h-4 w-4" />
              </Button>
            </a>
          ) : null}
        </div>

        {loading ? (
          <div className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-600 shadow-soft">
            Loading...
          </div>
        ) : error ? (
          <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-sm text-red-700 shadow-soft">
            {error}
          </div>
        ) : !record ? (
          <div className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-600 shadow-soft">
            No record found.
          </div>
        ) : !heatmapUrl ? (
          <div className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-600 shadow-soft">
            This analysis does not have a heatmap.
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
            <Card className="xl:col-span-2">
              <CardHeader>
                <CardTitle>Comparison</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                  <div className="rounded-2xl border border-slate-200 bg-white p-4">
                    <div className="text-sm font-semibold text-slate-900">Original</div>
                    <div className="mt-2 grid h-[320px] place-items-center rounded-xl border border-dashed border-slate-200 bg-slate-50 text-sm text-slate-600">
                      Original image is not available from the API response.
                    </div>
                  </div>
                  <div className="rounded-2xl border border-slate-200 bg-white p-4">
                    <div className="text-sm font-semibold text-slate-900">Heatmap</div>
                    <div className="mt-2 overflow-hidden rounded-xl border border-slate-200 bg-white">
                      <img src={heatmapUrl} alt="Heatmap overlay" className="h-[320px] w-full object-contain" />
                    </div>
                  </div>
                </div>
                <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
                  Highlighted regions indicate areas that influenced the model’s decision most. This visualization is an
                  interpretability aid and should not be treated as proof.
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3">
                  <div className="text-slate-600">Filename</div>
                  <div className="max-w-[65%] truncate font-medium text-slate-900">{record.original_filename}</div>
                </div>
                <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3">
                  <div className="text-slate-600">Prediction</div>
                  <Badge variant={record.prediction_label === "FAKE" ? "fake" : "real"}>{record.prediction_label}</Badge>
                </div>
                <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3">
                  <div className="text-slate-600">Confidence</div>
                  <div className="font-medium text-slate-900">{Number(record.confidence).toFixed(2)}%</div>
                </div>
                <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3">
                  <div className="text-slate-600">Timestamp</div>
                  <div className="font-medium text-slate-900">{new Date(record.created_at).toLocaleString()}</div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  )
}

