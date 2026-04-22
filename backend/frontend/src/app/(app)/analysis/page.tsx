"use client"

import { useMemo, useState } from "react"
import { ArrowUpRight, FileImage, Sparkles } from "lucide-react"

import { Topbar } from "@/components/layout/topbar"
import { UploadDropzone } from "@/components/analysis/upload-dropzone"
import { ResultPanel } from "@/components/analysis/result-panel"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { PredictionResponse } from "@/types/prediction"
import { predictImage } from "@/services/predictions"
import { resolveBackendPath } from "@/lib/asset"

export default function ImageAnalysisPage() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [withHeatmap, setWithHeatmap] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)

  const accepted = "JPG, JPEG, PNG, BMP, WEBP"

  const canAnalyze = useMemo(() => Boolean(file) && !loading, [file, loading])

  const onSelectFile = (f: File) => {
    setError(null)
    setResult(null)
    setFile(f)
    const url = URL.createObjectURL(f)
    setPreviewUrl(url)
  }

  const run = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const res = await predictImage(file, withHeatmap)
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <Topbar title="Image Analysis" subtitle="Upload an image to detect whether it is REAL or FAKE." />
      <div className="space-y-6 p-6">
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
          <div className="space-y-6 xl:col-span-2">
            <UploadDropzone
              onFileSelected={onSelectFile}
              accept="image/jpeg,image/png,image/bmp,image/webp"
              note={`Accepted types: ${accepted}.`}
            />

            {previewUrl ? (
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0">
                  <CardTitle>Preview</CardTitle>
                  <div className="flex items-center gap-2">
                    <label className="flex items-center gap-2 text-sm text-slate-700">
                      <input
                        type="checkbox"
                        className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
                        checked={withHeatmap}
                        onChange={(e) => setWithHeatmap(e.target.checked)}
                      />
                      Generate heatmap
                    </label>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="overflow-hidden rounded-xl border border-slate-200 bg-white">
                    <img src={previewUrl} alt="Selected file preview" className="max-h-[360px] w-full object-contain" />
                  </div>
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="text-sm text-slate-600">
                      <span className="font-medium text-slate-900">{file?.name}</span>
                    </div>
                    <Button onClick={run} disabled={!canAnalyze}>
                      {loading ? "Analyzing..." : "Analyze image"}
                      <Sparkles className="ml-2 h-4 w-4" />
                    </Button>
                  </div>
                  {error ? (
                    <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                      {error}
                    </div>
                  ) : null}
                </CardContent>
              </Card>
            ) : null}

            {result ? (
              <ResultPanel
                result={result}
                onViewHeatmap={() => {
                  if (!result.heatmap_path) return
                  const url = resolveBackendPath(result.heatmap_path)
                  window.open(url, "_blank", "noopener,noreferrer")
                }}
              />
            ) : null}
          </div>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>How this works</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-slate-700">
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 grid h-9 w-9 place-items-center rounded-xl bg-brand-50 text-brand-700">
                    <FileImage className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="font-semibold text-slate-900">Artifact-focused classification</div>
                    <div className="mt-1 text-slate-600">
                      The model evaluates visual cues that can indicate synthetic generation or manipulation.
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 grid h-9 w-9 place-items-center rounded-xl bg-slate-100 text-slate-700">
                    <Badge>Confidence</Badge>
                  </div>
                  <div>
                    <div className="font-semibold text-slate-900">Confidence scoring</div>
                    <div className="mt-1 text-slate-600">
                      Outputs REAL/FAKE plus calibrated probabilities and a clear threshold-based explanation.
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 grid h-9 w-9 place-items-center rounded-xl bg-slate-100 text-slate-700">
                    <Sparkles className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="font-semibold text-slate-900">Explainability heatmaps</div>
                    <div className="mt-1 text-slate-600">
                      Optional Grad-CAM highlights regions that influenced the decision. This is an interpretability aid,
                      not proof.
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Tips</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-slate-700">
                <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3">
                  <span>Use high-resolution images</span>
                  <Badge variant="outline">Recommended</Badge>
                </div>
                <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3">
                  <span>Compare with heatmap when unsure</span>
                  <Badge variant="outline">Explainability</Badge>
                </div>
                <div className="mt-4">
                  <a href="/history" className="text-sm font-medium text-brand-600 hover:text-brand-700">
                    View saved analyses <ArrowUpRight className="ml-1 inline h-4 w-4" />
                  </a>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}

