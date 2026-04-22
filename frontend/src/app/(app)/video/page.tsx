"use client"

import { useMemo, useState } from "react"
import { ArrowUpRight, FileVideo, Sparkles } from "lucide-react"

import { Topbar } from "@/components/layout/topbar"
import { UploadDropzone } from "@/components/analysis/upload-dropzone"
import { ResultPanel } from "@/components/analysis/result-panel"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { PredictionResponse } from "@/types/prediction"
import { predictVideo } from "@/services/predictions"

export default function VideoAnalysisPage() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)

  const accepted = "MP4, AVI, MOV, MKV"

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
      const res = await predictVideo(file)
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Video prediction failed.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <Topbar title="Video Analysis" subtitle="Upload a video to detect deepfake content using LipNet 3D-CNN." />
      <div className="space-y-6 p-6">
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
          <div className="space-y-6 xl:col-span-2">
            <UploadDropzone
              onFileSelected={onSelectFile}
              accept="video/mp4,video/x-msvideo,video/quicktime,video/x-matroska"
              title="Drag and drop a video"
              note={`Accepted types: ${accepted}. Max size 100MB.`}
            />

            {previewUrl ? (
              <Card>
                <CardHeader>
                  <CardTitle>Video Preview</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="overflow-hidden rounded-xl border border-slate-200 bg-black aspect-video flex items-center justify-center">
                    <video 
                      src={previewUrl} 
                      controls 
                      className="max-h-full max-w-full"
                    />
                  </div>
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="text-sm text-slate-600">
                      <span className="font-medium text-slate-900">{file?.name}</span>
                    </div>
                    <Button onClick={run} disabled={!canAnalyze}>
                      {loading ? "Analyzing Video..." : "Analyze video"}
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
              <ResultPanel result={result} />
            ) : null}
          </div>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Spatiotemporal Analysis</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-slate-700">
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 grid h-9 w-9 place-items-center rounded-xl bg-brand-50 text-brand-700">
                    <FileVideo className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="font-semibold text-slate-900">LipNet 3D-CNN Architecture</div>
                    <div className="mt-1 text-slate-600">
                      Our model uses 3D convolutions to analyze movement and texture simultaneously across 20 sampled frames.
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 grid h-9 w-9 place-items-center rounded-xl bg-slate-100 text-slate-700">
                    <Badge>Temporal</Badge>
                  </div>
                  <div>
                    <div className="font-semibold text-slate-900">Motion Consistency</div>
                    <div className="mt-1 text-slate-600">
                      Detects frame-to-frame inconsistencies that are often present in generative video deepfakes.
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 grid h-9 w-9 place-items-center rounded-xl bg-slate-100 text-slate-700">
                    <Sparkles className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="font-semibold text-slate-900">Forensic Evidence</div>
                    <div className="mt-1 text-slate-600">
                      The analysis focuses on facial dynamics and lip-sync patterns to identify manipulation.
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Guidelines</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-slate-700">
                <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3">
                  <span>Focus on facial sequences</span>
                  <Badge variant="outline">Optimal</Badge>
                </div>
                <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3">
                  <span>Length: 2-10 seconds</span>
                  <Badge variant="outline">Best Results</Badge>
                </div>
                <div className="mt-4">
                  <a href="/history" className="text-sm font-medium text-brand-600 hover:text-brand-700">
                    Review video history <ArrowUpRight className="ml-1 inline h-4 w-4" />
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
