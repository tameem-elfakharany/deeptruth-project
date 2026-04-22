"use client"

import { useMemo, useState } from "react"
import { ArrowUpRight, Mic, Sparkles } from "lucide-react"

import { Topbar } from "@/components/layout/topbar"
import { UploadDropzone } from "@/components/analysis/upload-dropzone"
import { ResultPanel } from "@/components/analysis/result-panel"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { PredictionResponse } from "@/types/prediction"
import { predictAudio } from "@/services/predictions"

export default function AudioAnalysisPage() {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)

  const accepted = "WAV, MP3, FLAC, OGG, M4A"

  const canAnalyze = useMemo(() => Boolean(file) && !loading, [file, loading])

  const onSelectFile = (f: File) => {
    setError(null)
    setResult(null)
    setFile(f)
  }

  const run = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const res = await predictAudio(file)
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Audio prediction failed.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <Topbar title="Audio Analysis" subtitle="Upload an audio file to detect synthetic speech and voice manipulation." />
      <div className="space-y-6 p-6">
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
          <div className="space-y-6 xl:col-span-2">
            <UploadDropzone
              onFileSelected={onSelectFile}
              accept="audio/wav,audio/mpeg,audio/flac,audio/ogg,audio/mp4,audio/x-m4a"
              title="Drag and drop an audio file"
              note={`Accepted types: ${accepted}. Max size 50MB.`}
            />

            {file ? (
              <Card>
                <CardHeader>
                  <CardTitle>Audio File</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="overflow-hidden rounded-xl border border-slate-200 bg-slate-50 p-6 flex items-center gap-4">
                    <div className="grid h-12 w-12 place-items-center rounded-xl bg-brand-50 text-brand-700 shrink-0">
                      <Mic className="h-6 w-6" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="truncate font-medium text-slate-900">{file.name}</div>
                      <div className="mt-0.5 text-sm text-slate-500">
                        {(file.size / (1024 * 1024)).toFixed(2)} MB
                      </div>
                    </div>
                  </div>
                  <audio controls src={URL.createObjectURL(file)} className="w-full" />
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="text-sm text-slate-600">
                      Ready for analysis
                    </div>
                    <Button onClick={run} disabled={!canAnalyze}>
                      {loading ? "Analyzing Audio..." : "Analyze audio"}
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
                <CardTitle>Acoustic Analysis</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-slate-700">
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 grid h-9 w-9 place-items-center rounded-xl bg-brand-50 text-brand-700">
                    <Mic className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="font-semibold text-slate-900">Wav2Vec2 Architecture</div>
                    <div className="mt-1 text-slate-600">
                      Our model uses Facebook's Wav2Vec2 transformer to extract deep acoustic features directly from raw waveforms.
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 grid h-9 w-9 place-items-center rounded-xl bg-slate-100 text-slate-700">
                    <Badge>Spectral</Badge>
                  </div>
                  <div>
                    <div className="font-semibold text-slate-900">Frequency Artifacts</div>
                    <div className="mt-1 text-slate-600">
                      Detects unnatural frequency patterns and prosodic inconsistencies common in TTS and voice-cloning systems.
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 grid h-9 w-9 place-items-center rounded-xl bg-slate-100 text-slate-700">
                    <Sparkles className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="font-semibold text-slate-900">Speaker Verification</div>
                    <div className="mt-1 text-slate-600">
                      Analyzes voice embeddings to identify synthetic speech and potential identity spoofing.
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
                  <span>Clear speech recordings</span>
                  <Badge variant="outline">Optimal</Badge>
                </div>
                <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3">
                  <span>Length: 2–10 seconds</span>
                  <Badge variant="outline">Best Results</Badge>
                </div>
                <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3">
                  <span>Sample rate: 16kHz+</span>
                  <Badge variant="outline">Recommended</Badge>
                </div>
                <div className="mt-4">
                  <a href="/history" className="text-sm font-medium text-brand-600 hover:text-brand-700">
                    Review audio history <ArrowUpRight className="ml-1 inline h-4 w-4" />
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
