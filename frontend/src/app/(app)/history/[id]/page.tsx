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

export default function ResultDetailsPage() {
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

  const isFake = useMemo(() => record?.prediction_label === "FAKE", [record?.prediction_label])

  return (
    <div>
      <Topbar title="Result Details" subtitle="Forensic case detail view for a single analysis." />
      <div className="space-y-6 p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Link href="/history">
            <Button variant="outline">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to history
            </Button>
          </Link>
          {record?.heatmap_path ? (
            <Link href={`/heatmaps/${record.id}`}>
              <Button variant="secondary">
                View heatmap <Flame className="ml-2 h-4 w-4" />
              </Button>
            </Link>
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
        ) : (
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
            <Card className="xl:col-span-2">
              <CardHeader className="flex flex-row items-center justify-between space-y-0">
                <CardTitle>Case Summary</CardTitle>
                <Badge variant={isFake ? "fake" : "real"}>{record.prediction_label}</Badge>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
                  <Metric label="Confidence" value={`${Number(record.confidence).toFixed(2)}%`} />
                  <Metric label="Fake probability" value={`${Number(record.fake_probability).toFixed(2)}%`} />
                  <Metric label="Real probability" value={`${Number(record.real_probability).toFixed(2)}%`} />
                  <Metric label="Raw prediction" value={Number(record.raw_prediction).toFixed(6)} />
                </div>
                <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                  <div className="text-sm font-semibold text-slate-900">Explanation</div>
                  <div className="mt-1 text-sm text-slate-700">{record.explanation || "—"}</div>
                </div>
                {record.heatmap_path ? (
                  <div className="rounded-xl border border-slate-200 bg-white p-4">
                    <div className="text-sm font-semibold text-slate-900">Heatmap</div>
                    <div className="mt-2 text-sm text-slate-600">Heatmap path saved for this analysis.</div>
                    <div className="mt-3 flex flex-wrap gap-3">
                      <Link href={`/heatmaps/${record.id}`}>
                        <Button variant="secondary">
                          Open viewer <Flame className="ml-2 h-4 w-4" />
                        </Button>
                      </Link>
                      <a href={resolveBackendPath(record.heatmap_path)} target="_blank" rel="noreferrer">
                        <Button variant="outline">Open raw image</Button>
                      </a>
                    </div>
                  </div>
                ) : null}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Metadata</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <MetaRow label="Record ID" value={String(record.id)} />
                <MetaRow label="Filename" value={record.original_filename} />
                <MetaRow label="Timestamp" value={new Date(record.created_at).toLocaleString()} />
                <MetaRow label="User ID" value={record.user_id === null ? "Guest" : String(record.user_id)} />
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="mt-1 text-base font-semibold text-slate-900">{value}</div>
    </div>
  )
}

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3">
      <div className="text-slate-600">{label}</div>
      <div className="max-w-[65%] truncate font-medium text-slate-900">{value}</div>
    </div>
  )
}

