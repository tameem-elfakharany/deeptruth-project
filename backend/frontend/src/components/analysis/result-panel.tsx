"use client"

import Link from "next/link"
import { ArrowUpRight, Flame } from "lucide-react"

import type { PredictionResponse } from "@/types/prediction"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function ResultPanel({
  result,
  onViewHeatmap
}: {
  result: PredictionResponse
  onViewHeatmap?: () => void
}) {
  const isFake = result.prediction_label === "FAKE"
  const hasTypeBreakdown = result.type_probabilities && Object.keys(result.type_probabilities).length > 0

  // Sort type probabilities descending, exclude "Real" when result is FAKE
  const sortedTypes = hasTypeBreakdown
    ? Object.entries(result.type_probabilities!)
        .filter(([name]) => isFake ? name !== "Real" : name === "Real")
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5)
    : []

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle>Analysis Result</CardTitle>
        <Badge variant={isFake ? "fake" : "real"}>{result.prediction_label}</Badge>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Core metrics */}
        <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
          <Metric label="Confidence" value={`${Number(result.confidence).toFixed(2)}%`} />
          <Metric label="Fake probability" value={`${Number(result.fake_probability).toFixed(2)}%`} />
          <Metric label="Real probability" value={`${Number(result.real_probability).toFixed(2)}%`} />
          <Metric label="Raw prediction" value={Number(result.raw_prediction).toFixed(6)} />
        </div>

        {/* Manipulation type (ONNX model only) */}
        {result.fake_type && result.fake_type !== "Real" && isFake ? (
          <div className="rounded-xl border border-orange-200 bg-orange-50 px-4 py-3">
            <div className="flex items-center gap-2">
              <div className="text-sm font-semibold text-orange-900">Detected manipulation type</div>
              <Badge variant="outline" className="border-orange-300 text-orange-800 text-xs">
                {result.fake_type}
              </Badge>
              {result.fake_type_confidence != null ? (
                <span className="ml-auto text-xs text-orange-700">
                  {result.fake_type_confidence.toFixed(1)}% confidence
                </span>
              ) : null}
            </div>
          </div>
        ) : null}

        {/* Type probability breakdown */}
        {sortedTypes.length > 1 ? (
          <div className="space-y-2">
            <div className="text-sm font-semibold text-slate-700">Manipulation type breakdown</div>
            <div className="space-y-1.5">
              {sortedTypes.map(([name, pct]) => (
                <TypeBar key={name} label={name} pct={pct} isTop={sortedTypes[0][0] === name} />
              ))}
            </div>
          </div>
        ) : null}

        {/* Explanation */}
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
          <div className="text-sm font-semibold text-slate-900">Explanation</div>
          <div className="mt-1 text-sm text-slate-700">{result.explanation}</div>
        </div>

        {/* Actions */}
        <div className="flex flex-wrap gap-3">
          {result.heatmap_path ? (
            <Button type="button" variant="secondary" onClick={onViewHeatmap}>
              View heatmap <Flame className="ml-2 h-4 w-4" />
            </Button>
          ) : null}
          <Link href="/history">
            <Button type="button" variant="outline">
              View history <ArrowUpRight className="ml-2 h-4 w-4" />
            </Button>
          </Link>
        </div>
      </CardContent>
    </Card>
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

function TypeBar({ label, pct, isTop }: { label: string; pct: number; isTop: boolean }) {
  return (
    <div className="flex items-center gap-3">
      <div className="w-40 shrink-0 truncate text-xs text-slate-600">{label}</div>
      <div className="flex-1 h-2 rounded-full bg-slate-100 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${isTop ? "bg-brand-500" : "bg-slate-300"}`}
          style={{ width: `${Math.min(pct, 100).toFixed(1)}%` }}
        />
      </div>
      <div className="w-12 shrink-0 text-right text-xs text-slate-500">{pct.toFixed(1)}%</div>
    </div>
  )
}
