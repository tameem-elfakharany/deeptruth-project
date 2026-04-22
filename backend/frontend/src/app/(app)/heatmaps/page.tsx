"use client"

import Link from "next/link"
import { useEffect, useMemo, useState } from "react"
import { Flame, Search } from "lucide-react"

import type { PredictionRecord } from "@/types/prediction"
import { getMyPredictionHistory } from "@/services/predictions"
import { Topbar } from "@/components/layout/topbar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

export default function HeatmapsPage() {
  const [records, setRecords] = useState<PredictionRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState("")

  useEffect(() => {
    setLoading(true)
    setError(null)
    getMyPredictionHistory()
      .then((r) => setRecords(r))
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load history."))
      .finally(() => setLoading(false))
  }, [])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return records
      .filter((r) => Boolean(r.heatmap_path))
      .filter((r) => (q ? r.original_filename.toLowerCase().includes(q) : true))
  }, [query, records])

  return (
    <div>
      <Topbar title="Heatmap Viewer" subtitle="Browse explainability outputs and compare influential regions." />
      <div className="space-y-6 p-6">
        <Card>
          <CardContent className="p-6">
            <div className="relative max-w-xl">
              <Search className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
              <Input
                placeholder="Search heatmaps by filename..."
                className="pl-9"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-0">
            {loading ? (
              <div className="p-6 text-sm text-slate-600">Loading...</div>
            ) : error ? (
              <div className="m-6 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                {error}
              </div>
            ) : filtered.length === 0 ? (
              <div className="p-10 text-center">
                <div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-slate-100 text-slate-700">
                  <Flame className="h-5 w-5" />
                </div>
                <div className="mt-3 text-sm font-medium text-slate-900">No heatmaps available</div>
                <div className="mt-1 text-sm text-slate-600">
                  Run an image analysis with heatmap generation enabled to populate this page.
                </div>
                <div className="mt-4">
                  <Link href="/analysis">
                    <Button>
                      Generate heatmap <Flame className="ml-2 h-4 w-4" />
                    </Button>
                  </Link>
                </div>
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Filename</TableHead>
                    <TableHead>Result</TableHead>
                    <TableHead>Confidence</TableHead>
                    <TableHead>Date</TableHead>
                    <TableHead className="text-right">Action</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filtered.map((r) => (
                    <TableRow key={r.id}>
                      <TableCell className="font-medium text-slate-900">{r.original_filename}</TableCell>
                      <TableCell>
                        <Badge variant={r.prediction_label === "FAKE" ? "fake" : "real"}>{r.prediction_label}</Badge>
                      </TableCell>
                      <TableCell>{Number(r.confidence).toFixed(2)}%</TableCell>
                      <TableCell className="text-slate-600">{new Date(r.created_at).toLocaleString()}</TableCell>
                      <TableCell className="text-right">
                        <Link href={`/heatmaps/${r.id}`}>
                          <Button size="sm" variant="secondary">
                            View
                          </Button>
                        </Link>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

