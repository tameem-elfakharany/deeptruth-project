"use client"

import Link from "next/link"
import { useEffect, useMemo, useState } from "react"
import { ArrowUpRight, Filter, Search } from "lucide-react"

import type { PredictionRecord } from "@/types/prediction"
import { getMyPredictionHistory } from "@/services/predictions"
import { Topbar } from "@/components/layout/topbar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

type FilterValue = "all" | "real" | "fake" | "with_heatmap"
type SortValue = "newest" | "oldest" | "highest_confidence"

export default function HistoryPage() {
  const [records, setRecords] = useState<PredictionRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [query, setQuery] = useState("")
  const [filter, setFilter] = useState<FilterValue>("all")
  const [sort, setSort] = useState<SortValue>("newest")

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
    let list = records
    if (q) list = list.filter((r) => r.original_filename.toLowerCase().includes(q))
    if (filter === "real") list = list.filter((r) => r.prediction_label === "REAL")
    if (filter === "fake") list = list.filter((r) => r.prediction_label === "FAKE")
    if (filter === "with_heatmap") list = list.filter((r) => Boolean(r.heatmap_path))
    if (sort === "newest") list = [...list].sort((a, b) => +new Date(b.created_at) - +new Date(a.created_at))
    if (sort === "oldest") list = [...list].sort((a, b) => +new Date(a.created_at) - +new Date(b.created_at))
    if (sort === "highest_confidence") list = [...list].sort((a, b) => Number(b.confidence) - Number(a.confidence))
    return list
  }, [filter, query, records, sort])

  return (
    <div>
      <Topbar title="Analysis History" subtitle="Search, filter, and review your saved analyses." />
      <div className="space-y-6 p-6">
        <Card>
          <CardContent className="p-6">
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
              <div className="lg:col-span-6">
                <div className="relative">
                  <Search className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
                  <Input
                    placeholder="Search by filename..."
                    className="pl-9"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                </div>
              </div>
              <div className="lg:col-span-3">
                <div className="relative">
                  <Filter className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
                  <select
                    className="h-10 w-full appearance-none rounded-md border border-slate-200 bg-white pl-9 pr-3 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
                    value={filter}
                    onChange={(e) => setFilter(e.target.value as FilterValue)}
                  >
                    <option value="all">All</option>
                    <option value="fake">Fake</option>
                    <option value="real">Real</option>
                    <option value="with_heatmap">With heatmap</option>
                  </select>
                </div>
              </div>
              <div className="lg:col-span-3">
                <select
                  className="h-10 w-full appearance-none rounded-md border border-slate-200 bg-white px-3 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
                  value={sort}
                  onChange={(e) => setSort(e.target.value as SortValue)}
                >
                  <option value="newest">Newest first</option>
                  <option value="oldest">Oldest first</option>
                  <option value="highest_confidence">Highest confidence</option>
                </select>
              </div>
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
                <div className="text-sm font-medium text-slate-900">No analyses found</div>
                <div className="mt-1 text-sm text-slate-600">Try adjusting filters or run a new analysis.</div>
                <div className="mt-4">
                  <Link href="/analysis">
                    <Button>
                      Run analysis <ArrowUpRight className="ml-2 h-4 w-4" />
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
                    <TableHead>Heatmap</TableHead>
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
                      <TableCell>
                        {r.heatmap_path ? <Badge variant="default">Available</Badge> : <Badge variant="outline">No</Badge>}
                      </TableCell>
                      <TableCell className="text-slate-600">{new Date(r.created_at).toLocaleString()}</TableCell>
                      <TableCell className="text-right">
                        <Link href={`/history/${r.id}`}>
                          <Button size="sm" variant="outline">
                            Details
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

