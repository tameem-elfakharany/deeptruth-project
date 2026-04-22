"use client"

import Link from "next/link"
import { useEffect, useMemo, useState } from "react"
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts"
import { ArrowUpRight, BarChart3, CheckCircle2, Flame, ImageUp, SearchX, ShieldAlert } from "lucide-react"

import type { PredictionRecord } from "@/types/prediction"
import { getMyPredictionHistory } from "@/services/predictions"
import { confidenceBuckets, dailyCounts, heatmapAvailability, summarize } from "@/lib/analytics"
import { Topbar } from "@/components/layout/topbar"
import { StatCard } from "@/components/dashboard/stat-card"
import { ChartCard } from "@/components/dashboard/chart-card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

const PIE_COLORS = ["#1d5c96", "#ef4444"]

export default function DashboardPage() {
  const [records, setRecords] = useState<PredictionRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    getMyPredictionHistory()
      .then((r) => setRecords(r))
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load history."))
      .finally(() => setLoading(false))
  }, [])

  const summary = useMemo(() => summarize(records), [records])
  const dist = useMemo(
    () => [
      { name: "REAL", value: summary.real },
      { name: "FAKE", value: summary.fake }
    ],
    [summary.fake, summary.real]
  )
  const activity14 = useMemo(() => dailyCounts(records, 14), [records])
  const activity7 = useMemo(() => dailyCounts(records, 7), [records])
  const confBuckets = useMemo(() => confidenceBuckets(records), [records])
  const heatmapData = useMemo(() => heatmapAvailability(records), [records])
  const recent = useMemo(() => records.slice(0, 10), [records])

  return (
    <div>
      <Topbar title="Dashboard" subtitle="AI-powered deepfake detection and explainability platform" />
      <div className="space-y-6 p-6">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
          <StatCard label="Total Analyses" value={String(summary.total)} icon={<BarChart3 className="h-5 w-5" />} />
          <StatCard
            label="Fake Detected"
            value={String(summary.fake)}
            tone="bad"
            icon={<ShieldAlert className="h-5 w-5" />}
          />
          <StatCard
            label="Real Detected"
            value={String(summary.real)}
            tone="good"
            icon={<CheckCircle2 className="h-5 w-5" />}
          />
          <StatCard
            label="Average Confidence"
            value={`${summary.avgConfidence.toFixed(2)}%`}
            icon={<Flame className="h-5 w-5" />}
          />
        </div>

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
          <ChartCard title="Real vs Fake Distribution">
            {summary.total === 0 ? (
              <EmptyState />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Tooltip
                    contentStyle={{ borderRadius: 12, borderColor: "#e2e8f0" }}
                    formatter={(v: any, n: any) => [`${v}`, n]}
                  />
                  <Legend verticalAlign="bottom" height={24} />
                  <Pie
                    data={dist}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={70}
                    outerRadius={110}
                    paddingAngle={2}
                  >
                    {dist.map((_, i) => (
                      <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            )}
          </ChartCard>

          <ChartCard title="Recent Activity Trend (14 days)">
            {summary.total === 0 ? (
              <EmptyState />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={activity14}>
                  <CartesianGrid stroke="#e2e8f0" strokeDasharray="4 4" />
                  <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 12 }} />
                  <YAxis tick={{ fill: "#64748b", fontSize: 12 }} allowDecimals={false} />
                  <Tooltip contentStyle={{ borderRadius: 12, borderColor: "#e2e8f0" }} />
                  <Line type="monotone" dataKey="count" stroke="#1d5c96" strokeWidth={3} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </ChartCard>
        </div>

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
          <ChartCard title="Confidence Distribution">
            {summary.total === 0 ? (
              <EmptyState />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={confBuckets}>
                  <CartesianGrid stroke="#e2e8f0" strokeDasharray="4 4" />
                  <XAxis dataKey="range" tick={{ fill: "#64748b", fontSize: 12 }} />
                  <YAxis tick={{ fill: "#64748b", fontSize: 12 }} allowDecimals={false} />
                  <Tooltip contentStyle={{ borderRadius: 12, borderColor: "#e2e8f0" }} />
                  <Bar dataKey="count" fill="#1d5c96" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </ChartCard>

          <ChartCard title="User Activity (7 days)">
            {summary.total === 0 ? (
              <EmptyState />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={activity7}>
                  <defs>
                    <linearGradient id="activityFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#1d5c96" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#1d5c96" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#e2e8f0" strokeDasharray="4 4" />
                  <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 12 }} />
                  <YAxis tick={{ fill: "#64748b", fontSize: 12 }} allowDecimals={false} />
                  <Tooltip contentStyle={{ borderRadius: 12, borderColor: "#e2e8f0" }} />
                  <Area type="monotone" dataKey="count" stroke="#1d5c96" strokeWidth={2} fill="url(#activityFill)" />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </ChartCard>

          <ChartCard title="Heatmap Availability">
            {summary.total === 0 ? (
              <EmptyState />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={heatmapData} layout="vertical">
                  <CartesianGrid stroke="#e2e8f0" strokeDasharray="4 4" />
                  <XAxis type="number" tick={{ fill: "#64748b", fontSize: 12 }} allowDecimals={false} />
                  <YAxis type="category" dataKey="name" tick={{ fill: "#64748b", fontSize: 12 }} width={110} />
                  <Tooltip contentStyle={{ borderRadius: 12, borderColor: "#e2e8f0" }} />
                  <Bar dataKey="value" radius={[0, 8, 8, 0]}>
                    {heatmapData.map((_, i) => (
                      <Cell key={i} fill={i === 0 ? "#1d5c96" : "#94a3b8"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </ChartCard>
        </div>

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
          <Card className="xl:col-span-2">
            <CardHeader className="flex flex-row items-center justify-between space-y-0">
              <CardTitle>Recent Analyses</CardTitle>
              <Link href="/history" className="text-sm font-medium text-brand-600 hover:text-brand-700">
                View all <ArrowUpRight className="ml-1 inline h-4 w-4" />
              </Link>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="py-10 text-sm text-slate-600">Loading...</div>
              ) : error ? (
                <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                  {error}
                </div>
              ) : recent.length === 0 ? (
                <EmptyState />
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
                    {recent.map((r) => (
                      <TableRow key={r.id}>
                        <TableCell className="font-medium text-slate-900">{r.original_filename}</TableCell>
                        <TableCell>
                          <Badge variant={r.prediction_label === "FAKE" ? "fake" : "real"}>
                            {r.prediction_label}
                          </Badge>
                        </TableCell>
                        <TableCell>{Number(r.confidence).toFixed(2)}%</TableCell>
                        <TableCell className="text-slate-600">{new Date(r.created_at).toLocaleString()}</TableCell>
                        <TableCell className="text-right">
                          <Link href={`/history/${r.id}`}>
                            <Button size="sm" variant="outline">
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

          <Card>
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Link href="/analysis">
                <Button className="w-full justify-between" variant="default">
                  Upload Image <ImageUp className="h-4 w-4" />
                </Button>
              </Link>
              <Link href="/history">
                <Button className="w-full justify-between" variant="secondary">
                  View History <ArrowUpRight className="h-4 w-4" />
                </Button>
              </Link>
              <Link href="/heatmaps">
                <Button className="w-full justify-between" variant="outline">
                  Open Heatmaps <Flame className="h-4 w-4" />
                </Button>
              </Link>
              <div className="mt-6 rounded-xl border border-slate-200 bg-slate-50 p-4">
                <div className="text-sm font-semibold text-slate-900">Milestone Roadmap</div>
                <div className="mt-3 space-y-2 text-sm">
                  <RoadmapItem label="Image Analysis" badge="Active" variant="default" />
                  <RoadmapItem label="Audio Analysis" badge="Coming Soon" variant="soon" />
                  <RoadmapItem label="Video Analysis" badge="Coming Soon" variant="soon" />
                  <RoadmapItem label="Multimodal Fusion" badge="Coming Soon" variant="soon" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

function RoadmapItem({
  label,
  badge,
  variant
}: {
  label: string
  badge: string
  variant: "default" | "soon"
}) {
  return (
    <div className="flex items-center justify-between">
      <div className="text-slate-700">{label}</div>
      <Badge variant={variant === "soon" ? "soon" : "default"}>{badge}</Badge>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="grid h-full place-items-center">
      <div className="text-center">
        <div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-slate-100 text-slate-600">
          <SearchX className="h-5 w-5" />
        </div>
        <div className="mt-3 text-sm font-medium text-slate-900">No data yet</div>
        <div className="mt-1 text-sm text-slate-600">Run an image analysis to populate dashboard insights.</div>
        <div className="mt-4">
          <Link href="/analysis">
            <Button>
              Start Analysis <ArrowUpRight className="ml-2 h-4 w-4" />
            </Button>
          </Link>
        </div>
      </div>
    </div>
  )
}

