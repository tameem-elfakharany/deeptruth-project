"use client"

import { useEffect, useMemo, useState } from "react"
import { Mail, User2 } from "lucide-react"

import { useAuth } from "@/lib/auth"
import type { PredictionRecord } from "@/types/prediction"
import { getMyPredictionHistory } from "@/services/predictions"
import { summarize } from "@/lib/analytics"
import { Topbar } from "@/components/layout/topbar"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { StatCard } from "@/components/dashboard/stat-card"

export default function ProfilePage() {
  const { user } = useAuth()
  const [records, setRecords] = useState<PredictionRecord[]>([])

  useEffect(() => {
    getMyPredictionHistory().then(setRecords).catch(() => setRecords([]))
  }, [])

  const summary = useMemo(() => summarize(records), [records])

  return (
    <div>
      <Topbar title="Profile" subtitle="Account overview and usage summary." />
      <div className="space-y-6 p-6">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
          <StatCard label="Total Analyses" value={String(summary.total)} icon={<User2 className="h-5 w-5" />} />
          <StatCard label="Fake Detected" value={String(summary.fake)} tone="bad" icon={<User2 className="h-5 w-5" />} />
          <StatCard label="Real Detected" value={String(summary.real)} tone="good" icon={<User2 className="h-5 w-5" />} />
          <StatCard
            label="Average Confidence"
            value={`${summary.avgConfidence.toFixed(2)}%`}
            icon={<User2 className="h-5 w-5" />}
          />
        </div>

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
          <Card className="xl:col-span-2">
            <CardHeader>
              <CardTitle>Account</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3">
                <div className="flex items-center gap-2 text-slate-600">
                  <User2 className="h-4 w-4" />
                  Full name
                </div>
                <div className="font-medium text-slate-900">{user?.full_name || "—"}</div>
              </div>
              <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3">
                <div className="flex items-center gap-2 text-slate-600">
                  <Mail className="h-4 w-4" />
                  Email
                </div>
                <div className="font-medium text-slate-900">{user?.email}</div>
              </div>
              <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3">
                <div className="text-slate-600">Created</div>
                <div className="font-medium text-slate-900">
                  {user?.created_at ? new Date(user.created_at).toLocaleString() : "—"}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Dashboard readiness</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-slate-700">
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                <div className="text-sm font-semibold text-slate-900">User-linked history</div>
                <div className="mt-1 text-sm text-slate-600">
                  Predictions made while logged in can be associated with your user account for dashboard features.
                </div>
              </div>
              <div className="rounded-xl border border-slate-200 bg-white p-4">
                <div className="text-sm font-semibold text-slate-900">Explainability</div>
                <div className="mt-1 text-sm text-slate-600">
                  Heatmap availability: {summary.withHeatmap}/{summary.total}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

