"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function ComingSoon({
  title,
  milestone,
  description,
  planned
}: {
  title: string
  milestone: string
  description: string
  planned: string[]
}) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0">
          <CardTitle>{title}</CardTitle>
          <Badge variant="soon">Coming Soon</Badge>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-slate-700">
          <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3">
            <div className="text-slate-600">Milestone</div>
            <div className="font-medium text-slate-900">{milestone}</div>
          </div>
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <div className="text-sm font-semibold text-slate-900">Overview</div>
            <div className="mt-1 text-sm text-slate-700">{description}</div>
          </div>
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <div className="text-sm font-semibold text-slate-900">Planned features</div>
            <ul className="mt-2 space-y-2 text-sm text-slate-700">
              {planned.map((p) => (
                <li key={p} className="flex items-start gap-2">
                  <span className="mt-1 h-1.5 w-1.5 rounded-full bg-brand-500" />
                  <span>{p}</span>
                </li>
              ))}
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

