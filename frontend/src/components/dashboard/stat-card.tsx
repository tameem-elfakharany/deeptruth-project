"use client"

import { cn } from "@/lib/utils"
import { Card, CardContent } from "@/components/ui/card"

export function StatCard({
  label,
  value,
  icon,
  tone = "default"
}: {
  label: string
  value: string
  icon: React.ReactNode
  tone?: "default" | "good" | "bad"
}) {
  const toneClass =
    tone === "good"
      ? "bg-emerald-50 text-emerald-700"
      : tone === "bad"
        ? "bg-red-50 text-red-700"
        : "bg-brand-50 text-brand-700"

  return (
    <Card>
      <CardContent className="flex items-center justify-between p-6">
        <div>
          <div className="text-sm text-slate-600">{label}</div>
          <div className="mt-1 text-2xl font-semibold text-slate-900">{value}</div>
        </div>
        <div className={cn("grid h-11 w-11 place-items-center rounded-xl", toneClass)}>{icon}</div>
      </CardContent>
    </Card>
  )
}

