"use client"

import { ShieldCheck } from "lucide-react"

import { useAuth } from "@/lib/auth"

export function Topbar({ title, subtitle }: { title: string; subtitle?: string }) {
  const { user } = useAuth()
  return (
    <div className="flex items-start justify-between gap-4 border-b border-slate-200 bg-white px-6 py-5">
      <div>
        <div className="text-xl font-semibold text-slate-900">{title}</div>
        {subtitle ? <div className="mt-1 text-sm text-slate-600">{subtitle}</div> : null}
      </div>
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-sm text-slate-700">
          <ShieldCheck className="h-4 w-4 text-brand-600" />
          <span className="max-w-[220px] truncate">{user?.full_name || user?.email}</span>
        </div>
      </div>
    </div>
  )
}

