"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  Activity,
  BarChart3,
  FileAudio2,
  FileImage,
  FileVideo2,
  Flame,
  History,
  LogOut,
  User2
} from "lucide-react"

import { cn } from "@/lib/utils"
import { useAuth } from "@/lib/auth"
import { Badge } from "@/components/ui/badge"

type NavItem = {
  href: string
  label: string
  icon: React.ComponentType<{ className?: string }>
  soon?: boolean
}

const navItems: NavItem[] = [
  { href: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { href: "/analysis", label: "Image Analysis", icon: FileImage },
  { href: "/history", label: "Analysis History", icon: History },
  { href: "/heatmaps", label: "Heatmap Viewer", icon: Flame, soon: true },
  { href: "/audio", label: "Audio Analysis", icon: FileAudio2 },
  { href: "/video", label: "Video Analysis", icon: FileVideo2 },
  { href: "/fusion", label: "Multimodal Fusion", icon: Activity, soon: true },
  { href: "/profile", label: "Profile", icon: User2 }
]

export function Sidebar() {
  const pathname = usePathname()
  const { logout } = useAuth()

  return (
    <aside className="hidden md:flex md:w-72 md:flex-col md:border-r md:border-slate-200 md:bg-white">
      <div className="flex h-16 items-center gap-3 px-6">
        <div className="grid h-10 w-10 place-items-center rounded-xl bg-brand-500 text-white shadow-soft">
          <span className="text-sm font-semibold">DT</span>
        </div>
        <div className="leading-tight">
          <div className="text-base font-semibold text-slate-900">DeepTruth</div>
          <div className="text-xs text-slate-500">AI Forensics Console</div>
        </div>
      </div>

      <nav className="flex-1 px-3 pb-6">
        <div className="mt-2 space-y-1">
          {navItems.map((item) => {
            const active = pathname === item.href || pathname?.startsWith(item.href + "/")
            const Icon = item.icon
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center justify-between rounded-lg px-3 py-2 text-sm transition-colors",
                  active
                    ? "bg-brand-50 text-brand-700"
                    : "text-slate-700 hover:bg-slate-50 hover:text-slate-900"
                )}
              >
                <span className="flex items-center gap-3">
                  <Icon className={cn("h-4 w-4", active ? "text-brand-700" : "text-slate-500")} />
                  {item.label}
                </span>
                {item.soon ? <Badge variant="soon">Coming Soon</Badge> : null}
              </Link>
            )
          })}
        </div>
      </nav>

      <div className="border-t border-slate-200 p-4">
        <button
          type="button"
          onClick={logout}
          className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm text-slate-700 hover:bg-slate-50 hover:text-slate-900"
        >
          <span className="flex items-center gap-3">
            <LogOut className="h-4 w-4 text-slate-500" />
            Logout
          </span>
        </button>
      </div>
    </aside>
  )
}

