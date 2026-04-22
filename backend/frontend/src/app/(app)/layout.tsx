import { AuthGuard } from "@/components/auth/auth-guard"
import { Sidebar } from "@/components/layout/sidebar"

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthGuard>
      <div className="min-h-screen bg-slate-50">
        <div className="mx-auto flex min-h-screen w-full max-w-[1440px]">
          <Sidebar />
          <main className="flex-1">{children}</main>
        </div>
      </div>
    </AuthGuard>
  )
}

