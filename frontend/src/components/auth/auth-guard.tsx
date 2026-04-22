"use client"

import { useEffect } from "react"
import { usePathname, useRouter } from "next/navigation"

import { useAuth } from "@/lib/auth"

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const pathname = usePathname()
  const { token, user, isLoading } = useAuth()

  useEffect(() => {
    if (isLoading) return
    if (!token) {
      router.replace("/login")
      return
    }
    if (!user) {
      router.replace("/login")
      return
    }
    if (pathname === "/login" || pathname === "/register") {
      router.replace("/dashboard")
    }
  }, [isLoading, pathname, router, token, user])

  if (isLoading || !token || !user) return null
  return <>{children}</>
}

