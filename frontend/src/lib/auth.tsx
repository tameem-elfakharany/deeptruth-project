"use client"

import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react"
import { usePathname, useRouter } from "next/navigation"

import type { AuthLoginResponse, User } from "@/types/auth"
import { apiFetch } from "@/services/api"
import { clearStoredToken, getStoredToken, setStoredToken } from "@/lib/token"

type AuthContextValue = {
  token: string | null
  user: User | null
  isLoading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (fullName: string, email: string, password: string) => Promise<void>
  refreshMe: () => Promise<User | null>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const pathname = usePathname()
  const [token, setToken] = useState<string | null>(null)
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const refreshMe = useCallback(async () => {
    const existingToken = getStoredToken()
    if (!existingToken) {
      console.log("No stored token found in refreshMe")
      setUser(null)
      return null
    }
    console.log("Found token, calling /auth/me...")
    try {
      const me = await apiFetch<User>("/auth/me", { method: "GET" })
      console.log("RefreshMe successful:", me)
      setUser(me)
      return me
    } catch (err) {
      console.error("RefreshMe failed:", err)
      clearStoredToken()
      setToken(null)
      setUser(null)
      return null
    }
  }, [])

  useEffect(() => {
    const existing = getStoredToken()
    setToken(existing)
    refreshMe().finally(() => setIsLoading(false))
  }, [refreshMe])

  const login = useCallback(
    async (email: string, password: string) => {
      console.log("Starting login for:", email)
      const body = JSON.stringify({ email, password })
      try {
        const data = await apiFetch<AuthLoginResponse>("/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body,
          auth: false
        })
        console.log("Login API response:", data)
        const accessToken = "access_token" in data ? data.access_token : (data as any).token
        if (!accessToken) {
          throw new Error("No access token received from server.")
        }
        setStoredToken(accessToken)
        setToken(accessToken)
        console.log("Token stored, refreshing user info...")
        await refreshMe()
        console.log("User refreshed, redirecting to dashboard...")
        if (pathname?.startsWith("/login") || pathname?.startsWith("/register")) {
          router.replace("/dashboard")
        }
      } catch (err) {
        console.error("Login process failed:", err)
        throw err
      }
    },
    [pathname, refreshMe, router]
  )

  const register = useCallback(
    async (fullName: string, email: string, password: string) => {
      await apiFetch("/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ full_name: fullName, email, password }),
        auth: false
      })
      await login(email, password)
    },
    [login]
  )

  const logout = useCallback(() => {
    clearStoredToken()
    setToken(null)
    setUser(null)
    router.replace("/login")
  }, [router])

  const value = useMemo<AuthContextValue>(
    () => ({
      token,
      user,
      isLoading,
      login,
      register,
      refreshMe,
      logout
    }),
    [token, user, isLoading, login, register, refreshMe, logout]
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used within AuthProvider")
  return ctx
}

