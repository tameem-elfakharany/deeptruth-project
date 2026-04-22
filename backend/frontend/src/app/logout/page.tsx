"use client"

import { useEffect } from "react"

import { useAuth } from "@/lib/auth"

export default function LogoutPage() {
  const { logout } = useAuth()

  useEffect(() => {
    logout()
  }, [logout])

  return null
}

