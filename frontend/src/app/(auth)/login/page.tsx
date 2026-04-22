"use client"

import Link from "next/link"
import { useForm } from "react-hook-form"
import { Eye, EyeOff, Lock, Mail } from "lucide-react"
import { useState } from "react"

import { useAuth } from "@/lib/auth"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

type FormValues = {
  email: string
  password: string
}

export default function LoginPage() {
  const { login } = useAuth()
  const [error, setError] = useState<string | null>(null)
  const [showPassword, setShowPassword] = useState(false)
  const {
    register,
    handleSubmit,
    formState: { isSubmitting, errors }
  } = useForm<FormValues>({ defaultValues: { email: "", password: "" } })

  const onSubmit = handleSubmit(async (values) => {
    setError(null)
    try {
      await login(values.email, values.password)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Login failed.")
    }
  })

  return (
    <div className="grid min-h-screen grid-cols-1 lg:grid-cols-2">
      <div className="relative hidden overflow-hidden bg-brand-500 lg:block">
        <div className="absolute inset-0 opacity-20">
          <div className="absolute -left-24 top-10 h-72 w-72 rounded-full bg-white" />
          <div className="absolute left-48 top-60 h-80 w-80 rounded-full bg-white" />
          <div className="absolute -right-24 bottom-10 h-72 w-72 rounded-full bg-white" />
        </div>
        <div className="relative flex h-full flex-col justify-between p-10 text-white">
          <div className="flex items-center gap-3">
            <div className="grid h-11 w-11 place-items-center rounded-xl bg-white/15">
              <span className="text-sm font-semibold">DT</span>
            </div>
            <div>
              <div className="text-lg font-semibold">DeepTruth</div>
              <div className="text-sm text-white/80">AI Forensics & Explainability</div>
            </div>
          </div>
          <div className="max-w-md">
            <div className="text-3xl font-semibold leading-tight">Detect deepfakes with confidence.</div>
            <div className="mt-4 text-sm text-white/80">
              A clean, data-driven console for image authenticity analysis, explainability heatmaps, and user-specific
              history.
            </div>
          </div>
          <div className="text-xs text-white/70">For research and decision support. Not proof.</div>
        </div>
      </div>

      <div className="flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-md">
          <Card>
            <CardHeader>
              <CardTitle>Welcome back</CardTitle>
              <CardDescription>Sign in to access your dashboard and analysis history.</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={onSubmit} className="space-y-5">
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
                    <Input
                      id="email"
                      type="email"
                      placeholder="you@example.com"
                      className="pl-9"
                      {...register("email", { required: "Email is required." })}
                    />
                  </div>
                  {errors.email ? <div className="text-xs text-red-600">{errors.email.message}</div> : null}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="password">Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
                    <Input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="••••••••"
                      className="pl-9 pr-10"
                      {...register("password", { required: "Password is required." })}
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword((v) => !v)}
                      className="absolute right-2 top-2 grid h-8 w-8 place-items-center rounded-md text-slate-500 hover:bg-slate-100"
                      aria-label={showPassword ? "Hide password" : "Show password"}
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                  {errors.password ? <div className="text-xs text-red-600">{errors.password.message}</div> : null}
                </div>

                {error ? (
                  <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                    {error}
                  </div>
                ) : null}

                <Button type="submit" className="w-full" disabled={isSubmitting}>
                  {isSubmitting ? "Signing in..." : "Sign in"}
                </Button>

                <div className="text-center text-sm text-slate-600">
                  No account?{" "}
                  <Link href="/register" className="font-medium text-brand-600 hover:text-brand-700">
                    Create one
                  </Link>
                </div>
              </form>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

