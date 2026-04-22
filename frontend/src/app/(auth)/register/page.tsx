"use client"

import Link from "next/link"
import { useForm } from "react-hook-form"
import { Eye, EyeOff, Lock, Mail, User2 } from "lucide-react"
import { useMemo, useState } from "react"

import { useAuth } from "@/lib/auth"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

type FormValues = {
  fullName: string
  email: string
  password: string
  confirmPassword: string
}

export default function RegisterPage() {
  const { register: registerUser } = useAuth()
  const [error, setError] = useState<string | null>(null)
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const {
    register,
    handleSubmit,
    watch,
    formState: { isSubmitting, errors }
  } = useForm<FormValues>({
    defaultValues: { fullName: "", email: "", password: "", confirmPassword: "" }
  })

  const password = watch("password")
  const passwordHint = useMemo(() => {
    if (!password) return null
    if (password.length < 8) return "Use at least 8 characters."
    return null
  }, [password])

  const onSubmit = handleSubmit(async (values) => {
    setError(null)
    if (values.password !== values.confirmPassword) {
      setError("Passwords do not match.")
      return
    }
    try {
      await registerUser(values.fullName, values.email, values.password)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Registration failed.")
    }
  })

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 px-6 py-12">
      <div className="w-full max-w-md">
        <div className="mb-6 text-center">
          <div className="mx-auto mb-3 grid h-12 w-12 place-items-center rounded-2xl bg-brand-500 text-white shadow-soft">
            <span className="text-sm font-semibold">DT</span>
          </div>
          <div className="text-2xl font-semibold text-slate-900">Create your account</div>
          <div className="mt-1 text-sm text-slate-600">Access a premium, data-driven deepfake forensics dashboard.</div>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Register</CardTitle>
            <CardDescription>Set up your profile to save predictions to your account.</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={onSubmit} className="space-y-5">
              <div className="space-y-2">
                <Label htmlFor="fullName">Full name</Label>
                <div className="relative">
                  <User2 className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
                  <Input
                    id="fullName"
                    placeholder="Your name"
                    className="pl-9"
                    {...register("fullName", { required: "Full name is required." })}
                  />
                </div>
                {errors.fullName ? <div className="text-xs text-red-600">{errors.fullName.message}</div> : null}
              </div>

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
                    {...register("password", { required: "Password is required.", minLength: 8 })}
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
                {passwordHint ? <div className="text-xs text-slate-500">{passwordHint}</div> : null}
              </div>

              <div className="space-y-2">
                <Label htmlFor="confirmPassword">Confirm password</Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
                  <Input
                    id="confirmPassword"
                    type={showConfirm ? "text" : "password"}
                    placeholder="••••••••"
                    className="pl-9 pr-10"
                    {...register("confirmPassword", { required: "Confirm your password." })}
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirm((v) => !v)}
                    className="absolute right-2 top-2 grid h-8 w-8 place-items-center rounded-md text-slate-500 hover:bg-slate-100"
                    aria-label={showConfirm ? "Hide password" : "Show password"}
                  >
                    {showConfirm ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>

              {error ? (
                <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                  {error}
                </div>
              ) : null}

              <Button type="submit" className="w-full" disabled={isSubmitting}>
                {isSubmitting ? "Creating account..." : "Create account"}
              </Button>

              <div className="text-center text-sm text-slate-600">
                Already have an account?{" "}
                <Link href="/login" className="font-medium text-brand-600 hover:text-brand-700">
                  Sign in
                </Link>
              </div>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

