import { getStoredToken } from "@/lib/token"

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "") || "http://127.0.0.1:8000"

type ApiOptions = RequestInit & { auth?: boolean }

export async function apiFetch<T>(path: string, options: ApiOptions = {}): Promise<T> {
  const cleanPath = path.startsWith("/") ? path.substring(1) : path
  const url = `${API_BASE_URL}/${cleanPath}`
  const headers = new Headers(options.headers)

  if (!headers.has("Accept")) headers.set("Accept", "application/json")

  const shouldAuth = options.auth !== false
  if (shouldAuth) {
    const token = getStoredToken()
    if (token) {
      headers.set("Authorization", `Bearer ${token}`)
    } else {
      console.warn(`Auth required for ${path} but no token found.`)
    }
  }

  console.log(`Fetching: ${options.method || "GET"} ${url}`)
  const res = await fetch(url, { ...options, headers }).catch((err) => {
    console.error("Fetch error for URL:", url, err)
    throw new Error(`Connection to backend failed at ${url}. Please ensure the server is running and CORS is allowed.`)
  })

  console.log(`Response: ${res.status} ${res.statusText}`)
  const contentType = res.headers.get("content-type") || ""
  const isJson = contentType.includes("application/json")

  if (!res.ok) {
    console.error(`Request failed with status ${res.status}`)
    const detail = isJson ? await res.json().catch(() => null) : await res.text().catch(() => "")
    console.error("Error detail:", detail)
    const message =
      typeof detail === "string"
        ? detail
        : detail?.detail || detail?.message || `Request failed (${res.status})`
    throw new Error(message)
  }

  if (!isJson) {
    return (await res.text()) as unknown as T
  }

  return (await res.json()) as T
}

