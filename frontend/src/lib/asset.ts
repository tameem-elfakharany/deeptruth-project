import { API_BASE_URL } from "@/services/api"

export function resolveBackendPath(path: string) {
  const trimmed = path.trim()
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) return trimmed
  if (trimmed.startsWith("/")) return `${API_BASE_URL}${trimmed}`
  return `${API_BASE_URL}/${trimmed}`
}

