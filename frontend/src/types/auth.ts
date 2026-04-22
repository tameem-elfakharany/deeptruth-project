export type User = {
  id: number
  full_name?: string | null
  email: string
  created_at?: string
}

export type AuthLoginResponse =
  | { access_token: string; token_type?: string }
  | { token: string; token_type?: string }

