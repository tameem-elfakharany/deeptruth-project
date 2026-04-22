import type { Config } from "tailwindcss"

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f0f7ff",
          100: "#dfeeff",
          200: "#b9dcff",
          300: "#86c3ff",
          400: "#4fa3ff",
          500: "#1d5c96",
          600: "#174a79",
          700: "#123a5f",
          800: "#0d2a45",
          900: "#071a2d"
        }
      },
      boxShadow: {
        soft: "0 1px 2px rgba(16,24,40,0.05), 0 8px 24px rgba(16,24,40,0.08)"
      }
    }
  },
  plugins: []
}

export default config

