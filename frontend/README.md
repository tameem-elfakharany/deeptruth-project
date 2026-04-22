# DeepTruth Frontend

Modern Next.js (TypeScript) frontend for the DeepTruth deepfake detection platform.

## Tech Stack

- Next.js (App Router) + TypeScript
- Tailwind CSS (white + #1d5c96 branding)
- shadcn/ui-style components (Button, Card, Input, Badge, Table)
- Recharts for dashboard charts
- React Hook Form for login/register forms
- Lucide icons

## Setup

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Configure backend base URL (optional):

Set `NEXT_PUBLIC_API_BASE_URL` (defaults to `http://localhost:8000`).

Example:

```bash
# macOS/Linux
export NEXT_PUBLIC_API_BASE_URL="http://localhost:8000"

# Windows PowerShell
$env:NEXT_PUBLIC_API_BASE_URL="http://localhost:8000"
```

3. Run the frontend:

```bash
npm run dev
```

Open:
- `http://localhost:3000`

## Pages

- `/login` Login
- `/register` Registration
- `/dashboard` Protected dashboard with multiple charts
- `/analysis` Image analysis (uploads to `/predict` or `/predict-with-heatmap`)
- `/history` User analysis history (expects user-history endpoint)
- `/history/[id]` Result details page
- `/heatmaps` Heatmap list (filters history for heatmap availability)
- `/heatmaps/[id]` Heatmap viewer (side-by-side layout, based on available API fields)
- `/audio` Coming Soon
- `/video` Coming Soon
- `/fusion` Coming Soon
- `/profile` User profile page

## Auth Flow

- Token is persisted in `localStorage` under `deeptruth_token`
- API calls automatically attach `Authorization: Bearer <token>`
- Protected routes redirect unauthenticated users to `/login`

Expected backend endpoints:
- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`

## Prediction + Data

Expected backend endpoints:
- `POST /predict`
- `POST /predict-with-heatmap`

History/data endpoints are configured in one place:
- `src/services/endpoints.ts`

If your backend uses different routes, update:
- `endpoints.predictions.myHistory`
- `endpoints.predictions.recent`
- `endpoints.predictions.byId`

## Notes on Heatmaps

The frontend uses `heatmap_path` returned by the API.

If your backend returns a filesystem path, configure your backend to serve that path (static files) or return a URL. The frontend resolves relative paths against `NEXT_PUBLIC_API_BASE_URL`.

