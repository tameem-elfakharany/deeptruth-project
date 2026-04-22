export const endpoints = {
  auth: {
    register: "/auth/register",
    login: "/auth/login",
    me: "/auth/me"
  },
  predictions: {
    predict: "/predict",
    predictWithHeatmap: "/predict-with-heatmap",
    predictVideo: "/predict-video",
    predictAudio: "/predict-audio",
    myHistory: "/predictions/me",
    recent: "/predictions/recent",
    byId: (id: number | string) => `/predictions/${id}`
  }
} as const

