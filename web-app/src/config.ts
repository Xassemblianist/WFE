// WFE API tabanı. Geliştirmede varsayılan olarak yerel FastAPI (:8000).
// .env üzerinden geçersiz kılınabilir:  VITE_API_BASE=https://api.ornek.com
export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '') ||
  'http://localhost:8000'

export const apiUrl = (path: string) => `${API_BASE}${path}`
