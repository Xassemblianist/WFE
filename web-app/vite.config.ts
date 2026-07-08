import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Geliştirmede API çağrıları doğrudan http://localhost:8000'e gider (bkz. src/config.ts).
// İstenirse aşağıdaki proxy ile aynı-köken üzerinden de geçirilebilir.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
})
