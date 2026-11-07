import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const proxy = {
  '/api': {
    target: 'http://127.0.0.1:8000',
    changeOrigin: false,
  },
};

export default defineConfig({
  plugins: [react()],
  server: {host: 'localhost', port: 5173, strictPort: true, proxy},
  preview: {host: 'localhost', port: 5173, strictPort: true, proxy},
})
