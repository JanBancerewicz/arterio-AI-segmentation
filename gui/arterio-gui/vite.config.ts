import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

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
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    css: {modules: {classNameStrategy: 'non-scoped'}},
    coverage: {
      provider: 'v8',
      include: ['src/**/*.{ts,tsx}'],
      exclude: ['src/**/*.test.{ts,tsx}', 'src/test/**', 'src/mocks/**', 'src/main.tsx'],
      reporter: ['text', 'html'],
    },
  },
})
