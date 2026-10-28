import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // `npm run dev`: the Flask backend (bash run.sh) handles /api on port 5001
  server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:5001' } },
});
