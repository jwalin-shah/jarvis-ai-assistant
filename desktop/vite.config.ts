import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { resolve } from 'path';

// Check if running in Tauri context (TAURI_ENV_PLATFORM is set during tauri dev/build)
const isTauri = !!process.env.TAURI_ENV_PLATFORM;

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [svelte()],
  // Prevent vite from obscuring rust errors
  clearScreen: false,
  // Tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    watch: {
      // Tell vite to ignore watching `src-tauri`
      ignored: ['**/src-tauri/**'],
    },
  },
  resolve: {
    alias: isTauri
      ? {
          '@': resolve(__dirname, './src'),
        }
      : {
          '@': resolve(__dirname, './src'),
          // Stub Tauri APIs when not running in Tauri context (for E2E tests)
          '@tauri-apps/api/event': resolve(__dirname, 'tests/mocks/tauri-event.ts'),
          '@tauri-apps/api/core': resolve(__dirname, 'tests/mocks/tauri-core.ts'),
          '@tauri-apps/plugin-sql': resolve(__dirname, 'tests/mocks/tauri-sql.ts'),
        },
  },
  build: {
    target: 'esnext',
    minify: 'esbuild',
    esbuild: {
      drop: process.env.NODE_ENV === 'production' ? ['console', 'debugger'] : [],
    },
    rollupOptions: {
      output: {
        // Manual chunk splitting for better caching
        manualChunks: {
          // Vendor chunks - third party libraries
          'vendor-d3': ['d3'],
          // Feature chunks - large components that can be loaded separately
          'feature-graph': ['./src/lib/components/graph/RelationshipGraph.svelte'],
          'feature-dashboard': ['./src/lib/components/Dashboard.svelte'],
          'feature-settings': ['./src/lib/components/Settings.svelte'],
          'feature-templates': ['./src/lib/components/TemplateBuilder.svelte'],
          'feature-chat': ['./src/lib/components/ChatView.svelte'],
        },
        // Chunk naming strategy
        chunkFileNames: (chunkInfo) => {
          const prefix = chunkInfo.name?.startsWith('vendor-')
            ? 'vendor'
            : chunkInfo.name?.startsWith('feature-')
              ? 'feature'
              : 'chunk';
          return `assets/${prefix}-[name]-[hash].js`;
        },
        // Ensure CSS is extracted for better caching
        assetFileNames: (assetInfo) => {
          const info = assetInfo.name || '';
          if (info.endsWith('.css')) {
            return 'assets/styles-[hash][extname]';
          }
          return 'assets/[name]-[hash][extname]';
        },
      },
    },
    // Split CSS for better caching
    cssCodeSplit: true,
    // Report chunk sizes
    reportCompressedSize: true,
  },
});
