// fronted/vite.config.js

import { fileURLToPath, URL } from 'node:url'
import { defineConfig, loadEnv } from 'vite' // <--- (1) 导入 loadEnv
import vue from '@vitejs/plugin-vue'
import vueJsx from '@vitejs/plugin-vue-jsx'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => { // <--- (2) 将配置转换为函数

  // (3) 加载当前模式 (development/production) 的 .env 文件
  const env = loadEnv(mode, process.cwd(), '');
  const proxyTarget = env.VITE_PROXY_TARGET || 'http://127.0.0.1:8000';

  return {
    plugins: [
      vue(),
      vueJsx(),
    ],
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url))
      }
    },

    server: {
      proxy: {
        // (4) 代理 /api
        '/api': {
          target: proxyTarget, // <--- (5) 使用 .env 变量
          changeOrigin: true,
        },
        // (4) 代理 /media
        '/media': {
          target: proxyTarget, // <--- (5) 使用 .env 变量
          changeOrigin: true,
        }
      }
    }
  }
})
