<script setup>
import { RouterLink, RouterView } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useRouter } from 'vue-router'

const authStore = useAuthStore()
const router = useRouter()

// 点击设置图标的跳转逻辑
const goToSettings = () => {
  if (authStore.isAuthenticated) {
    router.push('/settings')
  } else {
    router.push('/login')
  }
}
</script>

<template>
  <div id="app-layout">
    <header>
      <nav>
        <!-- 左侧导航 -->
        <div class="nav-left">
          <RouterLink to="/">首页</RouterLink>
          <RouterLink to="/device-management">器件管理</RouterLink>
          <RouterLink to="/device-analysis">器件分析</RouterLink>
          <RouterLink to="/device-comparison">器件对比</RouterLink>
          <RouterLink to="/device-parsing">器件解析</RouterLink>
          <RouterLink to="/rag">RAG检索</RouterLink>
          <RouterLink to="/assessment">评估模块</RouterLink>
          <RouterLink to="/failure-probability">失效概率模块</RouterLink>
        </div>

        <!-- 右侧认证与设置 -->
        <div class="nav-right">
          <template v-if="!authStore.isAuthenticated">
            <RouterLink to="/login">登录</RouterLink>
            <RouterLink to="/register">注册</RouterLink>
          </template>
          <template v-else>
            <a @click="authStore.logout()" href="#">退出登录</a>
          </template>
          <!-- 设置图标 -->
          <a @click="goToSettings" class="settings-icon" title="模型设置">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
          </a>
        </div>
      </nav>
    </header>

    <main>
      <RouterView />
    </main>
  </div>
</template>

<style scoped>
/* 整体布局 */
#app-layout {
  display: flex;
  flex-direction: column;
  height: 100vh;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  overflow: hidden;
}
/* 头部和导航栏 */
header {
  background-color: #2c3e50;
  color: white;
  padding: 1rem 2rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  flex-shrink: 0;
}
/* 导航栏容器 */
nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.nav-left, .nav-right {
  display: flex;
  align-items: center;
  gap: 1.5rem;
}
/* 导航链接通用样式 */
nav a {
  color: #bdc3c7;
  text-decoration: none;
  font-weight: bold;
  padding: 0.5rem;
  border-radius: 5px;
  transition: all 0.3s ease;
  position: relative;
}
nav a:hover {
  color: #ffffff;
}
.nav-right a {
  cursor: pointer;
}
/* 激活链接的发光效果 */
nav a.router-link-exact-active {
  color: #42b983;
  text-shadow: 0 0 10px rgba(66, 185, 131, 0.7);
}

/* 新增：设置图标样式 */
.settings-icon {
  display: flex;
  align-items: center;
  justify-content: center;
}
.settings-icon svg {
  color: #bdc3c7;
  transition: all 0.3s ease;
}
.settings-icon:hover svg {
  color: #ffffff;
}

/* 主内容区域 */
main {
  flex-grow: 1;
  padding: 20px;
  overflow-y: auto;
  background-color: #f4f6f9;
  height: 0;
}
</style>

