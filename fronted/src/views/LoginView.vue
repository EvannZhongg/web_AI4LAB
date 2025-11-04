<template>
  <div class="auth-form">
    <h2>登录</h2>
    <form @submit.prevent="handleLogin">
      <input type="text" v-model="username" placeholder="用户名" required />
      <input type="password" v-model="password" placeholder="密码" required />
      <button type="submit">登录</button>
    </form>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useAuthStore } from '@/stores/auth'

const username = ref('')
const password = ref('')
const authStore = useAuthStore()

const handleLogin = () => {
  authStore.login(username.value, password.value)
}
</script>

<style scoped>
/* (样式见下方统一提供) */
.auth-form { max-width: 400px; margin: 50px auto; padding: 2rem; border: 1px solid #ccc; border-radius: 8px; }
.auth-form h2 { text-align: center; }
.auth-form input { display: block; width: 100%; padding: 0.8rem; margin-bottom: 1rem; box-sizing: border-box; }
.auth-form button { width: 100%; padding: 0.8rem; background-color: #42b983; color: white; border: none; cursor: pointer; }
</style>
