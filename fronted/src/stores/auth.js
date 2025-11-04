import { defineStore } from 'pinia'
import router from '@/router'
import apiService from '@/services/apiService'; // 1. 移除 axios, 导入 apiService

export const useAuthStore = defineStore('auth', {
  state: () => ({
    token: localStorage.getItem('token') || null,
    user: null,
    profile: null
  }),
  getters: {
    isAuthenticated: (state) => !!state.token,
    authHeaders: (state) => {
      return { Authorization: `Bearer ${state.token}` };
    },
    activeConfig(state) {
        if (!state.profile) {
            return null;
        }
        if (state.profile.active_config_id === 'default') {
            return {
                id: 'default',
                name: '系统默认配置',
                ...state.profile.default_config
            };
        }
        return state.profile.user_configs.find(
            config => config.id === state.profile.active_config_id
        ) || null;
    }
  },
  actions: {
    // 2. 重构所有 actions, 使用 apiService
    async fetchProfile() {
      if (!this.token) return;
      try {
        const response = await apiService.getProfile();
        this.profile = response.data;
      } catch (error) {
        console.error('获取用户配置失败:', error);
        // 如果获取失败（例如token过期），则强制登出
        this.logout();
      }
    },
    async updateProfile(profileData) {
        if (!this.token) return false;
        try {
            const response = await apiService.updateProfile(profileData);
            // 更新本地 state
            if(response.data.user_configs) {
                this.profile.user_configs = response.data.user_configs;
            }
            if(response.data.active_config_id) {
                this.profile.active_config_id = response.data.active_config_id;
            }
            return true;
        } catch (error) {
            console.error('更新用户配置失败:', error);
            return false;
        }
    },
    async login(username, password) {
      try {
        const response = await apiService.login(username, password);
        this.token = response.data.access
        localStorage.setItem('token', this.token)
        // 登录成功后，立即获取用户的 Profile
        // 注意：此时 apiService 的拦截器会自动附加新的 token
        await this.fetchProfile();
        await router.push('/')
      } catch (error) {
        console.error('登录失败:', error)
        alert('用户名或密码错误！')
      }
    },
    async register(username, password) {
      try {
        await apiService.register(username, password);
        await router.push('/login')
        alert('注册成功，请登录！');
      } catch (error) {
        console.error('注册失败:', error)
        alert('注册失败，用户名可能已存在！')
      }
    },
    logout() {
      this.token = null
      this.user = null
      this.profile = null;
      localStorage.removeItem('token')
      router.push('/login')
    }
  }
})
