import axios from 'axios';
import { useAuthStore } from '@/stores/auth';
import { ElMessage } from 'element-plus';

// (*** 关键修复：将回退值也改为相对路径 ***)
// 这确保了在 .env 文件加载失败时，它仍然会尝试使用 Vite 代理
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/';

// 1. 创建并配置一个 Axios 实例
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
    'ngrok-skip-browser-warning': 'true'
  },
});

// 2. 添加请求拦截器 (Request Interceptor) - 无需修改
apiClient.interceptors.request.use(
  (config) => {
    const authStore = useAuthStore();
    // 如果用户已登录，则在每个请求的头部自动附加 Token
    if (authStore.isAuthenticated) {
      config.headers['Authorization'] = `Bearer ${authStore.token}`;
      // *** 重要：如果是 FormData 请求，让浏览器自动设置 Content-Type ***
      if (config.data instanceof FormData) {
        // 删除我们之前默认设置的 'application/json'，否则上传会失败
        delete config.headers['Content-Type'];
      }
    }
    return config;
  },
  (error) => {
    // 对请求错误做些什么
    return Promise.reject(error);
  }
);

// 3. 添加响应拦截器 (Response Interceptor) - 无需修改
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response) {
        if (error.response.status === 401) {
            const authStore = useAuthStore();
            authStore.logout();
            ElMessage.error('您的登录已过期，请重新登录。');
        }
    }
    return Promise.reject(error);
  }
);


// 4. 定义并导出所有 API 调用函数 (保留原有，新增 Experiment 相关)
export default {
    // --- 器件相关 (Device) ---
    getDevices(params) {
        return apiClient.get('devices/', { params });
    },
    getDeviceById(id) {
        // 这个接口现在会返回嵌套的 experiments 数据
        return apiClient.get(`devices/${id}/`);
    },
    createDevice(data) {
        // 创建器件基础信息
        return apiClient.post('devices/', data);
    },
    updateDevice(id, data) {
        // 更新器件基础信息
        return apiClient.patch(`devices/${id}/`, data);
    },
    deleteDevice(id) {
        // 删除器件及其所有关联数据 (experiments, parameters, datapoints)
        return apiClient.delete(`devices/${id}/`);
    },
    compareDevices(data) {
        // 器件对比接口 (后端逻辑已更新)
        return apiClient.post('compare/', data);
    },

    // --- 实验相关 (Experiment) ---
    getExperiment(id) {
        // 获取单个实验的详细信息（包括 parameters 和 datapoints）
        return apiClient.get(`experiments/${id}/`);
    },
    createExperiment(data) {
        // 创建一个新的实验记录 (基础信息)
        // 'data' 应包含 { device: deviceId, name: '...', experiment_type: '...' }
        return apiClient.post('experiments/', data);
    },
    updateExperiment(id, data) {
        // 更新实验的基础信息 (name, experiment_type)
        return apiClient.patch(`experiments/${id}/`, data);
    },
    deleteExperiment(id) {
        // 删除一个实验及其关联的 parameters 和 datapoints
        return apiClient.delete(`experiments/${id}/`);
    },
    updateExperimentGridData(id, data) {
        // 更新实验的表格数据 (覆盖式)
        // 'data' 应为 { parameters: [...], datapoints: [...] }
        return apiClient.patch(`experiments/${id}/grid_data/`, data);
    },
    updateExperimentCsvMetadata(id, data) {
        // 更新实验关联的 CSV 元数据
        // 'data' 应为 { csv_files_metadata: [...] }
        return apiClient.patch(`experiments/${id}/csv_metadata/`, data);
    },
// --- !!! 新增：上传 CSV 文件 !!! ---
    uploadExperimentCsv(experimentId, formData) {
      // 移除开头的 / ，使其与 getExperiment 等函数保持一致
      return apiClient.post(`experiments/${experimentId}/upload_csv/`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000
      });
    },
    // --- 获取 CSV 文件内容 ---
    getExperimentCsvData(experimentId, metadataId) {
      // 移除开头的 /
      return apiClient.get(`experiments/${experimentId}/csv_data/${metadataId}/`, {
        timeout: 60000
      });
    },
    // --- (可选) 通过 Device 端点添加 Experiment ---
    addExperimentToDevice(deviceId, data) {
        // 'data' 应包含 { name: '...', experiment_type: '...' }
        // 注意：如果使用此方法创建，后续仍需调用 updateExperimentGridData/CsvMetadata 填充数据
        return apiClient.post(`devices/${deviceId}/add_experiment/`, data);
    },

    // --- 认证相关 (保持不变) ---
    login(username, password) {
        return apiClient.post('token/', { username, password });
    },
    register(username, password) {
        return apiClient.post('register/', { username, password });
    },
    getProfile() {
        return apiClient.get('profile/');
    },
    updateProfile(data) {
        return apiClient.patch('profile/', data);
    },

    // --- 评估与计算 (保持不变) ---
    assessDamage(data) {
        return apiClient.post('assess/damage/', data);
    },
    assessLink(data) {
        return apiClient.post('assess/link/', data);
    },
    calculateFailureProbability(data) {
        return apiClient.post('probability/calculate/', data);
    },

    // --- 失效概率数据集 (保持不变) ---
    getProbabilityDatasets() {
        return apiClient.get('probability-datasets/');
    },
    createProbabilityDataset(data) {
        return apiClient.post('probability-datasets/', data);
    },

    uploadPdf(formData) {
        // POST /api/pdf_parser/tasks/
        // formData 应包含 'pdf_file'
        // 拦截器会自动处理 FormData 的 Content-Type
        return apiClient.post('pdf_parser/tasks/', formData);
    },
    getParsingTasks() {
        // GET /api/pdf_parser/tasks/
        return apiClient.get('pdf_parser/tasks/');
    },
    getTaskStatus(taskId) {
        // GET /api/pdf_parser/tasks/<id>/
        return apiClient.get(`pdf_parser/tasks/${taskId}/`);
    },
    deleteParsingTask(taskId) {
        // DELETE /api/pdf_parser/tasks/<id>/
        return apiClient.delete(`pdf_parser/tasks/${taskId}/`);
    },
    getTaskGraphData(taskId) {
      return apiClient.get(`pdf_parser/tasks/${taskId}/graph/`);
    },
};
export const getRawMediaBlob = async (fullUrl) => {
  const authStore = useAuthStore();
  const headers = {
    'ngrok-skip-browser-warning': 'true',
  };
  if (authStore.isAuthenticated) {
    headers['Authorization'] = `Bearer ${authStore.token}`;
  }

  try {
    const response = await fetch(fullUrl, {
      method: 'GET',
      headers: headers,
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
    }

    return await response.blob();
  } catch (error) {
    console.error(`Error fetching raw media blob from ${fullUrl}:`, error);
    return null;
  }
};
