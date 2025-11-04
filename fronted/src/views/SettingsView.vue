<script setup>
import { ref, onMounted, watch } from 'vue';
import { useAuthStore } from '@/stores/auth';
import { ElMessage, ElMessageBox } from 'element-plus';

const authStore = useAuthStore();

// --- 组件内部状态 ---
const loading = ref(true);
const activeConfigId = ref('');
const userConfigs = ref([]);

// --- 编辑对话框状态 ---
const dialogVisible = ref(false);
const isNewConfig = ref(false);
const currentConfig = ref({
  id: null, name: '', llm_api_url: '', llm_api_key: '', llm_model_name: ''
});

// --- 数据初始化与同步 ---
const syncFromStore = () => {
    if (authStore.profile) {
        activeConfigId.value = authStore.profile.active_config_id;
        userConfigs.value = JSON.parse(JSON.stringify(authStore.profile.user_configs));
        loading.value = false;
    }
};

watch(() => authStore.profile, (newProfile) => {
    if (newProfile) { syncFromStore(); }
}, { immediate: true });

onMounted(() => {
    if (!authStore.profile) {
        authStore.fetchProfile().then(() => { syncFromStore(); });
    } else {
        syncFromStore();
    }
});

// --- 事件处理函数 ---

// 新增：当用户切换激活配置时，立即自动保存
const handleActiveConfigChange = async (newId) => {
    const success = await authStore.updateProfile({ active_config_id: newId });
    if (success) {
        ElMessage.success('激活配置已更新！');
    } else {
        ElMessage.error('激活配置更新失败，请重试！');
        // 如果失败，则回滚UI上的选择
        activeConfigId.value = authStore.profile.active_config_id;
    }
};

const handleAdd = () => {
    isNewConfig.value = true;
    currentConfig.value = {
        id: Date.now(), name: '我的新配置', llm_api_url: '', llm_api_key: '', llm_model_name: ''
    };
    dialogVisible.value = true;
};

const handleEdit = (config) => {
    isNewConfig.value = false;
    currentConfig.value = JSON.parse(JSON.stringify(config));
    dialogVisible.value = true;
};

// 在对话框中保存（仅更新本地 userConfigs，然后触发一次总保存）
const handleSaveInDialog = async () => {
    if (!currentConfig.value.name) {
        ElMessage.warning('配置名称不能为空！');
        return;
    }

    if (isNewConfig.value) {
        userConfigs.value.push(currentConfig.value);
    } else {
        const index = userConfigs.value.findIndex(c => c.id === currentConfig.value.id);
        if (index !== -1) {
            userConfigs.value.splice(index, 1, currentConfig.value);
        }
    }
    dialogVisible.value = false;
    await saveUserConfigs(); // 保存整个列表
};

const handleDelete = (configId) => {
    ElMessageBox.confirm('确定要删除此配置吗？', '警告', { type: 'warning' })
    .then(async () => {
        const index = userConfigs.value.findIndex(c => c.id === configId);
        if (index !== -1) {
            userConfigs.value.splice(index, 1);
            // 如果删除的是当前激活的，则自动切换回默认并保存
            if (activeConfigId.value === configId) {
                activeConfigId.value = 'default';
                await saveAllSettings();
            } else {
                await saveUserConfigs();
            }
            ElMessage.success('配置已删除！');
        }
    }).catch(() => {});
};

// 仅保存用户配置列表
const saveUserConfigs = async () => {
    const success = await authStore.updateProfile({ user_configs: userConfigs.value });
    if (!success) { ElMessage.error('保存配置列表失败！'); }
    return success;
}

// 保存所有设置（主要用于删除激活项后的回滚）
const saveAllSettings = async () => {
    const success = await authStore.updateProfile({
        user_configs: userConfigs.value,
        active_config_id: activeConfigId.value
    });
     if (!success) { ElMessage.error('保存设置失败！'); }
}
</script>

<template>
  <div class="settings-page" v-loading="loading">
    <el-card shadow="never">
      <template #header>
        <div class="card-header">
          <strong>大模型API设置</strong>
          <!-- “保存全部”按钮已移除，因为激活选择是实时保存的 -->
        </div>
      </template>

      <div v-if="authStore.profile">
        <!-- 激活配置选择 -->
        <el-card shadow="inner" class="config-section">
          <template #header><strong>激活配置选择 (点击立即生效)</strong></template>
          <el-radio-group v-model="activeConfigId" @change="handleActiveConfigChange">
            <el-radio label="default" border>
              <strong>系统默认配置</strong>
              <div class="config-details">
                <span>URL: {{ authStore.profile.default_config.llm_api_url || '未设置' }}</span>
                <span>模型: {{ authStore.profile.default_config.llm_model_name || '未设置' }}</span>
              </div>
            </el-radio>
            <el-radio v-for="config in userConfigs" :key="config.id" :label="config.id" border>
              <strong>{{ config.name }}</strong>
              <div class="config-details">
                <span>URL: {{ config.llm_api_url || '未设置' }}</span>
                <span>模型: {{ config.llm_model_name || '未设置' }}</span>
              </div>
            </el-radio>
          </el-radio-group>
        </el-card>

        <!-- 用户自定义配置管理 -->
        <el-card shadow="inner" class="config-section">
            <template #header>
                <div class="card-header">
                    <strong>用户自定义配置</strong>
                    <el-button type="success" @click="handleAdd">添加新配置</el-button>
                </div>
            </template>
            <el-table :data="userConfigs" stripe>
                <el-table-column prop="name" label="配置名称" />
                <el-table-column prop="llm_api_url" label="API URL" />
                <el-table-column prop="llm_model_name" label="模型名称" />
                <el-table-column label="操作" width="150">
                    <template #default="scope">
                        <el-button type="primary" link @click="handleEdit(scope.row)">编辑</el-button>
                        <el-button type="danger" link @click="handleDelete(scope.row.id)">删除</el-button>
                    </template>
                </el-table-column>
            </el-table>
            <el-empty v-if="userConfigs.length === 0" description="暂无自定义配置" />
        </el-card>
      </div>
      <el-empty v-else description="正在加载用户配置..." />
    </el-card>

    <!-- 编辑/新增配置对话框 -->
    <el-dialog v-model="dialogVisible" :title="isNewConfig ? '添加新配置' : '编辑配置'" width="50%">
        <el-form :model="currentConfig" label-width="120px">
            <el-form-item label="配置名称">
                <el-input v-model="currentConfig.name" />
            </el-form-item>
            <el-form-item label="API URL">
                <el-input v-model="currentConfig.llm_api_url" />
            </el-form-item>
            <el-form-item label="API Key">
                <el-input v-model="currentConfig.llm_api_key" type="password" show-password />
            </el-form-item>
            <el-form-item label="模型名称">
                <el-input v-model="currentConfig.llm_model_name" />
            </el-form-item>
        </el-form>
        <template #footer>
            <el-button @click="dialogVisible = false">取消</el-button>
            <el-button type="primary" @click="handleSaveInDialog">确认</el-button>
        </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.settings-page {
  max-width: 1000px;
  margin: auto;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.config-section {
  margin-top: 20px;
}
.el-radio-group {
    display: flex;
    flex-direction: column;
    gap: 15px;
}
.el-radio.is-bordered {
    width: 100%;
    height: auto;
    padding: 15px;
}
.config-details {
    margin-top: 8px;
    font-size: 12px;
    color: #909399;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
</style>
