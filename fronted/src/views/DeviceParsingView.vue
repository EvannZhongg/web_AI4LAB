<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import { useAuthStore } from '@/stores/auth';
import apiService from '@/services/apiService';

const authStore = useAuthStore();

const taskList = ref([]);
const isLoading = ref(false);
const isUploading = ref(false);
const uploadFileRef = ref(null);

let pollInterval = null;

// --- API 调用 ---

const fetchTasks = async () => {
  if (!authStore.isAuthenticated) return;
  isLoading.value = true;
  try {
    const response = await apiService.getParsingTasks();
    taskList.value = response.data;
  } catch (error) {
    if (error.response?.status !== 401) {
      ElMessage.error('获取任务列表失败');
    }
    console.error(error);
  } finally {
    isLoading.value = false;
  }
};

const handleFileUpload = async (options) => {
  const file = options.file;
  if (!file) return;

  if (file.type !== 'application/pdf') {
      ElMessage.error('只能上传 PDF 文件！');
      return;
  }

  isUploading.value = true;
  const formData = new FormData();
  formData.append('pdf_file', file);

  try {
    const response = await apiService.uploadPdf(formData);
    taskList.value.unshift(response.data);
    ElMessage.success(`文件 [${file.name}] 已提交后台解析`);
    uploadFileRef.value.clearFiles();
  } catch (error) {
    if (error.response?.status !== 401) {
      ElMessage.error('文件上传失败');
    }
    console.error(error);
  } finally {
    isUploading.value = false;
  }
};

const handleDeleteTask = (task) => {
    ElMessageBox.confirm(`确定要删除任务 [${task.pdf_filename}] 吗？相关的解析结果（如果有）也将被删除。`, '警告', { type: 'warning' })
    .then(async () => {
        try {
            await apiService.deleteParsingTask(task.id);
            ElMessage.success('任务已删除');
            taskList.value = taskList.value.filter(t => t.id !== task.id);
        } catch (error) {
            if (error.response?.status !== 401) {
                ElMessage.error('删除失败');
            }
        }
    }).catch(() => {});
};

// --- 轮询和辅助函数 ---

const pollPendingTasks = async () => {
    const pendingTaskIds = taskList.value
        .filter(t => t.status !== 'COMPLETED' && t.status !== 'FAILED')
        .map(t => t.id);

    if (pendingTaskIds.length === 0) return;

    for (const taskId of pendingTaskIds) {
        try {
            const response = await apiService.getTaskStatus(taskId);
            const updatedTask = response.data;
            const index = taskList.value.findIndex(t => t.id === taskId);
            if (index !== -1) {
                taskList.value[index] = updatedTask;
            }
        } catch (error) {
            console.error(`轮询任务 ${taskId} 失败:`, error);
        }
    }
};

const getStatusType = (status) => {
  switch (status) {
    case 'COMPLETED': return 'success';
    case 'FAILED': return 'danger';
    case 'TEXT_ANALYSIS': return 'primary'; // 1
    case 'VLM_ANALYSIS': return 'warning';  // 2
    case 'TEXT_CHUNKING': return 'primary'; // 3
    case 'MODEL_EXTRACTION': return 'warning'; // 4
    case 'PARAM_EXTRACTION': return 'primary'; // 5
    case 'PARAM_FUSION': return 'warning'; // 6
    case 'IMAGE_ASSOCIATION': return 'primary'; // 7
    case 'MANUFACTURER_STANDARDIZATION': return 'warning'; // 8
    case 'CLASSIFICATION': return 'primary'; // 9
    case 'GRAPH_CONSTRUCTION': return 'warning'; // 10 (新增)
    case 'PENDING': return 'info';
    default: return 'info';
  }
};

const getStatusText = (status) => {
  switch (status) {
    case 'COMPLETED': return '已完成';
    case 'FAILED': return '失败';
    case 'TEXT_ANALYSIS': return '阶段1: 文本解析中...';
    case 'VLM_ANALYSIS': return '阶段2: 图片分析中...';
    case 'TEXT_CHUNKING': return '阶段3: 文本分块中...';
    case 'MODEL_EXTRACTION': return '阶段4: 型号抽取/融合中...';
    case 'PARAM_EXTRACTION': return '阶段5: 参数提取中...';
    case 'PARAM_FUSION': return '阶段6: 参数融合/细化中...';
    case 'IMAGE_ASSOCIATION': return '阶段7: 图片关联中...';
    case 'MANUFACTURER_STANDARDIZATION': return '阶段8: 厂商标准化...';
    case 'CLASSIFICATION': return '阶段9: 器件分类中...'; // <--- 新增
    case 'GRAPH_CONSTRUCTION': return '阶段10: 构建知识图谱...';
    case 'PENDING': return '排队中...';
    default: return '未知';
  }
};
// --- 生命周期钩子 ---

onMounted(() => {
  if (authStore.isAuthenticated) {
    fetchTasks();
    pollInterval = setInterval(pollPendingTasks, 5000);
  } else {
      ElMessage.warning('请先登录以使用此功能');
  }
});

onUnmounted(() => {
  if (pollInterval) {
    clearInterval(pollInterval);
  }
});

</script>

<template>
  <div class="device-parsing-page">
    <el-card shadow="never">
      <template #header>
        <div class="card-header">
          <strong>PDF 器件解析任务</strong>
        </div>
      </template>

      <el-card shadow="inner" class="upload-card">
        <template #header><strong>上传新的 PDF 文件开始解析</strong></template>
        <el-upload
          ref="uploadFileRef"
          :http-request="handleFileUpload"
          :auto-upload="true"
          :limit="1"
          :disabled="isUploading"
          accept="application/pdf"
        >
          <el-button type="primary" :loading="isUploading">
            {{ isUploading ? '正在上传...' : '点击上传 PDF' }}
          </el-button>
          <template #tip>
            <div class="el-upload__tip">
              文件将在后台进行流水线解析，您可以在下方查看任务状态。
            </div>
          </template>
        </el-upload>
      </el-card>

      <el-card shadow="inner" class="task-list-card">
        <template #header><strong>历史解析任务</strong></template>
        <el-table :data="taskList" v-loading="isLoading" stripe>
          <el-table-column prop="pdf_filename" label="文件名" min-width="200" />
          <el-table-column label="状态" width="200">
            <template #default="scope">
              <el-tag :type="getStatusType(scope.row.status)">
                {{ getStatusText(scope.row.status) }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="created_at" label="上传时间" width="180">
              <template #default="scope">
                  {{ new Date(scope.row.created_at).toLocaleString() }}
              </template>
          </el-table-column>
          <el-table-column prop="error_message" label="信息/错误" min-width="200" show-overflow-tooltip />
          <el-table-column label="操作" width="150">
            <template #default="scope">
              <el-button
                v-if="scope.row.status === 'COMPLETED'"
                type="primary"
                link
                @click="() => {/* TODO: 导航到结果页 */ ElMessage.info('功能待实现')}"
              >
                查看结果
              </el-button>
              <el-button
                type="danger"
                link
                @click="handleDeleteTask(scope.row)"
              >
                删除
              </el-button>
            </template>
          </el-table-column>
        </el-table>
         <el-empty v-if="!isLoading && taskList.length === 0" description="暂无解析任务" />
      </el-card>

    </el-card>
  </div>
</template>

<style scoped>
.device-parsing-page {
  padding: 20px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.upload-card {
  margin-bottom: 20px;
}
.task-list-card {
  margin-top: 20px;
}
</style>
