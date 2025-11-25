<script setup>
// --- 1. 导入 (保持不变) ---
import { ref, onMounted, onUnmounted, computed } from 'vue';
import { ElMessage, ElMessageBox, ElDialog } from 'element-plus';
import { useAuthStore } from '@/stores/auth';
import apiService from '@/services/apiService';
import GraphVisualizer from '@/components/GraphVisualizer.vue';

// --- 2. 状态 (保持不变) ---
const authStore = useAuthStore();
const taskList = ref([]);
const isLoading = ref(false);
const uploadFileRef = ref(null);
const activeUploads = ref(0);
const isUploading = computed(() => activeUploads.value > 0);
const dialogVisible = ref(false);
const selectedGraphData = ref(null);
const isGraphLoading = ref(false);
const fullNodeCount = ref(0);
const fullEdgeCount = ref(0);
const isGraphCropped = ref(false);

let pollInterval = null;

// --- 3. API 调用 (handleViewResults 等函数保持不变) ---
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
  } finally {
    isLoading.value = false;
  }
};

const handleFileUpload = async (options) => {
  const file = options.file;
  if (!file) return;

  if (file.type !== 'application/pdf') {
    ElMessage.error(`文件 [${file.name}] 不是 PDF，已跳过`);
    return;
  }

  activeUploads.value++;
  const formData = new FormData();
  formData.append('pdf_file', file);

  try {
    const response = await apiService.uploadPdf(formData);
    taskList.value.unshift(response.data);
    ElMessage.success(`文件 [${file.name}] 已提交后台解析`);
  } catch (error) {
    if (error.response?.status !== 401) {
      if (uploadFileRef.value) {
        uploadFileRef.value.handleRemove(file);
      }
      ElMessage.error(`文件 [${file.name}] 上传失败`);
    }
  } finally {
    activeUploads.value--;
  }
};

const handleExceed = (files) => {
  ElMessage.warning(
    `当前限制选择 10 个文件，本次选择了 ${files.length} 个文件，请重新选择。`
  );
};

const handleDeleteTask = (task) => {
  ElMessageBox.confirm(`确定要删除任务 [${task.pdf_filename}] 吗？相关的解析结果（如果有）也将被删除。`, { type: 'warning' })
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
    }).catch(() => { });
};

const handleViewResults = async (task) => {
  dialogVisible.value = true;
  isGraphLoading.value = true;
  selectedGraphData.value = null; // 清空旧数据
  fullNodeCount.value = 0;
  fullEdgeCount.value = 0;
  isGraphCropped.value = false; // 重置裁剪状态

  const MAX_NODES_TO_RENDER = 150;

  try {
    const response = await apiService.getTaskGraphData(task.id);
    const fullGraphData = response.data;

    if (!fullGraphData || !fullGraphData.nodes) {
        throw new Error("API返回的图谱数据格式不正确");
    }

    fullNodeCount.value = fullGraphData.nodes.length;
    fullEdgeCount.value = fullGraphData.edges.length;

    // --- 3. (*** 过滤逻辑 ***) ---
    const allNodes = fullGraphData.nodes;
    const deviceNodeIds = new Set(
        allNodes.filter(n => n.group === 'Device').map(n => n.id)
    );

    const userFilteredEdges = fullGraphData.edges.filter(edge => {
      return deviceNodeIds.has(edge.source);
    });

    const connectedNodeIds = new Set();
    userFilteredEdges.forEach(edge => {
      connectedNodeIds.add(edge.source);
      connectedNodeIds.add(edge.target);
    });

    const userFilteredNodes = allNodes.filter(node =>
      deviceNodeIds.has(node.id) || connectedNodeIds.has(node.id)
    );
    // --- 过滤结束 ---

    let nodesToRender = userFilteredNodes;
    let edgesToRender = userFilteredEdges;

    if (nodesToRender.length > MAX_NODES_TO_RENDER) {
      isGraphCropped.value = true;

      const deviceNodes = nodesToRender.filter(n => n.group === 'Device');
      const otherNodes = nodesToRender.filter(n => n.group !== 'Device');
      const remainingLimit = MAX_NODES_TO_RENDER - deviceNodes.length;
      const limitedOtherNodes = otherNodes.slice(0, Math.max(0, remainingLimit));
      nodesToRender = [...deviceNodes, ...limitedOtherNodes];

      const finalNodeIds = new Set(nodesToRender.map(n => n.id));
      edgesToRender = userFilteredEdges.filter(
        e => finalNodeIds.has(e.source) && finalNodeIds.has(e.target)
      );
    }

    selectedGraphData.value = {
      nodes: nodesToRender,
      edges: edgesToRender,
    };

  } catch (error) {
    console.error('获取图谱数据失败:', error);
    ElMessage.error('获取图谱数据失败，请检查API或网络');
    dialogVisible.value = false;
  } finally {
    isGraphLoading.value = false;
  }
};

// --- (5) 轮询和辅助函数 (保持不变) ---
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
    case 'GRAPH_CONSTRUCTION': return 'warning'; // 10
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
    case 'CLASSIFICATION': return '阶段9: 器件分类中...';
    case 'GRAPH_CONSTRUCTION': return '阶段10: 构建知识图谱...';
    case 'PENDING': return '排队中...';
    default: return '未知';
  }
};
const getStatusPercentage = (status) => {
  switch (status) {
    case 'PENDING': return 5;
    case 'TEXT_ANALYSIS': return 10;
    case 'VLM_ANALYSIS': return 20;
    case 'TEXT_CHUNKING': return 30;
    case 'MODEL_EXTRACTION': return 40;
    case 'PARAM_EXTRACTION': return 50;
    case 'PARAM_FUSION': return 60;
    case 'IMAGE_ASSOCIATION': return 70;
    case 'MANUFACTURER_STANDARDIZATION': return 80;
    case 'CLASSIFICATION': return 90;
    case 'GRAPH_CONSTRUCTION': return 95;
    case 'COMPLETED': return 100;
    case 'FAILED': return 0;
    default: return 0;
  }
};


// --- (6) 生命周期钩子 (保持不变) ---
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
          :limit="10"
          :on-exceed="handleExceed"
          :disabled="isUploading"
          accept="application/pdf"
          multiple
        >
          <el-button type="primary" :loading="isUploading">
            {{ isUploading ? `正在上传 (${activeUploads}个)...` : '点击上传 PDF (可多选)' }}
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
          <el-table-column prop="pdf_filename" label="文件名" min-width="200"/>

          <el-table-column label="状态" width="220">
            <template #default="scope">
              <el-tag v-if="scope.row.status === 'COMPLETED'" type="success" effect="dark">
                {{ getStatusText(scope.row.status) }}
              </el-tag>
              <el-tag v-else-if="scope.row.status === 'FAILED'" type="danger" effect="dark">
                {{ getStatusText(scope.row.status) }}
              </el-tag>

              <div v-else>
                <el-progress
                  :percentage="getStatusPercentage(scope.row.status)"
                  :status="getStatusType(scope.row.status)"
                  :stroke-width="20"
                  :text-inside="true"
                >
                  <span
                    style="font-size: 12px; color: #fff; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
                    {{ getStatusText(scope.row.status) }}
                  </span>
                </el-progress>
              </div>
            </template>
          </el-table-column>

          <el-table-column prop="created_at" label="上传时间" width="180">
            <template #default="scope">
              {{ new Date(scope.row.created_at).toLocaleString() }}
            </template>
          </el-table-column>
          <el-table-column prop="error_message" label="信息/错误" min-width="200"
                           show-overflow-tooltip/>
          <el-table-column label="操作" width="150">
            <template #default="scope">
              <el-button
                v-if="scope.row.status === 'COMPLETED'"
                type="primary"
                link
                @click="handleViewResults(scope.row)"
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
        <el-empty v-if="!isLoading && taskList.length === 0" description="暂无解析任务"/>
      </el-card>

    </el-card>

    <el-dialog
      v-model="dialogVisible"
      title="解析结果知识图谱"
      width="75%"
      top="5vh"
    >
      <div v-loading="isGraphLoading" style="min-height: 400px;">
        <div v-if="selectedGraphData">
          <el-alert
            type="success"
            :closable="false"
            style="margin-bottom: 15px;"
          >
            <template #title>
              <strong>解析成功！</strong>
              共提取到 {{ fullNodeCount }} 个节点
              和 {{ fullEdgeCount }} 条关系。
            </template>
          </el-alert>

          <el-alert
            v-if="isGraphCropped"
            type="warning"
            :closable="false"
            style="margin-bottom: 15px;"
          >
            <template #title>
              <strong>渲染提醒：</strong>
              为保证渲染流畅，当前仅显示 {{ selectedGraphData.nodes.length }} / {{ fullNodeCount }}
              个节点 (已优先包含所有 'Device' 节点)
              和 {{ selectedGraphData.edges.length }} / {{ fullEdgeCount }} 条相关关系。
            </template>
          </el-alert>
          <GraphVisualizer :graphData="selectedGraphData"/>
        </div>
      </div>
    </el-dialog>

  </div>
</template>

<style scoped>
/* (样式保持不变) */
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

/* (*** 关键修改 ***)
  移除了 .el-table .cell div { text-align: center; }
  因为它会导致进度条文本（如果不在内部）也居中，且不再需要
*/
</style>
