<script setup>
import { ref, reactive, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import { useAuthStore } from '@/stores/auth';
import apiService from '@/services/apiService'; // 1. 导入新的 apiService

const router = useRouter();
const authStore = useAuthStore();

const deviceList = ref([]);
const loading = ref(false);
const allDeviceTypes = ref([]);
const allExperimentTypes = ref([]);

const queryForm = reactive({
  device_type: '',
  experiment_type: '',
  search: ''
});
const deleteDeviceNumber = ref('');

const importDialogVisible = ref(false);
const importForm = reactive({
  name: '',
  device_type: '',
  substrate: '',
  device_number: '',
  tech_description: '',
  photo_data: ''
});

const editDialogVisible = ref(false);
const editForm = reactive({
  id: null,
  name: '',
  device_type: '',
  substrate: '',
  device_number: '',
  tech_description: '',
});


// --- API 调用 (已重构) ---
const fetchDevices = async () => {
  loading.value = true;
  try {
    // 2. 使用 apiService
    const response = await apiService.getDevices(queryForm);
    deviceList.value = response.data.results || response.data;
  } catch (error) {
    if (error.response?.status !== 401) {
      ElMessage.error('数据加载失败！');
    }
    console.error(error);
  } finally {
    loading.value = false;
  }
};

const fetchFilterOptions = async () => {
    try {
        // 3. 使用 apiService
        const response = await apiService.getDevices({ limit: 1000 });
        const devices = response.data.results || response.data;
        const types = new Set(devices.map(d => d.device_type).filter(Boolean));
        allDeviceTypes.value = Array.from(types);

        const experimentTypes = new Set();
        devices.forEach(device => {
            if (device.test_types_display) {
                device.test_types_display.split('/').forEach(type => {
                    if (type) experimentTypes.add(type.trim());
                });
            }
        });
        allExperimentTypes.value = Array.from(experimentTypes);
    } catch (error) {
       if (error.response?.status !== 401) {
            ElMessage.error('获取筛选选项失败！');
       }
       console.error(error);
    }
}


onMounted(() => {
    fetchDevices();
    fetchFilterOptions();
});

const handleQuery = () => fetchDevices();

const handlePhotoUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        importForm.photo_data = e.target.result;
        ElMessage.success("照片已选择");
    };
    reader.readAsDataURL(file);
    event.target.value = '';
};

const handleImport = async () => {
    if (!importForm.device_number) { ElMessage.warning('器件编号不能为空！'); return; }
    try {
        // 4. 使用 apiService (不再需要手动添加 headers)
        await apiService.createDevice(importForm);
        ElMessage.success('新器件导入成功！');
        importDialogVisible.value = false;
        Object.keys(importForm).forEach(key => importForm[key] = '');
        await fetchDevices();
        await fetchFilterOptions();
    } catch (error) {
        if (error.response?.status !== 401) {
            ElMessage.error('导入失败，请检查数据格式或器件编号是否唯一！');
        }
        console.error(error);
    }
};

const openEditDialog = (device) => {
    Object.assign(editForm, device);
    editDialogVisible.value = true;
};

const handleUpdate = async () => {
    if (!editForm.id) return;
    try {
        const payload = {
            name: editForm.name,
            device_type: editForm.device_type,
            substrate: editForm.substrate,
            tech_description: editForm.tech_description
        };
        // 5. 使用 apiService
        await apiService.updateDevice(editForm.id, payload);
        ElMessage.success('器件信息更新成功！');
        editDialogVisible.value = false;
        await fetchDevices();
        await fetchFilterOptions();
    } catch (error) {
        if (error.response?.status !== 401) {
            ElMessage.error('更新失败！');
        }
        console.error(error);
    }
};

const handleDeleteByNumber = async () => {
    if (!deleteDeviceNumber.value) { ElMessage.warning('请输入需删除的器件编号！'); return; }
    try {
        const response = await apiService.getDevices({ search: deleteDeviceNumber.value });
        if (response.data.results.length === 0) { ElMessage.error('未找到该器件编号！'); return; }
        const device = response.data.results[0];
        await ElMessageBox.confirm(`确定要删除器件 [${device.name}]，编号为 [${device.device_number}] 的数据吗？`, '警告', { type: 'warning' });

        // 6. 使用 apiService
        await apiService.deleteDevice(device.id);
        ElMessage.success('删除成功！');
        await fetchDevices();
        await fetchFilterOptions();
        deleteDeviceNumber.value = '';
    } catch (error) {
        if (error !== 'cancel' && error.response?.status !== 401) {
            ElMessage.error('删除操作失败！');
        }
        console.error(error);
    }
};

const handleDeleteByRow = async (row) => {
    try {
        await ElMessageBox.confirm(`确定要删除器件 [${row.name}]，编号为 [${row.device_number}] 的数据吗？`, '警告', { type: 'warning' });
        // 7. 使用 apiService
        await apiService.deleteDevice(row.id);
        ElMessage.success('删除成功！');
        await fetchDevices();
        await fetchFilterOptions();
    } catch (error) {
        if (error !== 'cancel' && error.response?.status !== 401) {
            ElMessage.error('删除操作失败！');
        }
        console.error(error);
    }
};


// --- 导航 ---
const goToAnalysis = (deviceId) => {
  router.push({ name: 'device-analysis', params: { id: deviceId } });
};
</script>

<template>
  <div class="device-management-page">
    <!-- 顶部操作栏 -->
    <el-card shadow="never" class="controls-card">
      <el-form :inline="true" :model="queryForm" class="control-form">
        <el-form-item>
          <el-button type="primary" @click="importDialogVisible = true">导入新器件</el-button>
        </el-form-item>
        <el-form-item label="综合查询">
            <el-input v-model="queryForm.search" placeholder="名称/编号/说明" clearable style="width: 180px;"></el-input>
        </el-form-item>
        <el-form-item label="按器件类型筛选">
          <el-select
            v-model="queryForm.device_type"
            placeholder="选择或输入类型"
            clearable
            filterable
            allow-create
            style="width: 150px;">
            <el-option
              v-for="item in allDeviceTypes"
              :key="item"
              :label="item"
              :value="item" />
          </el-select>
        </el-form-item>
        <el-form-item label="按实验类型筛选">
            <el-select
                v-model="queryForm.experiment_type"
                placeholder="选择或输入类型"
                clearable
                filterable
                allow-create
                style="width: 180px;">
                <el-option
                    v-for="item in allExperimentTypes"
                    :key="item"
                    :label="item"
                    :value="item" />
            </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="success" @click="handleQuery">查询</el-button>
        </el-form-item>
        <el-form-item label="按编号删除" class="delete-item">
          <el-input v-model="deleteDeviceNumber" placeholder="输入器件编号" />
        </el-form-item>
        <el-form-item>
          <el-button type="danger" @click="handleDeleteByNumber">删除</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 主列表 -->
    <el-card shadow="never" class="table-card">
      <template #header><strong>器件数据列表</strong></template>
      <el-table :data="deviceList" v-loading="loading" border stripe>
        <el-table-column prop="name" label="器件名称" width="120" />
        <el-table-column prop="device_type" label="器件类型" width="100" />
        <el-table-column prop="device_number" label="器件编号" width="150" />
        <el-table-column prop="test_types_display" label="实验类型" width="180" />
        <el-table-column prop="tech_description" label="技术说明" min-width="200" show-overflow-tooltip />
        <el-table-column label="操作" width="180" fixed="right">
          <template #default="scope">
            <el-button type="primary" link @click="goToAnalysis(scope.row.id)">进入分析</el-button>
            <el-button type="warning" link @click="openEditDialog(scope.row)">编辑</el-button>
            <el-button type="danger" link @click="handleDeleteByRow(scope.row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 导入新器件对话框 -->
    <el-dialog v-model="importDialogVisible" title="导入新器件基础信息" width="50%">
      <el-form :model="importForm" label-width="120px">
        <el-row :gutter="20">
          <el-col :span="12"><el-form-item label="器件名称"><el-input v-model="importForm.name" /></el-form-item></el-col>
          <el-col :span="12">
            <el-form-item label="器件类型">
              <el-select
                v-model="importForm.device_type"
                placeholder="选择或输入类型"
                filterable
                allow-create
                style="width: 100%;">
                <el-option
                  v-for="item in allDeviceTypes"
                  :key="item"
                  :label="item"
                  :value="item" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12"><el-form-item label="衬底材料"><el-input v-model="importForm.substrate" /></el-form-item></el-col>
          <el-col :span="12"><el-form-item label="器件编号"><el-input v-model="importForm.device_number" /></el-form-item></el-col>
          <el-col :span="24"><el-form-item label="技术说明"><el-input v-model="importForm.tech_description" type="textarea" :rows="2" /></el-form-item></el-col>
          <el-col :span="24">
            <el-form-item label="微观照片">
                <input type="file" @change="handlePhotoUpload" accept="image/*" />
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
      <template #footer>
        <el-button @click="importDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="handleImport">确认导入</el-button>
      </template>
    </el-dialog>

    <!-- 编辑器件对话框 -->
    <el-dialog v-model="editDialogVisible" title="编辑器件基础信息" width="50%">
      <el-form :model="editForm" label-width="120px">
        <el-row :gutter="20">
          <el-col :span="12"><el-form-item label="器件名称"><el-input v-model="editForm.name" /></el-form-item></el-col>
          <el-col :span="12">
            <el-form-item label="器件类型">
              <el-select
                v-model="editForm.device_type"
                placeholder="选择或输入类型"
                filterable
                allow-create
                style="width: 100%;">
                <el-option
                  v-for="item in allDeviceTypes"
                  :key="item"
                  :label="item"
                  :value="item" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12"><el-form-item label="衬底材料"><el-input v-model="editForm.substrate" /></el-form-item></el-col>
          <el-col :span="12"><el-form-item label="器件编号"><el-input v-model="editForm.device_number" disabled /></el-form-item></el-col>
          <el-col :span="24"><el-form-item label="技术说明"><el-input v-model="editForm.tech_description" type="textarea" :rows="2" /></el-form-item></el-col>
        </el-row>
      </el-form>
      <template #footer>
        <el-button @click="editDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="handleUpdate">确认更新</el-button>
      </template>
    </el-dialog>

  </div>
</template>

<style scoped>
.device-management-page {
  display: flex;
  flex-direction: column;
  height: 100%;
  gap: 20px;
}
.controls-card {
  flex-shrink: 0;
}
.table-card {
  flex-grow: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}
.table-card :deep(.el-card__body) {
  flex-grow: 1;
  padding: 0;
}
.table-card .el-table {
  height: 100%;
}
.control-form {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 15px;
}
.control-form .el-form-item {
  margin-bottom: 0;
}
.delete-item {
  margin-left: auto;
}
</style>
