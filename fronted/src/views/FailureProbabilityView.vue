<script setup>
import { ref, reactive, onMounted, computed } from 'vue'
// 1. 移除 axios 导入，导入 apiService
import apiService from '@/services/apiService';
import { ElMessage } from 'element-plus'
import Papa from 'papaparse'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import { TitleComponent, TooltipComponent, GridComponent, LegendComponent } from 'echarts/components'
import VChart from 'vue-echarts'

use([CanvasRenderer, LineChart, TitleComponent, TooltipComponent, GridComponent, LegendComponent]);

// --- 状态定义 ---
const components = ref([]) // 左侧组件列表
const prDbmInput = ref(-6.379)
const systemFailureProbability = ref(null)
const calculationLoading = ref(false)

const datasets = ref([]) // 右侧数据库中的数据集列表
const selectedDatasetId = ref(null)
const selectedDatasetForChart = ref(null)
const newDatasetName = ref('组件1_概率')

// --- ECharts 配置 ---
const chartOption = computed(() => {
  const data = selectedDatasetForChart.value?.data;
  if (!data || !data.x || !data.y) {
    return { title: { text: '请从下拉框选择一个数据集以显示曲线', left: 'center', top: 'center' } };
  }
  return {
    title: { text: '功率-概率曲线图', left: 'center' },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', name: '功率', data: data.x },
    yAxis: { type: 'value', name: '概率' },
    series: [{ data: data.y, type: 'line', smooth: true }],
  }
})

// --- 左侧功能：组件导入与计算 ---
const handleComponentFileChange = (event, component) => {
  const file = event.target.files[0];
  if (file) {
    component.file = file;
    component.fileName = file.name;
  }
}

const addComponent = () => {
  components.value.push({ id: Date.now(), weight: 0.4, file: null, fileName: '未选择文件', data: null });
}

const importComponents = () => {
  const parsePromises = components.value
    .filter(c => c.file)
    .map(c => new Promise((resolve, reject) => {
      Papa.parse(c.file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          c.data = results.data; // 将解析的数据存回组件对象
          resolve();
        },
        error: (err) => reject(err)
      });
    }));

  Promise.all(parsePromises)
    .then(() => ElMessage.success('所有文件已成功导入并解析！'))
    .catch(() => ElMessage.error('文件解析失败！'));
}

const calculateSystemFailure = async () => {
  calculationLoading.value = true;
  try {
    const payload = {
      pr_dbm: prDbmInput.value,
      components: components.value.filter(c => c.data && c.weight > 0) // 只发送有数据和权重的组件
    };
    // 2. 使用 apiService 进行计算
    const response = await apiService.calculateFailureProbability(payload);
    systemFailureProbability.value = response.data.system_failure_probability;
  } catch (error) {
    // 401 错误已在 apiService 中统一处理
    if (error.response?.status !== 401) {
        ElMessage.error('计算失败！');
    }
    console.error(error); // 仍然可以保留 console.error 用于调试
  } finally {
    calculationLoading.value = false;
  }
}

// --- 右侧功能：数据集管理 ---
const fetchDatasets = async () => {
  try {
    // 3. 使用 apiService 获取数据集列表
    const response = await apiService.getProbabilityDatasets();
    datasets.value = response.data;
  } catch(error) {
     if (error.response?.status !== 401) {
        ElMessage.error('获取数据集列表失败！');
     }
     console.error(error);
  }
}
onMounted(fetchDatasets);

const handleDatasetSelectionChange = () => {
  const selected = datasets.value.find(d => d.id === selectedDatasetId.value);
  selectedDatasetForChart.value = selected;
}

const handlePowerProbFileChange = (event) => {
  const file = event.target.files[0];
  if (!file) return;
  Papa.parse(file, {
    header: true,
    skipEmptyLines: true,
    complete: async (results) => {
      if (!newDatasetName.value) {
        ElMessage.warning('请输入要存入的数据库名称！');
        return;
      }
      const chartData = {
        x: results.data.map(row => row['功率']),
        y: results.data.map(row => row['概率']),
      };
      try {
        // 4. 使用 apiService 创建数据集
        await apiService.createProbabilityDataset({ name: newDatasetName.value, data: chartData });
        ElMessage.success(`数据集 [${newDatasetName.value}] 已保存到数据库！`);
        fetchDatasets(); // 刷新列表
      } catch(error) {
        if (error.response?.status !== 401) {
           ElMessage.error('保存到数据库失败，名称可能已存在！');
        }
        console.error(error);
      }
    }
  });
}
</script>

<template>
  <div class="failure-prob-page">
    <el-row :gutter="20">
      <el-col :span="12">
        <div class="left-panel">
          <el-card shadow="never">
            <template #header><strong>组件权重与文件路径</strong></template>
            <div v-for="comp in components" :key="comp.id" class="component-input-row">
              <el-input v-model.number="comp.weight" placeholder="权重" style="width: 80px;" />
              <div class="file-input-wrapper">
                <span>{{ comp.fileName }}</span>
                <input type="file" @change="handleComponentFileChange($event, comp)" accept=".csv" />
              </div>
            </div>
            <el-button @click="addComponent" type="text" style="margin-top: 10px;">+ 添加组件</el-button>
            <el-button @click="importComponents" type="primary" style="margin-left: 20px;">导入</el-button>
          </el-card>

          <el-card shadow="never">
            <template #header><strong>组件权重概率数据</strong></template>
            <el-table :data="components.flatMap(c => c.data || [])" border height="200px">
              <el-table-column prop="功率" label="功率" />
              <el-table-column prop="概率" label="概率" />
            </el-table>
          </el-card>

          <el-card shadow="never">
            <div class="calculation-section">
              <el-form-item label="到靶功率 Pr_dbm 输入:">
                <el-input v-model.number="prDbmInput" type="number" />
              </el-form-item>
              <el-button @click="calculateSystemFailure" :loading="calculationLoading">计算系统失效概率</el-button>
              <el-form-item label="系统失效概率:">
                <strong class="result-text">{{ systemFailureProbability?.toFixed(6) ?? 'N/A' }}</strong>
              </el-form-item>
            </div>
          </el-card>
        </div>
      </el-col>

      <el-col :span="12">
        <div class="right-panel">
          <el-card shadow="never">
            <template #header><strong>功率-概率曲线图</strong></template>
            <v-chart class="chart" :option="chartOption" autoresize />
          </el-card>

          <el-card shadow="never">
             <div class="dataset-controls">
                <div class="file-input-wrapper">
                    <span>选择功率概率文件并保存</span>
                    <input type="file" @change="handlePowerProbFileChange" accept=".csv" />
                </div>
                <el-input v-model="newDatasetName" placeholder="存入数据库名称" />
             </div>
             <div class="dataset-controls">
                <el-select v-model="selectedDatasetId" placeholder="从数据库读取" @change="handleDatasetSelectionChange" style="width: 100%;">
                    <el-option v-for="ds in datasets" :key="ds.id" :label="ds.name" :value="ds.id" />
                </el-select>
             </div>
          </el-card>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.failure-prob-page { height: 100%; }
.left-panel, .right-panel { display: flex; flex-direction: column; gap: 20px; }
.component-input-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.file-input-wrapper {
  position: relative;
  overflow: hidden;
  display: inline-block;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  padding: 5px 10px;
  cursor: pointer;
  flex-grow: 1;
}
.file-input-wrapper input[type=file] {
  position: absolute;
  left: 0;
  top: 0;
  opacity: 0;
  width: 100%;
  height: 100%;
  cursor: pointer;
}
.calculation-section { display: flex; align-items: center; gap: 15px; }
.result-text { color: #409eff; font-size: 1.1em; }
.chart { height: 400px; }
.dataset-controls { display: flex; gap: 10px; margin-bottom: 10px; align-items: center; }
</style>
