<script setup>
import { ref, reactive, onMounted, computed } from 'vue';
import { ElMessage } from 'element-plus';
import { use } from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import { LineChart } from 'echarts/charts';
import { TitleComponent, TooltipComponent, GridComponent, LegendComponent, DataZoomComponent } from 'echarts/components';
import VChart from 'vue-echarts';
import { useAuthStore } from '@/stores/auth';
import apiService from '@/services/apiService';

use([CanvasRenderer, LineChart, TitleComponent, TooltipComponent, GridComponent, LegendComponent, DataZoomComponent]);

const authStore = useAuthStore();

// --- 筛选表单状态 ---
const allDeviceTypes = ref([]);
const allExperimentTypes = ref([]);
const availableParams = ref([]);

const filterForm = reactive({
  device_type: '',
  experiment_type: '',
  fixed_params: [], // { name: '', min: '', max: '' }
  x_axis_param: '',
  y_axis_param: '',
});

// --- 数据与UI状态 ---
const filterLoading = ref(false);
const chartLoading = ref(false);
const filteredDevices = ref([]); // 后端筛选后返回的器件列表
const selectedDeviceIds = ref([]); // 用户勾选的、用于绘图的器件ID

const chartOption = ref({
    title: { text: '请配置筛选条件并生成图表', left: 'center', top: 'center' }
});

// --- API 调用 ---
const fetchInitialOptions = async () => {
    try {
        const response = await apiService.getDevices({ limit: 1000 });
        const devices = response.data.results || response.data;
        allDeviceTypes.value = Array.from(new Set(devices.map(d => d.device_type).filter(Boolean)));

        const expTypes = new Set();
        devices.forEach(d => {
            if(d.test_types_display) {
                d.test_types_display.split('/').forEach(type => {
                    if(type) expTypes.add(type.trim());
                });
            }
        });
        allExperimentTypes.value = Array.from(expTypes);
    } catch {
        ElMessage.error('获取筛选选项失败！');
    }
};

onMounted(fetchInitialOptions);

// --- 核心筛选与绘图逻辑 ---

const handleFilterDevices = async () => {
    if (!filterForm.device_type || !filterForm.experiment_type) {
        ElMessage.warning('请先选择器件类型和实验类型！');
        return;
    }
    if (!authStore.isAuthenticated) {
        ElMessage.error('请先登录以执行筛选操作！');
        return;
    }

    filterLoading.value = true;
    try {
        const payload = {
            device_type: filterForm.device_type,
            experiment_type: filterForm.experiment_type,
            fixed_params: filterForm.fixed_params,
        };
        const response = await apiService.compareDevices(payload, { timeout: 30000 }); // 30秒超时
        filteredDevices.value = response.data.filtered_devices;
        availableParams.value = response.data.available_params;
        selectedDeviceIds.value = filteredDevices.value.map(d => d.id);
        if(filteredDevices.value.length === 0){
            ElMessage.info('没有找到符合所有范围条件的器件。');
        }
    } catch (error) {
        if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
            ElMessage.error('筛选超时，请尝试缩小参数范围或检查后端性能。');
        } else if (error.response?.status !== 401) {
             ElMessage.error('筛选器件失败！');
        }
        console.error(error);
    } finally {
        filterLoading.value = false;
    }
};

const generateComparisonChart = () => {
    if (!filterForm.x_axis_param || !filterForm.y_axis_param) {
        ElMessage.warning('请选择X轴和Y轴的参数！');
        return;
    }
    if(selectedDeviceIds.value.length === 0){
        ElMessage.warning('请至少勾选一个器件进行渲染！');
        return;
    }

    chartLoading.value = true;

    const devicesToPlot = filteredDevices.value.filter(d => selectedDeviceIds.value.includes(d.id));
    const allXValues = new Set();
    const seriesData = [];

    devicesToPlot.forEach(device => {
        const deviceSeries = {
            name: `${device.name} (${device.device_number})`,
            type: 'line',
            smooth: true,
            data: []
        };

        const relevantTable = device.device_specific_data.find(t => t.experiment_type === filterForm.experiment_type);
        if (!relevantTable) return;

        const headers = relevantTable.grid_data[0];
        const rows = relevantTable.grid_data.slice(1);
        const xIndex = headers.indexOf(filterForm.x_axis_param);
        const yIndex = headers.indexOf(filterForm.y_axis_param);

        if (xIndex === -1 || yIndex === -1) return;

        rows.forEach(row => {
            try {
                const xVal = parseFloat(row[xIndex]);
                const yVal = parseFloat(row[yIndex]);
                if (!isNaN(xVal) && !isNaN(yVal)) {
                    allXValues.add(xVal);
                    deviceSeries.data.push([xVal, yVal]);
                }
            } catch {}
        });

        deviceSeries.data.sort((a, b) => a[0] - b[0]);
        seriesData.push(deviceSeries);
    });

    const sortedXAxis = Array.from(allXValues).sort((a, b) => a - b);

    chartOption.value = {
        // 核心修复：使用富文本优化标题和副标题
        title: {
            text: `{a|器件对比: ${filterForm.y_axis_param} vs ${filterForm.x_axis_param}}\n{b|器件类型: ${filterForm.device_type} | 实验类型: ${filterForm.experiment_type}}`,
            left: 'center',
            textStyle: {
                rich: {
                    a: {
                        fontSize: 18,
                        fontWeight: 'bold',
                        color: '#303133'
                    },
                    b: {
                        fontSize: 14,
                        color: '#909399',
                        padding: [10, 0, 0, 0] // 为副标题添加上边距
                    }
                }
            }
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'cross' }
        },
        // 恢复器件图例，并优化布局
        legend: {
            orient: 'vertical',
            right: 20,
            top: 'center',
            data: seriesData.map(s => s.name),
            type: 'scroll'
        },
        grid: {
            left: '10%',
            right: 200, // 为右侧的图例留出固定空间
            bottom: '15%',
            containLabel: true
        },
        xAxis: { type: 'category', name: filterForm.x_axis_param, data: sortedXAxis },
        yAxis: { type: 'value', name: filterForm.y_axis_param, scale: true },
        dataZoom: [{ type: 'inside' }, { type: 'slider' }],
        series: seriesData
    };

    chartLoading.value = false;
};

// --- 辅助功能 ---
const addFixedParam = () => {
    filterForm.fixed_params.push({ name: '', min: '', max: '' });
};

const removeFixedParam = (index) => {
    filterForm.fixed_params.splice(index, 1);
};
</script>

<template>
  <div class="device-comparison-page">
    <el-row :gutter="20" class="main-layout">
      <!-- 左侧：筛选与控制 -->
      <el-col :span="8" class="control-panel">
        <el-card shadow="never">
          <template #header><strong>异类数据比较筛选</strong></template>
          <el-form :model="filterForm" label-position="top">

            <el-form-item label="1. 选择基础类型">
              <el-select v-model="filterForm.device_type" placeholder="选择器件类型" style="width: 100%; margin-bottom: 10px;">
                <el-option v-for="item in allDeviceTypes" :key="item" :label="item" :value="item" />
              </el-select>
              <el-select v-model="filterForm.experiment_type" placeholder="选择实验类型" style="width: 100%;">
                 <el-option v-for="item in allExperimentTypes" :key="item" :label="item" :value="item" />
              </el-select>
            </el-form-item>

            <el-form-item label="2. 固定参数范围筛选">
                <div v-for="(param, index) in filterForm.fixed_params" :key="index" class="param-row">
                    <el-select v-model="param.name" placeholder="参数名称" class="param-name">
                        <el-option v-for="p in availableParams" :key="p" :label="p" :value="p" />
                    </el-select>
                    <el-input v-model="param.min" placeholder="最小值" class="param-value" />
                    <span>-</span>
                    <el-input v-model="param.max" placeholder="最大值" class="param-value" />
                    <el-button type="danger" link @click="removeFixedParam(index)">移除</el-button>
                </div>
                <el-button @click="addFixedParam" size="small" style="width:100%">+ 添加固定参数</el-button>
            </el-form-item>

            <el-button type="primary" @click="handleFilterDevices" :loading="filterLoading" style="width: 100%; margin-top: 10px;">筛选器件</el-button>

            <el-divider />

            <el-card shadow="never" class="filtered-devices-card" v-if="filteredDevices.length > 0">
                <template #header><strong>3. 勾选需要渲染的器件</strong></template>
                <el-checkbox-group v-model="selectedDeviceIds">
                    <el-checkbox v-for="device in filteredDevices" :key="device.id" :label="device.id">
                        {{ device.name }} ({{ device.device_number }})
                    </el-checkbox>
                </el-checkbox-group>
            </el-card>

            <el-form-item label="4. 定义可视化轴">
                <el-select v-model="filterForm.x_axis_param" placeholder="选择X轴参数" style="width: 100%; margin-bottom: 10px;">
                    <el-option v-for="p in availableParams" :key="p" :label="p" :value="p" />
                </el-select>
                <el-select v-model="filterForm.y_axis_param" placeholder="选择Y轴参数" style="width: 100%;">
                    <el-option v-for="p in availableParams" :key="p" :label="p" :value="p" />
                </el-select>
            </el-form-item>

            <el-button type="success" @click="generateComparisonChart" :loading="chartLoading" style="width: 100%; margin-top: 10px;">生成对比图表</el-button>
          </el-form>
        </el-card>
      </el-col>

      <!-- 右侧：图表展示 -->
      <el-col :span="16">
        <el-card shadow="never" class="chart-card">
            <v-chart class="chart" :option="chartOption" autoresize v-loading="chartLoading || filterLoading" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.device-comparison-page, .main-layout, .control-panel, .chart-card {
    height: 100%;
}
.control-panel .el-card, .chart-card {
    display: flex;
    flex-direction: column;
    height: 100%;
}
.control-panel :deep(.el-card__body) {
    flex-grow: 1;
    overflow-y: auto;
}
.chart-card :deep(.el-card__body) {
    flex-grow: 1;
    padding: 10px;
    display: flex;
}
.chart {
    flex-grow: 1;
    min-height: 0;
}
.param-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}
.param-name { flex: 2; }
.param-value { flex: 1; }

.filtered-devices-card :deep(.el-card__body) {
    max-height: 200px;
    overflow-y: auto;
}
.el-checkbox-group {
    display: flex;
    flex-direction: column;
}
</style>

