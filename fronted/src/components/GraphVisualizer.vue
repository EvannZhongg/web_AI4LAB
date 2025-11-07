<script setup>
import { ref, watch, onMounted } from 'vue';
import { use } from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import { GraphChart } from 'echarts/charts';
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
} from 'echarts/components';
import VChart from 'vue-echarts';

// 注册 ECharts 组件
use([
  CanvasRenderer,
  GraphChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
]);

const props = defineProps({
  graphData: {
    type: Object,
    required: true, // { nodes: [], edges: [] }
  },
});

const chartOption = ref({});

// 将节点和边数据转换为 ECharts Graph series
const updateChartData = (data) => {
  if (!data || !data.nodes) return;

  // ECharts 'graph' 使用 'categories'
  const categories = Array.from(new Set(data.nodes.map(n => n.group)));

  const echartsNodes = data.nodes.map(node => ({
    id: node.id,
    name: node.label, // ECharts 'label' in 'graph' uses 'name'
    category: categories.indexOf(node.group), // 分配类别索引
    symbolSize: node.group === 'Device' ? 50 : (node.group === 'Category' ? 40 : 30),
    label: {
      show: true,
      formatter: '{b}' // {b} 显示 'name'
    },
    // 存储原始数据
    value: node.value,
    originalGroup: node.group
  }));

  const echartsEdges = data.edges.map(edge => ({
    source: edge.source,
    target: edge.target,
    label: {
      show: true,
      formatter: '{c}' // {c} 显示 'value'
    },
    value: edge.label, // 存储关系类型
    lineStyle: {
      color: '#aaa',
      curveness: 0.1
    }
  }));

  chartOption.value = {
    tooltip: {
      formatter: (params) => {
        if (params.dataType === 'node') {
          return `<b>${params.data.name}</b><br />Community: ${params.data.originalGroup}`;
        }
        if (params.dataType === 'edge') {
          return `Relation: <b>${params.data.value}</b>`;
        }
        return '';
      }
    },
    legend: {
      data: categories,
      textStyle: {
        color: '#333'
      },
      bottom: 10
    },
    series: [
      {
        type: 'graph',
        layout: 'force',
        categories: categories.map(c => ({ name: c })),
        nodes: echartsNodes,
        edges: echartsEdges,
        roam: true, // 开启缩放和平移
        force: {
          repulsion: 150,
          edgeLength: [50, 100],
          layoutAnimation: true,
          gravity: 0.05
        },
        label: {
          position: 'right',
          color: '#000'
        },
        emphasis: {
          focus: 'adjacency',
          label: {
            show: true
          }
        }
      },
    ],
  };
};

// 监视 prop 变化并更新图表
watch(() => props.graphData, (newData) => {
  updateChartData(newData);
}, { deep: true });

// 初始加载
onMounted(() => {
  updateChartData(props.graphData);
});
</script>

<template>
  <div class="graph-container">
    <v-chart class="chart" :option="chartOption" autoresize />
  </div>
</template>

<style scoped>
.graph-container {
  width: 100%;
  height: 600px; /* 您可以按需调整图谱高度 */
}
.chart {
  width: 100%;
  height: 100%;
}
</style>
