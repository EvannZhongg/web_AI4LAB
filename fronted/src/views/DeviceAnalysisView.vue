<script setup>
import { ref, reactive, onMounted, watch, computed } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import Papa from 'papaparse';
import { use } from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import { LineChart } from 'echarts/charts';
import { TitleComponent, TooltipComponent, GridComponent, LegendComponent, DataZoomComponent } from 'echarts/components';
import VChart from 'vue-echarts';
import { useAuthStore } from '@/stores/auth';
import apiService from '@/services/apiService';

use([CanvasRenderer, LineChart, TitleComponent, TooltipComponent, GridComponent, LegendComponent, DataZoomComponent]);

const props = defineProps({ id: { type: String, required: false } });
const router = useRouter();
const authStore = useAuthStore();

// --- Core State ---
const allDevices = ref([]);
const selectedDeviceId = ref(props.id ? parseInt(props.id, 10) : null);
const selectedDevice = ref(null);
const experiments = ref([]);
const loading = ref(false);
const selectLoading = ref(false);

// --- Search & Filter State ---
const searchQuery = ref('');
const filterDeviceType = ref('');

// --- NEW: Temporary storage for files added in the current dialog session ---
// Maps metadata_id (frontend temp ID) to File object
const pendingCsvFiles = reactive({});

// --- Editor Dialog State ---
const editorDialog = reactive({
    visible: false,
    isNew: false,
    title: '',
    data: {
        id: null,
        name: '',
        experiment_type: '',
        grid_data_for_editing: [['参数'], ['']],
        csv_files_metadata: []
    }
});

// --- CSV Metadata Dialog State ---
const csvMetadataDialog = reactive({
    visible: false,
    filename: '',
    columns: [],
    // Store the actual file temporarily while defining metadata
    fileToDefine: null
});

// --- Photo Zoom/Pan State ---
const photoState = reactive({ /* ... (不变) ... */
    scale: 1, translateX: 0, translateY: 0, isDragging: false, lastX: 0, lastY: 0
});
const photoTransform = computed(() => `scale(${photoState.scale}) translate(${photoState.translateX}px, ${photoState.translateY}px)`);

// --- Chart State ---
const activeExperimentId = ref(null);
const gridVisualization = reactive({ xAxisKey: '', yAxisKey: '' });
const waveform = reactive({ /* ... (不变) ... */
    activeCsvId: null, // This ID should now match metadata_id from backend
    availableCsvs: [],
    xAxisKey: '',
    yAxisKey: '',
    loadingCsvData: false,
    csvData: null,
    fetchError: null
});

// --- Helper Function ---
const reconstructGridData = (experiment) => { /* ... (不变) ... */
    if (!experiment || !experiment.parameters || experiment.parameters.length === 0) {
        return [[''], ['']];
    }
    const sortedParams = [...experiment.parameters].sort((a, b) => a.column_index - b.column_index);
    const headers = sortedParams.map(p => p.name);
    const pointsMap = {};
    let maxRow = -1;
    sortedParams.forEach(param => {
        (param.datapoints || []).forEach(dp => {
            if (!pointsMap[dp.row_index]) { pointsMap[dp.row_index] = {}; }
            pointsMap[dp.row_index][param.id] = dp.value_text;
            if (dp.row_index > maxRow) { maxRow = dp.row_index; }
        });
    });
    const rows = [];
    for (let i = 0; i <= maxRow; i++) {
        rows.push(sortedParams.map(param => pointsMap[i]?.[param.id] ?? ''));
    }
    if (rows.length === 0 && headers.length > 0) {
        rows.push(Array(headers.length).fill(''));
    }
    return [headers, ...rows];
};


// --- Computed Properties ---
const filteredDevicesForSelect = computed(() => { /* ... (不变) ... */
     return allDevices.value.filter(device => {
        const typeMatch = !filterDeviceType.value || device.device_type === filterDeviceType.value;
        const searchMatch = !searchQuery.value ||
            device.name.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
            device.device_number.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
            (device.tech_description && device.tech_description.toLowerCase().includes(searchQuery.value.toLowerCase()));
        return typeMatch && searchMatch;
    });
});
const currentExperiment = computed(() => { /* ... (不变) ... */
    return experiments.value.find(exp => exp.id === activeExperimentId.value);
});
const gridChartAxisOptions = computed(() => { /* ... (不变) ... */
    const experiment = currentExperiment.value;
    return experiment?.grid_data?.[0] || [];
});
const gridChartOptions = computed(() => { /* ... (不变) ... */
    const experiment = currentExperiment.value;
    if (!experiment || !gridVisualization.xAxisKey || !gridVisualization.yAxisKey) {
        return { title: { text: '请选择数据表并配置X/Y轴', left: 'center', top: 'center' } };
    }
    const gridData = experiment.grid_data;
    if (!gridData || gridData.length < 2) return { title: { text: '无可用数据', left: 'center', top: 'center' } };
    const headers = gridData[0];
    const dataRows = gridData.slice(1);
    const xIndex = headers.indexOf(gridVisualization.xAxisKey);
    const yIndex = headers.indexOf(gridVisualization.yAxisKey);
    if (xIndex === -1 || yIndex === -1) return { title: { text: '轴配置错误', left: 'center', top: 'center' } };
    const validData = dataRows
        .map(row => ({ x: row[xIndex], y: parseFloat(row[yIndex]) }))
        .filter(item => item.x != null && !isNaN(item.y));
    return {
        title: { text: `${experiment.name} - 数据可视化`, left: 'center' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', name: headers[xIndex], data: validData.map(item => item.x) },
        yAxis: { type: 'value', name: headers[yIndex] },
        dataZoom: [{ type: 'inside' }, { type: 'slider' }],
        series: [{ data: validData.map(item => item.y), type: 'line', smooth: true }],
    };
});

const csvAxisOptions = computed(() => { /* ... (不变) ... */
    if (!waveform.activeCsvId) return [];
    const experiment = currentExperiment.value;
    // 使用 metadata_id 查找
    const csvMeta = experiment?.csv_files_metadata?.find(f => f.metadata_id === waveform.activeCsvId);
    return csvMeta ? csvMeta.columns : [];
});

const waveformChartOptions = computed(() => { /* ... (不变) ... */
    const experiment = currentExperiment.value;
    const csvMeta = experiment?.csv_files_metadata?.find(f => f.metadata_id === waveform.activeCsvId); // Use metadata_id

    if (!experiment || !waveform.activeCsvId || !csvMeta) {
        return { title: { text: '请选择一个CSV文件以加载波形', left: 'center', top: 'center' } };
    }
    if (waveform.loadingCsvData) {
        return { title: { text: `正在加载 ${csvMeta.filename}...`, left: 'center', top: 'center' } };
    }
    if (waveform.fetchError) {
         return { title: { text: `加载 ${csvMeta.filename} 失败`, subtext: waveform.fetchError , textStyle: { color: 'red'}, left: 'center', top: 'center' } };
    }
    if (!waveform.xAxisKey || !waveform.yAxisKey || !waveform.csvData || waveform.csvData.errors?.length > 0) {
        const subtext = waveform.csvData?.errors?.length > 0 ? `CSV解析错误: ${waveform.csvData.errors[0].message}` : '请选择X/Y轴';
        return { title: { text: `${experiment.name} - ${csvMeta.filename}`, subtext: subtext, left: 'center', top: 'center' } };
    }

    const columns = csvMeta.columns;
    const data = waveform.csvData.data || [];
    const xCol = columns.find(c => c.key === waveform.xAxisKey);
    const yCol = columns.find(c => c.key === waveform.yAxisKey);
    if (!xCol || !yCol) return { title: { text: '轴配置错误', left: 'center', top: 'center' } };
    const xDataIndex = columns.findIndex(c => c.key === waveform.xAxisKey);
    const yDataIndex = columns.findIndex(c => c.key === waveform.yAxisKey);
    if (xDataIndex === -1 || yDataIndex === -1) {
        return { title: { text: '无法在CSV数据中找到选定轴', left: 'center', top: 'center' } };
    }
    const plotData = data.map(row => {
        const xVal = parseFloat(row[xDataIndex]);
        const yVal = parseFloat(row[yDataIndex]);
        if (!isNaN(xVal) && !isNaN(yVal)) { return [xVal, yVal]; }
        return null;
    }).filter(point => point !== null);

    return {
      title: { text: `${experiment.name} - ${csvMeta.filename}`, left: 'center' },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'value', name: `${xCol.name} (${xCol.unit || ''})`, scale: true },
      yAxis: { type: 'value', name: `${yCol.name} (${yCol.unit || ''})`, scale: true },
      dataZoom: [{ type: 'inside', filterMode: 'weakFilter' }, { type: 'slider', filterMode: 'weakFilter' }],
      series: [{ data: plotData, type: 'line', showSymbol: false, smooth: false }],
    };
});

// --- 获取并解析 CSV 数据 (不变) ---
const fetchCsvData = async (experimentId, metadataId) => { // Parameter renamed to metadataId for clarity
    if (!experimentId || !metadataId) {
        waveform.csvData = null; waveform.fetchError = null; return;
    }
    waveform.loadingCsvData = true; waveform.csvData = null; waveform.fetchError = null;
    console.log(`Fetching CSV data for experiment ${experimentId}, metadata_id ${metadataId}`);
    try {
        // Use metadataId in the API call
        const response = await apiService.getExperimentCsvData(experimentId, metadataId);
        const csvText = response.data;
        Papa.parse(csvText, {
            header: false, skipEmptyLines: true, dynamicTyping: false,
            complete: (results) => {
                console.log("CSV Parsed:", results);
                if (results.errors.length > 0) {
                     waveform.fetchError = `CSV解析错误: ${results.errors[0].message}`;
                     waveform.csvData = { errors: results.errors };
                } else { waveform.csvData = results; }
                waveform.loadingCsvData = false;
            },
            error: (error) => {
                console.error("PapaParse Error:", error);
                waveform.fetchError = 'CSV 解析失败'; waveform.loadingCsvData = false;
            }
        });
    } catch (error) {
        console.error("Error fetching CSV data:", error);
        waveform.fetchError = error.response?.data?.error || error.message || '获取 CSV 数据失败';
        if (error.response?.status === 404) { waveform.fetchError = '找不到对应的 CSV 文件 (404)'; }
        waveform.loadingCsvData = false;
    }
};

// --- Business Logic ---
const fetchAllDevices = async () => { /* ... (不变) ... */
   selectLoading.value = true;
  try {
    const response = await apiService.getDevices({ limit: 1000 }); // Fetch more for better filtering
    allDevices.value = response.data.results || response.data;
  } catch(error) {
    if (error.response?.status !== 401) { ElMessage.error('获取器件列表失败！'); }
    console.error(error);
  } finally { selectLoading.value = false; }
};

const loadAnalysisData = async (deviceId) => { /* ... (不变) ... */
  if (!deviceId) {
    selectedDevice.value = null; experiments.value = []; activeExperimentId.value = null;
    waveform.activeCsvId = null; waveform.csvData = null; return;
  }
  loading.value = true; waveform.csvData = null; waveform.fetchError = null;
  try {
    const response = await apiService.getDeviceById(deviceId);
    selectedDevice.value = response.data;
    // Map metadata_id from backend response
    experiments.value = (selectedDevice.value.experiments || []).map(exp => ({
        ...exp,
        grid_data: reconstructGridData(exp),
        // Ensure csv_files_metadata has the correct id field for selection
        csv_files_metadata: (exp.csv_files_metadata || []).map(meta => ({
            ...meta,
            id: meta.metadata_id // Use metadata_id as the primary id for frontend selection
        }))
    }));
    activeExperimentId.value = null; photoState.scale = 1; photoState.translateX = 0; photoState.translateY = 0;
    waveform.activeCsvId = null; waveform.availableCsvs = [];
    gridVisualization.xAxisKey = ''; gridVisualization.yAxisKey = '';
  } catch(error) {
    if (error.response?.status !== 401) { ElMessage.error(`加载器件ID: ${deviceId} 的数据失败！`); }
    selectedDevice.value = null; experiments.value = []; activeExperimentId.value = null;
    waveform.activeCsvId = null; waveform.csvData = null;
    router.replace({ name: 'device-analysis' }); console.error(error);
  } finally { loading.value = false; }
};

onMounted(fetchAllDevices);

watch(() => props.id, (newId) => { /* ... (不变) ... */
  const numId = newId ? parseInt(newId, 10) : null;
  if (selectedDeviceId.value !== numId) {
     selectedDeviceId.value = numId; loadAnalysisData(numId);
  } else if (numId && !selectedDevice.value) { loadAnalysisData(numId); }
  else if (!numId) {
      selectedDevice.value = null; experiments.value = []; activeExperimentId.value = null;
      waveform.activeCsvId = null; waveform.csvData = null;
  }
}, { immediate: true });


const handleSelectChange = (newId) => { /* ... (不变) ... */
  if (String(props.id || '') !== String(newId)) {
    router.push({ name: 'device-analysis', params: { id: newId } });
  }
};

// --- Event Handlers ---
const onPhotoWheel = (e) => { /* ... (不变) ... */ photoState.scale = Math.max(0.5, Math.min(5, photoState.scale + (e.deltaY > 0 ? -0.1 : 0.1))); };
const onPhotoMouseDown = (e) => { /* ... (不变) ... */ photoState.isDragging = true; photoState.lastX = e.clientX; photoState.lastY = e.clientY; };
const onPhotoMouseMove = (e) => { /* ... (不变) ... */
    if (!photoState.isDragging) return; const dx = e.clientX - photoState.lastX; const dy = e.clientY - photoState.lastY;
    photoState.translateX += dx / photoState.scale; photoState.translateY += dy / photoState.scale;
    photoState.lastX = e.clientX; photoState.lastY = e.clientY;
};
const onPhotoMouseUp = () => { /* ... (不变) ... */ photoState.isDragging = false; };

const handleTableSelect = (newActiveExperimentId) => { /* ... (不变) ... */
    activeExperimentId.value = newActiveExperimentId; const experiment = currentExperiment.value;
    waveform.csvData = null; waveform.fetchError = null;
    if (!experiment) {
        waveform.activeCsvId = null; waveform.availableCsvs = [];
        gridVisualization.xAxisKey = ''; gridVisualization.yAxisKey = ''; return;
    }
    const gridData = experiment.grid_data;
    if (gridData && gridData.length > 1 && gridData[0].length > 0) {
        const headers = gridData[0]; gridVisualization.xAxisKey = headers[0] || '';
        gridVisualization.yAxisKey = headers.length > 1 ? headers[1] : headers[0];
    } else { gridVisualization.xAxisKey = ''; gridVisualization.yAxisKey = ''; }
    // Ensure availableCsvs uses metadata_id as 'id'
    waveform.availableCsvs = (experiment.csv_files_metadata || []).map(meta => ({
         ...meta,
         id: meta.metadata_id // Important for el-select binding
    }));
    if (waveform.availableCsvs.length > 0) {
        const currentSelectionExists = waveform.availableCsvs.some(csv => csv.id === waveform.activeCsvId);
        if (!currentSelectionExists || !waveform.activeCsvId) {
             waveform.activeCsvId = waveform.availableCsvs[0].id; // Use metadata_id
        }
         // Fetch is triggered by watcher
    }
    else { waveform.activeCsvId = null; }
};

watch(() => waveform.activeCsvId, (newCsvId, oldCsvId) => { /* ... (不变, 使用 metadata_id) ... */
    if (newCsvId && newCsvId !== oldCsvId) {
        const experiment = currentExperiment.value;
        const csvMeta = experiment?.csv_files_metadata?.find(f => f.metadata_id === newCsvId); // Use metadata_id
        if (csvMeta?.columns?.length > 0) {
            waveform.xAxisKey = csvMeta.columns[0].key;
            waveform.yAxisKey = csvMeta.columns.length > 1 ? csvMeta.columns[1].key : csvMeta.columns[0].key;
        } else { waveform.xAxisKey = ''; waveform.yAxisKey = ''; }
        fetchCsvData(activeExperimentId.value, newCsvId); // Pass metadata_id
    } else if (!newCsvId) {
        waveform.csvData = null; waveform.fetchError = null;
        waveform.xAxisKey = ''; waveform.yAxisKey = '';
    }
});


// --- Dialog and CRUD Logic ---

const openAddTableDialog = () => { /* ... (不变) ... */
  if (!selectedDevice.value) { ElMessage.warning('请先选择一个器件'); return; }
  // Clear pending files when opening dialog for new experiment
  Object.keys(pendingCsvFiles).forEach(key => delete pendingCsvFiles[key]);
  editorDialog.isNew = true; editorDialog.title = '添加新实验数据表';
  editorDialog.data = { id: null, name: '新的实验数据', experiment_type: '', grid_data_for_editing: [['参数'], ['']], csv_files_metadata: [] };
  editorDialog.visible = true;
};

const openEditTableDialog = (experiment) => { /* ... (不变) ... */
  // Clear pending files when opening dialog for editing
  Object.keys(pendingCsvFiles).forEach(key => delete pendingCsvFiles[key]);
  editorDialog.isNew = false; editorDialog.title = `编辑: ${experiment.name}`;
  editorDialog.data = {
      id: experiment.id, name: experiment.name, experiment_type: experiment.experiment_type,
      grid_data_for_editing: JSON.parse(JSON.stringify(experiment.grid_data || [[''], ['']])),
      // Ensure metadata has metadata_id
      csv_files_metadata: JSON.parse(JSON.stringify((experiment.csv_files_metadata || []).map(m => ({...m, id: m.metadata_id})))) // Use metadata_id as id
  };
  editorDialog.visible = true;
};

const triggerCsvUpload = () => { /* ... (不变) ... */ document.getElementById('csv-uploader').click(); };

// --- 修改 handleCsvFileChange 以存储文件 ---
const handleCsvFileChange = (event) => {
  const file = event.target.files[0];
  if (!file) return;

  // Store the file temporarily for the metadata dialog
  csvMetadataDialog.fileToDefine = file;

  Papa.parse(file, {
    header: false, preview: 1, skipEmptyLines: true,
    complete: (results) => {
      if (results.errors.length > 0) { ElMessage.error('CSV文件头部解析失败！'); console.error(results.errors); return; }
      if (!results.data || results.data.length === 0 || results.data[0].length === 0) {
           ElMessage.warning('CSV文件为空或格式不正确。'); return;
      }
      csvMetadataDialog.filename = file.name;
      csvMetadataDialog.columns = results.data[0].map((header, index) => ({
          key: `col_${index}`, name: header || `列 ${index + 1}`, unit: ''
      }));
      csvMetadataDialog.visible = true;
    },
    error: (err) => { ElMessage.error('CSV 文件读取出错！'); console.error(err); }
  });
  event.target.value = '';
};

// --- 修改 confirmCsvMetadata 以存储文件到 pending ---
const confirmCsvMetadata = () => {
  if (!csvMetadataDialog.fileToDefine) {
      ElMessage.error("内部错误：未找到待处理的CSV文件");
      return;
  }
  const tempMetadataId = Date.now(); // Use timestamp as temporary unique ID
  const newCsvMeta = {
      // id: tempMetadataId, // Use metadata_id instead for clarity
      metadata_id: tempMetadataId, // Add metadata_id field
      filename: csvMetadataDialog.filename,
      columns: JSON.parse(JSON.stringify(csvMetadataDialog.columns))
      // No data stored here
  };
  // Store the actual file object keyed by the temp metadata ID
  pendingCsvFiles[tempMetadataId] = csvMetadataDialog.fileToDefine;

  editorDialog.data.csv_files_metadata.push(newCsvMeta);
  csvMetadataDialog.visible = false;
  csvMetadataDialog.fileToDefine = null; // Clear the temporary file holder
};

// --- 修改 deleteCsvInDialog 以清理 pending 文件 ---
const deleteCsvInDialog = (metadataId) => { // Parameter renamed to metadataId
  const index = editorDialog.data.csv_files_metadata.findIndex(f => f.metadata_id === metadataId);
  if (index !== -1) {
      editorDialog.data.csv_files_metadata.splice(index, 1);
      // If the file was pending upload, remove it from pending list
      if (pendingCsvFiles[metadataId]) {
          delete pendingCsvFiles[metadataId];
          console.log(`Removed pending file for metadata_id: ${metadataId}`);
      }
      // TODO: If the file was already uploaded (editing existing), maybe call backend delete?
      // Requires knowing if the metadataId corresponds to an already saved file.
  }
};


// --- 修改 handleSaveChangesInDialog 以包含文件上传 ---
const handleSaveChangesInDialog = async () => {
  if (!editorDialog.data.name) { ElMessage.warning('表格名称不能为空！'); return; }
  if (!editorDialog.data.experiment_type) { ElMessage.warning('实验类型不能为空！'); return; }

  console.log('[Save Dialog Start] Pending files keys:', JSON.stringify(Object.keys(pendingCsvFiles)));

  const basicData = { /* ... (不变) ... */
      name: editorDialog.data.name,
      experiment_type: editorDialog.data.experiment_type,
  };
  const gridDataPayload = { /* ... (不变) ... */
      parameters: editorDialog.data.grid_data_for_editing[0].map((name, index) => ({ name: name || `参数${index+1}`, column_index: index, unit: '' })),
      datapoints: editorDialog.data.grid_data_for_editing.slice(1).flatMap((row, rowIndex) => row.map((cell, colIndex) => ({ parameter_col_index: colIndex, row_index: rowIndex, value_text: cell ?? '' })))
  };
  const csvMetadataPayload = { /* ... (不变) ... */
      csv_files_metadata: editorDialog.data.csv_files_metadata.map(meta => ({ metadata_id: meta.metadata_id, filename: meta.filename, columns: meta.columns }))
  };

loading.value = true;
  // --- 不要在这里关闭对话框 ---
  // editorDialog.visible = false;

  let savedExperimentId = editorDialog.data.id; // 提前获取 ID
  let successfullySaved = false; // 标记是否成功

  try {
      let isNewExperiment = editorDialog.isNew;

      // 1. Save Basic Info & Grid Data & Metadata (without files)
      if (isNewExperiment) {
          if (!selectedDevice.value?.id) throw new Error("No device selected.");
          const createPayload = { ...basicData, device: selectedDevice.value.id };
          const response = await apiService.createExperiment(createPayload);
          savedExperimentId = response.data.id; // 获取新 ID
          if (!savedExperimentId) throw new Error("Failed to create experiment.");
          // Add basic info locally immediately for responsiveness (or wait till reload)
          // experiments.value.push({ ...response.data, grid_data: [['...']], parameters: [], csv_files_metadata: [] });
      } else {
          await apiService.updateExperiment(savedExperimentId, basicData);
      }

      await apiService.updateExperimentGridData(savedExperimentId, gridDataPayload);
      await apiService.updateExperimentCsvMetadata(savedExperimentId, csvMetadataPayload);
      console.log('[Save Dialog After Meta Save] Pending files keys:', JSON.stringify(Object.keys(pendingCsvFiles)));

      // 2. Upload Pending CSV Files
      const uploadPromises = [];
      const successfullyUploadedMetaIds = [];

      console.log('[Save Dialog Before Upload Loop] Pending files keys:', JSON.stringify(Object.keys(pendingCsvFiles)));
      for (const metadataId in pendingCsvFiles) {
          const file = pendingCsvFiles[metadataId];
          if (file && savedExperimentId) {
              console.log(`[Save Dialog] Preparing to upload file for metadata_id: ${metadataId}`, file);
              const formData = new FormData();
              formData.append('file', file, file.name);
              formData.append('metadata_id', metadataId);

              uploadPromises.push(
                  apiService.uploadExperimentCsv(savedExperimentId, formData)
                    .then(() => {
                        successfullyUploadedMetaIds.push(metadataId);
                        console.log(`Successfully uploaded file for metadata_id: ${metadataId}`);
                    })
                    .catch(uploadError => {
                         console.error(`Failed to upload file for metadata_id: ${metadataId}`, uploadError);
                         ElMessage.error(`上传文件 ${file.name} 失败: ${uploadError.response?.data?.error || uploadError.message || '未知错误'}`);
                    })
              );
          } else {
              console.warn(`[Save Dialog] Skipped upload for metadata_id: ${metadataId}. File or savedExperimentId missing.`);
          }
      }

      console.log(`[Save Dialog] Starting ${uploadPromises.length} CSV uploads...`);
      await Promise.all(uploadPromises);
      console.log(`[Save Dialog] Finished CSV uploads.`);

      // 清理成功的上传
      successfullyUploadedMetaIds.forEach(id => {
          if (pendingCsvFiles[id]) { delete pendingCsvFiles[id]; }
      });

      const failedUploadCount = Object.keys(pendingCsvFiles).length;
      if (failedUploadCount === 0) {
           ElMessage.success('实验数据及所有文件保存成功！');
           successfullySaved = true; // 标记成功
           // 成功后再关闭对话框
           editorDialog.visible = false;
           // 重新加载数据
           await loadAnalysisData(selectedDeviceId.value);
      } else {
          ElMessage.error(`实验数据已保存，但有 ${failedUploadCount} 个文件上传失败，请重试或检查文件。`);
          // 不关闭对话框，允许用户重试？或者需要更复杂的重试逻辑
          // 可以考虑将失败的 ID 留在 pendingCsvFiles 中
          // 重新加载数据以更新已保存的部分
          await loadAnalysisData(selectedDeviceId.value);
      }

  } catch (error) {
       successfullySaved = false; // 标记失败
       if (error.response?.status !== 401) { ElMessage.error(`保存失败: ${error.message || '请检查数据或联系管理员'}`); }
       console.error("Save error:", error);
       // 错误时仍然清理 pending files
       Object.keys(pendingCsvFiles).forEach(key => delete pendingCsvFiles[key]);
  } finally {
      loading.value = false;
      // 只有在保存失败且对话框仍然可见时才在这里清理？
      // @closed 事件处理器会处理对话框关闭时的清理，这里的逻辑可以简化或移除
      // if (!successfullySaved && editorDialog.visible) {
      //     // Maybe don't clear here, let @closed handle it
      // }
  }
};

const deleteCustomTable = (experimentId) => { /* ... (不变) ... */
    ElMessageBox.confirm('确定要删除这个实验数据表吗？此操作不可撤销。', '警告', { type: 'warning' })
        .then(async () => {
            loading.value = true;
            try {
                await apiService.deleteExperiment(experimentId);
                ElMessage.success('实验数据表已删除！');
                const index = experiments.value.findIndex(exp => exp.id === experimentId);
                if (index !== -1) { experiments.value.splice(index, 1); }
                if (activeExperimentId.value === experimentId) {
                    activeExperimentId.value = null; waveform.activeCsvId = null; waveform.csvData = null;
                }
            } catch (error) {
                 if (error.response?.status !== 401) { ElMessage.error('删除失败！'); }
                 console.error(error);
            } finally { loading.value = false; }
        })
        .catch(() => {});
};

// --- Dialog Table Editing Helpers ---
const addColumnInDialog = () => { /* ... (不变) ... */
    const table = editorDialog.data.grid_data_for_editing; if (table.length === 0) table.push([]);
    const headerRow = table[0]; const newHeaderName = `新参数${headerRow.length + 1}`;
    headerRow.push(newHeaderName); for(let i = 1; i < table.length; i++) { table[i].push(''); }
};
const deleteColumnInDialog = (index) => { /* ... (不变) ... */
    if (index < 0 || editorDialog.data.grid_data_for_editing[0]?.length <= 1) return;
    editorDialog.data.grid_data_for_editing.forEach(row => { if (row.length > index) { row.splice(index, 1); } });
};
const addRowInDialog = () => { /* ... (不变) ... */
    const table = editorDialog.data.grid_data_for_editing; const headerLength = table.length > 0 ? table[0].length : 0;
    const newRow = Array(headerLength).fill(''); table.push(newRow);
};
const deleteRowInDialog = (rowIndex) => { /* ... (不变) ... */
    const table = editorDialog.data.grid_data_for_editing;
    if (rowIndex >= 0 && (rowIndex + 1) < table.length && table.length > 2) { table.splice(rowIndex + 1, 1); }
};

</script>

<template>
  <div class="device-analysis-page">
    <el-card shadow="never" class="selector-card">
       <el-form :inline="true" @submit.prevent>
        <el-form-item label="器件搜索"> <el-input v-model="searchQuery" placeholder="名称/编号/技术说明" clearable style="width: 250px;" /> </el-form-item>
        <el-form-item label="器件类型"> <el-select v-model="filterDeviceType" placeholder="所有类型" clearable style="width: 150px;"> <el-option label="PIN" value="PIN" /> <el-option label="其他" value="其他" /> </el-select> </el-form-item>
        <el-form-item label="选择器件">
          <el-select v-model="selectedDeviceId" filterable placeholder="请选择" style="width: 300px;" :loading="selectLoading" @change="handleSelectChange">
            <el-option v-for="device in filteredDevicesForSelect" :key="device.id" :label="`${device.name} (${device.device_number})`" :value="device.id"/>
          </el-select>
        </el-form-item>
      </el-form>
    </el-card>

    <div v-if="selectedDevice" class="analysis-content" v-loading="loading">
      <el-row :gutter="20" class="main-content">
        <el-col :span="14" class="full-height-col data-panel">
          <el-card shadow="never" class="photo-card-main">
              <template #header><strong>{{ selectedDevice.name }} 微观照片</strong></template>
               <div class="photo-container" @wheel.prevent="onPhotoWheel" @mousedown.prevent="onPhotoMouseDown" @mousemove.prevent="onPhotoMouseMove" @mouseup="onPhotoMouseUp" @mouseleave="onPhotoMouseUp">
                   <el-image v-if="selectedDevice && selectedDevice.photo_data" :src="selectedDevice.photo_data" fit="contain" class="micro-photo" :style="{ transform: photoTransform }"/>
                   <el-empty v-else description="暂无照片" />
               </div>
          </el-card>
          <el-card shadow="never" class="editable-table-card">
             <template #header> <div class="card-header"> <strong>{{ selectedDevice.name }} 实验数据</strong> <el-button type="success" size="small" @click="openAddTableDialog">添加新实验</el-button> </div> </template>
            <div class="custom-tables-container">
              <el-collapse accordion @change="handleTableSelect" v-if="experiments.length > 0">
                <el-collapse-item v-for="experiment in experiments" :key="experiment.id" :name="experiment.id">
                <template #title>
                    <div class="collapse-title-wrapper">
                        <span class="collapse-title">{{ experiment.name }}</span>
                        <div class="collapse-actions"> <el-button type="primary" link @click.stop="openEditTableDialog(experiment)">编辑</el-button> <el-button type="danger" link @click.stop="deleteCustomTable(experiment.id)">删除</el-button> </div>
                    </div>
                </template>
                <div class="table-meta"><strong>实验类型:</strong> {{ experiment.experiment_type || '未指定' }}</div>
                <table class="preview-table" v-if="experiment.grid_data && experiment.grid_data.length > 0">
                    <thead> <tr><th v-for="(header, index) in experiment.grid_data[0]" :key="`h-${index}`">{{ header }}</th></tr> </thead>
                    <tbody>
                    <tr v-for="(row, rowIndex) in experiment.grid_data.slice(1)" :key="`r-${rowIndex}`"> <td v-for="(cell, colIndex) in row" :key="`c-${rowIndex}-${colIndex}`">{{ cell }}</td> </tr>
                    <tr v-if="experiment.grid_data.length === 1"> <td :colspan="experiment.grid_data[0].length || 1">暂无数据行</td> </tr>
                    </tbody>
                </table>
                 <div v-else class="table-meta">此实验无表格数据</div>
                 <div v-if="experiment.csv_files_metadata && experiment.csv_files_metadata.length > 0" style="margin-top: 10px;">
                     <strong>关联CSV文件:</strong>
                     <el-tag v-for="csv in experiment.csv_files_metadata" :key="csv.metadata_id || csv.filename" size="small" style="margin-left: 5px;">{{ csv.filename }}</el-tag>
                 </div>
                </el-collapse-item>
              </el-collapse>
              <el-empty v-else description="暂无自定义实验数据"></el-empty>
            </div>
          </el-card>
        </el-col>

        <el-col :span="10" class="full-height-col visual-panel">
          <el-card shadow="never" class="data-visualization-card">
            <template #header> <div class="card-header"> <strong>数据可视化</strong> <div class="waveform-controls" v-if="activeExperimentId"> <el-select v-model="gridVisualization.xAxisKey" placeholder="X轴" size="small" style="width: 120px;"> <el-option v-for="key in gridChartAxisOptions" :key="key" :label="key" :value="key" /> </el-select> <el-select v-model="gridVisualization.yAxisKey" placeholder="Y轴" size="small" style="width: 120px;"> <el-option v-for="key in gridChartAxisOptions" :key="key" :label="key" :value="key" /> </el-select> </div> </div> </template>
            <v-chart class="waveform-chart" :option="gridChartOptions" autoresize />
          </el-card>

          <el-card shadow="never" class="waveform-card">
            <template #header> <div class="card-header"> <strong>CSV波形窗口</strong>
                 <div class="waveform-controls" v-if="activeExperimentId && waveform.activeCsvId">
                    <el-select v-model="waveform.activeCsvId" placeholder="选择CSV文件" size="small" v-if="waveform.availableCsvs.length > 1" style="width: 150px;">
                        <el-option v-for="csv in waveform.availableCsvs" :key="csv.metadata_id" :label="csv.filename" :value="csv.metadata_id" />
                    </el-select>
                    <el-select v-model="waveform.xAxisKey" placeholder="X轴" size="small" style="width: 100px;"> <el-option v-for="col in csvAxisOptions" :key="col.key" :label="col.name" :value="col.key" /> </el-select>
                    <el-select v-model="waveform.yAxisKey" placeholder="Y轴" size="small" style="width: 100px;"> <el-option v-for="col in csvAxisOptions" :key="col.key" :label="col.name" :value="col.key" /> </el-select>
                </div>
                 <div v-else-if="activeExperimentId && waveform.availableCsvs.length === 0" style="font-size: 12px; color: #909399;"> 当前实验无关联CSV </div>
            </div> </template>
            <v-chart class="waveform-chart" :option="waveformChartOptions" autoresize v-loading="waveform.loadingCsvData"/>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <el-card v-else shadow="never" class="placeholder-card"> <el-empty description="请从上方搜索并选择一个器件开始分析" /> </el-card>

    <el-dialog v-model="editorDialog.visible" :title="editorDialog.title" width="80%" top="5vh" @closed="Object.keys(pendingCsvFiles).forEach(key => delete pendingCsvFiles[key])"> <el-form :model="editorDialog.data" label-width="100px">
            <el-form-item label="实验名称"><el-input v-model="editorDialog.data.name" /></el-form-item>
            <el-form-item label="实验类型"><el-input v-model="editorDialog.data.experiment_type" placeholder="例如: 注入试验" /></el-form-item>
        </el-form>
        <el-divider content-position="left">关联的CSV文件元数据</el-divider>
        <div class="csv-list">
             <el-tag v-for="csv in editorDialog.data.csv_files_metadata" :key="csv.metadata_id" closable @close="deleteCsvInDialog(csv.metadata_id)">{{ csv.filename }}</el-tag>
            <el-button @click="triggerCsvUpload" size="small" type="primary" link>+ 添加CSV元数据</el-button>
            <input type="file" id="csv-uploader" @change="handleCsvFileChange" accept=".csv" style="display: none;" />
        </div>
        <el-divider content-position="left">表格数据</el-divider>
         <div class="table-controls-dialog"> <el-button @click="addRowInDialog" size="small">添加行</el-button> <el-button @click="addColumnInDialog" size="small">添加列</el-button> </div>
        <div class="editable-table-container-dialog">
            <table class="editable-table">
                <thead> <tr> <th v-for="(header, index) in editorDialog.data.grid_data_for_editing[0]" :key="index"> <el-input v-model="editorDialog.data.grid_data_for_editing[0][index]" size="small" placeholder="参数名称" /> <el-button type="danger" circle size="small" class="delete-btn" @click="deleteColumnInDialog(index)" :disabled="editorDialog.data.grid_data_for_editing[0].length <= 1">X</el-button> </th> <th></th> </tr> </thead>
                <tbody> <tr v-for="(row, rowIndex) in editorDialog.data.grid_data_for_editing.slice(1)" :key="rowIndex"> <td v-for="(cell, colIndex) in row" :key="colIndex"> <el-input v-model="editorDialog.data.grid_data_for_editing[rowIndex + 1][colIndex]" size="small" /> </td> <td> <el-button type="danger" circle size="small" class="delete-btn" @click="deleteRowInDialog(rowIndex)" :disabled="editorDialog.data.grid_data_for_editing.length <= 2">X</el-button> </td> </tr> </tbody>
            </table>
        </div>
        <template #footer> <el-button @click="editorDialog.visible = false">取消</el-button> <el-button type="primary" @click="handleSaveChangesInDialog">确认保存</el-button> </template>
    </el-dialog>
    <el-dialog v-model="csvMetadataDialog.visible" :title="`定义CSV文件列元数据: ${csvMetadataDialog.filename}`" width="60%">
        <p>请为导入的CSV文件 <strong>{{ csvMetadataDialog.filename }}</strong> 的每一列定义名称和单位 (此操作仅保存元数据)。</p>
        <el-table :data="csvMetadataDialog.columns" border> <el-table-column label="原始列" type="index" width="80" /> <el-table-column label="数据名称 (Header)"><template #default="scope"><el-input v-model="scope.row.name" /></template></el-table-column> <el-table-column label="单位"><template #default="scope"><el-input v-model="scope.row.unit" /></template></el-table-column> </el-table>
        <template #footer> <el-button @click="csvMetadataDialog.visible = false; csvMetadataDialog.fileToDefine = null;">取消</el-button> <el-button type="primary" @click="confirmCsvMetadata">确认元数据</el-button> </template>
    </el-dialog>
  </div>
</template>

<style scoped>
/* Styles remain unchanged */
.device-analysis-page { display: flex; flex-direction: column; height: 100%; gap: 20px; }
.selector-card { flex-shrink: 0; }
.selector-card .el-form-item { margin-bottom: 0; }
.analysis-content { flex-grow: 1; min-height: 0; }
.main-content { height: 100%; }
.full-height-col { height: 100%; display: flex; flex-direction: column; }
.card-header { display: flex; justify-content: space-between; align-items: center; width: 100%; }
.placeholder-card { flex-grow: 1; display: flex; align-items: center; justify-content: center; }
.data-panel { gap: 20px; }
.visual-panel { gap: 20px; }
.photo-card-main { flex: 1 1 40%; min-height: 0; display: flex; flex-direction: column; }
.photo-card-main :deep(.el-card__body) { flex: 1 1 auto; min-height: 0; padding: 10px; display: flex; align-items: center; justify-content: center; }
.photo-container { width: 100%; height: 100%; overflow: hidden; cursor: grab; }
.photo-container:active { cursor: grabbing; }
.micro-photo { width: 100%; height: 100%; transition: transform 0.1s ease-out; }
.editable-table-card { flex: 1 1 60%; min-height: 0; display: flex; flex-direction: column; }
.editable-table-card :deep(.el-card__body) { flex: 1 1 auto; min-height: 0; padding: 15px; display: flex; flex-direction: column; }
.custom-tables-container { flex: 1 1 auto; overflow-y: auto; }
.collapse-title-wrapper { display: flex; justify-content: space-between; align-items: center; width: 100%; }
.collapse-title { font-weight: bold; flex-grow: 1; }
.collapse-actions { margin-right: 10px; }
.table-meta { font-size: 12px; color: #909399; margin-bottom: 10px; }
.preview-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 10px;}
.preview-table th, .preview-table td { border: 1px solid #ebeef5; padding: 6px 8px; text-align: center; }
.preview-table th { background-color: #fafafa; }
.data-visualization-card, .waveform-card { flex: 1 1 50%; min-height: 0; display: flex; flex-direction: column; }
.data-visualization-card :deep(.el-card__body), .waveform-card :deep(.el-card__body) { flex: 1 1 auto; min-height: 0; padding: 10px; display: flex; }
.waveform-chart { flex: 1 1 auto; min-height: 0; }
.waveform-controls { display: flex; gap: 10px; }
.csv-list { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 20px; }
.table-controls-dialog { margin-bottom: 10px; }
.editable-table-container-dialog { max-height: 50vh; overflow: auto; border: 1px solid #ebeef5; margin-bottom: 20px;}
.editable-table { width: 100%; border-collapse: collapse; }
.editable-table th, .editable-table td { border: 1px solid #ebeef5; padding: 4px; text-align: center; vertical-align: middle; }
.editable-table th { position: relative; background-color: #fafafa; padding: 8px 4px;}
.editable-table th .el-input, .editable-table td .el-input { width: 100%; }
.editable-table .delete-btn { position: absolute; top: 50%; right: 2px; transform: translateY(-50%); opacity: 0.3; transition: opacity 0.2s; padding: 2px; min-height: unset; height: 18px; width: 18px; }
.editable-table th:hover .delete-btn { opacity: 1; }
.editable-table td:last-child { width: 40px; padding: 0; }
.editable-table td:last-child .delete-btn { position: static; transform: none; opacity: 0.5; display: inline-flex; vertical-align: middle; padding: 2px; min-height: unset; height: 18px; width: 18px; }
.editable-table td:last-child:hover .delete-btn { opacity: 1; }
.editable-table th:last-child { width: 40px; border: none; background-color: transparent; }
.editable-table td:not(:last-child) { padding: 2px; }
</style>
