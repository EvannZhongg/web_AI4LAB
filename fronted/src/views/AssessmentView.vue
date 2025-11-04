<script setup>
import { reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '@/stores/auth';
import apiService from '@/services/apiService'; // 1. 导入 apiService

const authStore = useAuthStore();

// --- 损伤评估模块的状态 ---
const damageInput = reactive({
  pt_gw: 1.5,
  gt_db: 40,
  gr_db: 20,
  f_ghz: 10,
  d_km: 100,
  lna_gain_db: 20
})
const damageOutput = reactive({
  ls_db: null,
  pr_dbm: null,
  lna_gain_db: null,
  limiter_loss_db: null,
  risk_level: '未知'
})
const damageLoading = ref(false)

// --- 通信链路模块的状态 ---
const linkInput = reactive({
  pt2_kw: 10,
  gt2_db: 30,
  gr2_db: 10,
  f2_ghz: 5,
  d2_km: 200,
  receiver_sensitivity_dbm: -90
})
const linkOutput = reactive({
  lp_db: null,
  link_margin_db: null,
  link_status: '未知'
})
const linkLoading = ref(false)

// --- API 调用函数 (已重构) ---
const handleDamageAssess = async () => {
  if (!authStore.isAuthenticated) {
    ElMessage.error('请先登录再进行评估！');
    return;
  }
  damageLoading.value = true
  try {
    // 2. 使用 apiService
    const response = await apiService.assessDamage(damageInput)
    Object.assign(damageOutput, response.data)
  } catch (error) {
    if (error.response?.status !== 401) {
      ElMessage.error('损伤评估计算失败！请检查输入值或后端服务。')
    }
    console.error(error)
  } finally {
    damageLoading.value = false
  }
}

const handleLinkAssess = async () => {
  if (!authStore.isAuthenticated) {
    ElMessage.error('请先登录再进行评估！');
    return;
  }
  linkLoading.value = true
  try {
    // 3. 使用 apiService
    const response = await apiService.assessLink(linkInput)
    Object.assign(linkOutput, response.data)
  } catch (error) {
    if (error.response?.status !== 401) {
      ElMessage.error('通信链路评估计算失败！请检查输入值或后端服务。')
    }
    console.error(error)
  } finally {
    linkLoading.value = false
  }
}

// --- 结果保存函数 ---
const saveResults = () => {
  let content = "评估结果报告\n\n";
  content += "--- 损伤评估 ---\n";
  content += `损耗 (Ls_dB): ${damageOutput.ls_db || 'N/A'}\n`;
  content += `到靶功率 (Pr_dBm): ${damageOutput.pr_dbm || 'N/A'}\n`;
  content += `评估等级: ${damageOutput.risk_level}\n\n`;

  content += "--- 通信链路 ---\n";
  content += `损耗 (Lp_dB): ${linkOutput.lp_db || 'N/A'}\n`;
  content += `链路余量 (dB): ${linkOutput.link_margin_db || 'N/A'}\n`;
  content += `通信状态: ${linkOutput.link_status}\n`;

  const blob = new Blob([content], {type: 'text/plain;charset=utf-8'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `评估结果_${new Date().toLocaleString()}.txt`;
  link.click();
  URL.revokeObjectURL(link.href);
  ElMessage.success('结果已保存到下载目录！');
}
</script>

<template>
  <div class="assessment-page">
    <el-row :gutter="20">
      <!-- 左侧：损伤评估 -->
      <el-col :span="12">
        <el-card shadow="never">
          <template #header><strong>损伤评估</strong></template>
          <div class="assessment-section">
            <el-card shadow="inner" class="input-card">
              <template #header><span>用户输入</span></template>
              <el-form :model="damageInput" label-width="120px">
                <el-form-item label="发射功率 (GW)">
                  <el-input v-model="damageInput.pt_gw" type="number"/>
                </el-form-item>
                <el-form-item label="发射增益 (dB)">
                  <el-input v-model="damageInput.gt_db" type="number"/>
                </el-form-item>
                <el-form-item label="接收增益 (dB)">
                  <el-input v-model="damageInput.gr_db" type="number"/>
                </el-form-item>
                <el-form-item label="频率 (GHz)">
                  <el-input v-model="damageInput.f_ghz" type="number"/>
                </el-form-item>
                <el-form-item label="距离 (km)">
                  <el-input v-model="damageInput.d_km" type="number"/>
                </el-form-item>
                <el-form-item label="低噪放增益 (dB)">
                  <el-input v-model="damageInput.lna_gain_db" type="number"/>
                </el-form-item>
              </el-form>
            </el-card>

            <el-card shadow="inner" class="output-card">
              <template #header><span>输出</span></template>
              <div class="output-item">
                <span>损耗 (Ls_dB):</span><strong>{{ damageOutput.ls_db ?? 'N/A' }}</strong></div>
              <div class="output-item">
                <span>到靶功率 (Pr_dBm):</span><strong>{{ damageOutput.pr_dbm ?? 'N/A' }}</strong>
              </div>
              <div class="output-item"><span>低噪放表增益 (dB):</span><strong>{{
                  damageOutput.lna_gain_db ?? 'N/A'
                }}</strong></div>
              <div class="output-item"><span>限幅器表插损 (dB):</span><strong>{{
                  damageOutput.limiter_loss_db ?? 'N/A'
                }}</strong></div>
            </el-card>

            <div class="action-bar">
              <el-button type="primary" @click="handleDamageAssess" :loading="damageLoading">
                低噪放评估
              </el-button>
              <div class="status-bar" :class="`status-${damageOutput.risk_level.toLowerCase()}`">
                {{ damageOutput.risk_level }}
              </div>
            </div>
          </div>
        </el-card>
      </el-col>

      <!-- 右侧：通信链路 -->
      <el-col :span="12">
        <el-card shadow="never">
          <template #header><strong>通信链路</strong></template>
          <div class="assessment-section">
            <el-card shadow="inner" class="input-card">
              <template #header><span>用户输入</span></template>
              <el-form :model="linkInput" label-width="140px">
                <el-form-item label="发射功率 (kW)">
                  <el-input v-model="linkInput.pt2_kw" type="number"/>
                </el-form-item>
                <el-form-item label="发射增益 (dB)">
                  <el-input v-model="linkInput.gt2_db" type="number"/>
                </el-form-item>
                <el-form-item label="接收增益 (dB)">
                  <el-input v-model="linkInput.gr2_db" type="number"/>
                </el-form-item>
                <el-form-item label="频率 (GHz)">
                  <el-input v-model="linkInput.f2_ghz" type="number"/>
                </el-form-item>
                <el-form-item label="地空距离 (km)">
                  <el-input v-model="linkInput.d2_km" type="number"/>
                </el-form-item>
                <el-form-item label="接收机灵敏度 (dBm)">
                  <el-input v-model="linkInput.receiver_sensitivity_dbm" type="number"/>
                </el-form-item>
              </el-form>
            </el-card>

            <el-card shadow="inner" class="output-card">
              <template #header><span>输出</span></template>
              <div class="output-item">
                <span>损耗 (Lp_dB):</span><strong>{{ linkOutput.lp_db ?? 'N/A' }}</strong></div>
              <div class="output-item">
                <span>链路余量 (dB):</span><strong>{{ linkOutput.link_margin_db ?? 'N/A' }}</strong>
              </div>
            </el-card>

            <div class="action-bar">
              <el-button type="success" @click="handleLinkAssess" :loading="linkLoading">
                评估通信状态
              </el-button>
              <div class="status-light-container">
                <span>通信状态:</span>
                <div class="status-light"
                     :class="`status-${linkOutput.link_status.toLowerCase()}`"></div>
              </div>
            </div>

            <div class="save-section">
              <el-button @click="saveResults" style="width: 100%;">保存结果</el-button>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.assessment-page {
  height: 100%;
}

.assessment-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.output-card .output-item {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px solid #ebeef5;
}

.output-card .output-item:last-child {
  border-bottom: none;
}

.action-bar {
  display: flex;
  align-items: center;
  gap: 20px;
}

.status-bar {
  flex-grow: 1;
  text-align: center;
  padding: 8px;
  color: white;
  font-weight: bold;
  border-radius: 4px;
  background-color: #909399; /* 未知 */
}

.status-bar.status-高危 {
  background-color: #f56c6c;
}

.status-bar.status-中危 {
  background-color: #e6a23c;
}

.status-bar.status-低危 {
  background-color: #67c23a;
}

.status-light-container {
  display: flex;
  align-items: center;
  gap: 10px;
}

.status-light {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background-color: #909399; /* 未知 */
  border: 2px solid #fff;
  box-shadow: 0 0 5px rgba(0, 0, 0, 0.2);
}

.status-light.status-正常 {
  background-color: #67c23a;
}

.status-light.status-中断 {
  background-color: #f56c6c;
}

.save-section {
  margin-top: 20px;
}
</style>
