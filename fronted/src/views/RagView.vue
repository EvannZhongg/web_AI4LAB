<script setup>
import { ref, reactive, nextTick, computed } from 'vue';
import { useAuthStore } from '@/stores/auth';
import { ElMessage } from 'element-plus';
import GraphVisualizer from '@/components/GraphVisualizer.vue';
import MarkdownIt from 'markdown-it';

const authStore = useAuthStore();
const query = ref('');
const isLoading = ref(false);
const messages = ref([]);
const retrievalContext = reactive({
  top_chunks: [],
  top_paths: [],
  diagnostics: {}
});

let abortController = null;
const chatScrollbar = ref(null);
const contextScrollbar = ref(null);
const pathViewMode = ref('text');

const currentTaskBasePath = ref(''); // 例如: 'md_results/51'
const md = new MarkdownIt();

// 存储默认的图像渲染规则
const defaultImageRenderer = md.renderer.rules.image || function(tokens, idx, options, env, self) {
  return self.renderToken(tokens, idx, options);
};

// --- (*** 关键修复：重写图像渲染规则 ***) ---
md.renderer.rules.image = (tokens, idx, options, env, self) => {
  const token = tokens[idx];
  let src = token.attrGet('src');

  // 1. 跳过绝对 URL (http, data) 或空的 src
  if (!src || src.startsWith('http') || src.startsWith('data:')) {
    return defaultImageRenderer(tokens, idx, options, env, self);
  }

  // Django 的 MEDIA_URL (从 .env 读取, 确保末尾有 /)
  const VITE_MEDIA_URL = (import.meta.env.VITE_MEDIA_URL || '/media/').replace(/\/+$/, '') + '/';
  let newSrc = '';

  // 1. 规范化 src (来自 LLM)
  let cleanSrc = src.replace(/^\.\//, '').replace(/\\/g, '/'); // 替换反斜杠

  // --- (*** 逻辑重排 ***) ---

  // 2. 检查是否是绝对本地文件路径 (e.g., D:/... or file:///...)
  // 必须在检查 !cleanSrc.startsWith('/') 之前
  if (cleanSrc.startsWith('file://') || cleanSrc.match(/^[a-zA-Z]:\//)) {
      const mediaMarker = '/media/';
      // 路径可能被 URI 编码 (例如 %5C 替换 \)
      const decodedSrc = decodeURIComponent(cleanSrc);
      const normalizedPath = decodedSrc.replace(/\\/g, '/');

      const mediaIndex = normalizedPath.toLowerCase().indexOf(mediaMarker.toLowerCase());

      if (mediaIndex !== -1) {
          // 截取 "media/" 之后的所有内容
          const relativePath = normalizedPath.substring(mediaIndex + mediaMarker.length);
          newSrc = `${VITE_MEDIA_URL}${relativePath}`; // e.g., /media/md_results/60/...
      } else {
          // 无法智能转换，阻止它
          console.warn(`Blocking local file path (could not find 'media/'): ${cleanSrc}`);
          return `[图像本地路径被阻止: ${token.content}]`;
      }
  }

  // 3. 检查是否是服务器相对路径 (由 get_path_details 正确生成)
  //    (例如: md_results/60/image/...)
  else if (cleanSrc.startsWith('md_results/')) {
    newSrc = `${VITE_MEDIA_URL}${cleanSrc}`;
  }

  // 4. 检查是否是 chunk 相对路径 (由 LLM 从 chunk 文本中提取)
  //    (例如: image/page_25.png)
  else if (!cleanSrc.startsWith('/')) {
    const basePath = (currentTaskBasePath.value || '').replace(/\\/g, '/').replace(/\/+$/, '');

    if (basePath) {
        // 最终路径: /media/ + md_results/54 + / + image/page_25.png
        newSrc = `${VITE_MEDIA_URL}${basePath}/${cleanSrc}`;
    } else {
        // Fallback: 无法确定 base path，可能无法加载
        console.warn(`currentTaskBasePath is not set. Image src "${cleanSrc}" may not load correctly.`);
        newSrc = cleanSrc; // 保持原样 (e.g., 'image/page_8...')
    }
  }

  // 5. 回退 (例如 /static/...)
  else {
    newSrc = cleanSrc;
  }
  // --- (*** 逻辑重排结束 ***) ---

  token.attrSet('src', newSrc);
  console.log(`Rewriting image src from '${src}' to '${newSrc}'`);

  return defaultImageRenderer(tokens, idx, options, env, self);
};
// --- (*** 修复结束 ***) ---


const renderMarkdown = (msg, index) => {
  let content = msg.content;
  if (isLoading.value && index === messages.value.length - 1) {
    if (!content.endsWith('▍')) {
        content += '▍';
    }
  }
  // 确保在渲染时，currentTaskBasePath.value 是最新的
  return md.render(content);
};

// (计算属性不变)
const mergedGraphData = computed(() => {
  const allNodes = new Map();
  const allEdges = new Map();
  if (!retrievalContext.top_paths) {
    return { nodes: [], edges: [] };
  }
  for (const path of retrievalContext.top_paths) {
    if (path.graph_data) {
      for (const node of path.graph_data.nodes) {
        if (!allNodes.has(node.id)) {
          allNodes.set(node.id, node);
        }
      }
      for (const edge of path.graph_data.edges) {
        const edgeKey = `${edge.source}-${edge.target}-${edge.label}`;
        if (!allEdges.has(edgeKey)) {
          allEdges.set(edgeKey, edge);
        }
      }
    }
  }
  return {
    nodes: Array.from(allNodes.values()),
    edges: Array.from(allEdges.values()),
  };
});

// (滚动函数不变)
const scrollToBottom = (scrollbarRef) => {
  nextTick(() => {
    if (scrollbarRef && scrollbarRef.value && scrollbarRef.value.wrapRef) {
      scrollbarRef.value.wrapRef.scrollTop = scrollbarRef.value.wrapRef.scrollHeight;
    }
  });
};

// (停止函数不变)
const handleStop = () => {
  if (abortController) {
    abortController.abort();
    isLoading.value = false;
    abortController = null;
  }
};

// (提交函数不变)
const handleSubmitQuery = async () => {
  if (!query.value.trim() || isLoading.value) return;
  if (!authStore.isAuthenticated) {
    ElMessage.error('请先登录以使用RAG检索功能。');
    return;
  }

  const userQuery = query.value;
  messages.value.push({ role: 'user', content: userQuery });

  isLoading.value = true;
  query.value = '';

  Object.assign(retrievalContext, { top_chunks: [], top_paths: [], diagnostics: {} });
  currentTaskBasePath.value = ''; // <--- 重置 base path
  const botMessage = reactive({ role: 'bot', content: '' });
  messages.value.push(botMessage);
  scrollToBottom(chatScrollbar);

  abortController = new AbortController();

  try {
    const VITE_API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api/';
    const url = `${VITE_API_BASE_URL}rag/search/`;

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authStore.token}`,
        'ngrok-skip-browser-warning': 'true'
      },
      body: JSON.stringify({ query: userQuery }),
      signal: abortController.signal
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: `HTTP error! status: ${response.status}` }));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }
    if (!response.body) {
      throw new Error("Response body is null");
    }

    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        isLoading.value = false;
        abortController = null;
        console.log("Stream finished.");
        break;
      }

      buffer += value;
      let boundary = buffer.indexOf('\n\n');

      while(boundary !== -1) {
        const message = buffer.substring(0, boundary);
        buffer = buffer.substring(boundary + 2);

        let event = 'message';
        let dataLines = [];

        const lines = message.split('\n');
        for (const line of lines) {
          if (line.startsWith('event:')) {
            event = line.substring(6).trim();
          } else if (line.startsWith('data:')) {
            dataLines.push(line.substring(5).trim());
          }
        }

        const data = dataLines.join('\n');

        if (event === 'context') {
          console.log("RAG Context received:", data);
          try {
            const context = JSON.parse(data);
            Object.assign(retrievalContext, context.data);

            if (context.data.top_chunks && context.data.top_chunks.length > 0) {
              if (context.data.top_chunks[0].output_directory) {
                currentTaskBasePath.value = context.data.top_chunks[0].output_directory;
                console.log('Set currentTaskBasePath to:', currentTaskBasePath.value);
              } else {
                 console.warn("Context received, but top_chunk[0].output_directory is missing or null.");
              }
            } else {
              console.warn("Context received, but no top_chunks found to set image base path.");
            }

            scrollToBottom(contextScrollbar);
          } catch(e) { console.error("Failed to parse context JSON:", e, data); }
        } else if (event === 'token') {
          try {
            const token = JSON.parse(data);
            if (token.data) {
              botMessage.content += token.data;
              scrollToBottom(chatScrollbar);
            }
          } catch(e) { console.error("Failed to parse token JSON:", e, data); }
        } else if (event === 'end') {
          console.log("Stream ended by server.");
          isLoading.value = false;
          reader.releaseLock();
          abortController = null;
          return;
        } else if (event === 'error') {
          try {
            const error = JSON.parse(data);
            throw new Error(error.data || 'Stream returned an error');
          } catch(e) { throw new Error(data || 'Stream returned an unknown error'); }
        }
        boundary = buffer.indexOf('\n\n');
      }
    }
  } catch (error) {
    if (error.name === 'AbortError') {
      console.log('Stream fetch aborted by user.');
      botMessage.content += '\n\n[检索已停止]';
    } else {
      console.error("Failed to connect to RAG stream:", error);
      ElMessage.error(`检索失败: ${error.message}`);
      botMessage.content = (botMessage.content || '') + `\n\n[检索失败: ${error.message}]`;
    }
    isLoading.value = false;
    abortController = null;
  }
};
</script>

<template>
  <div class="rag-page">
    <div class="main-content">

      <div class="context-area">
        <el-scrollbar class="context-scrollbar" ref="contextScrollbar">
          <div class="context-content">
            <el-card shadow="never">
              <template #header><strong>检索诊断</strong></template>
              <div v-if="retrievalContext.diagnostics.time_total_retrieval" class="diag-tags">
                <el-tag type="info" size="small">总检索: {{ retrievalContext.diagnostics.time_total_retrieval }}</el-tag>
                <el-tag type="info" size="small">阶段1: {{ retrievalContext.diagnostics.time_stage1_retrieval }}</el-tag>
                <el-tag type="info" size="small">阶段2: {{ retrievalContext.diagnostics.time_stage2_fusion }}</el-tag>
                <el-tag type="info" size="small">阶段3: {{ retrievalContext.diagnostics.time_stage3_ranking }}</el-tag>
              </div>
              <el-empty v-else description="尚未检索" :image-size="50" />
            </el-card>

            <el-card shadow="never">
              <template #header>
                <div class="card-header">
                  <strong>{{ `知识图谱路径 (Top ${retrievalContext.top_paths.length})` }}</strong>
                  <el-radio-group v-model="pathViewMode" size="small">
                    <el-radio-button label="text">文本</el-radio-button>
                    <el-radio-button label="graph">图谱</el-radio-button>
                  </el-radio-group>
                </div>
              </template>

              <div v-if="pathViewMode === 'text'">
                <div v-if="retrievalContext.top_paths.length > 0">
                  <div v-for="path in retrievalContext.top_paths" :key="path.path_readable" class="context-item">
                    <strong>路径:</strong> {{ path.path_readable }}
                    <br/>
                    <el-tag type="success" size="small">得分: {{ path.score.toFixed(3) }}</el-tag>
                    <pre class="context-detail">原因: {{ path.reason }}</pre>
                  </div>
                </div>
                <el-empty v-else description="未找到相关图谱路径" :image-size="50" />
              </div>

              <div v-else>
                <div v-if="retrievalContext.top_paths.length > 0">
                  <GraphVisualizer :graphData="mergedGraphData" />
                </div>
                <el-empty v-else description="无图谱数据可供可视化" :image-size="50" />
              </div>
            </el-card>

            <el-card shadow="never">
              <template #header><strong>{{ `文本证据 (Top ${retrievalContext.top_chunks.length})` }}</strong></template>
               <div v-if="retrievalContext.top_chunks.length > 0">
                <div v-for="chunk in retrievalContext.top_chunks" :key="chunk.id" class="context-item">
                  <strong>来源:</strong> {{ chunk.source_document }}
                  <br/>
                  <el-tag type="warning" size="small">得分: {{ (chunk.score || chunk.final_score).toFixed(3) }}</el-tag>
                  <pre class="context-detail">{{ chunk.content.substring(0, 150) }}...</pre>
                </div>
              </div>
              <el-empty v-else description="未找到相关文本证据" :image-size="50" />
            </el-card>
          </div>
        </el-scrollbar>
      </div>

      <div class="chat-area">
        <el-scrollbar class="chat-scrollbar" ref="chatScrollbar">
          <div class="chat-messages">
            <div v-if="messages.length === 0" class="empty-chat">
              <el-empty description="请输入您的问题以开始检索" />
            </div>

            <div v-for="(msg, index) in messages" :key="index"
                 class="message-bubble-wrapper"
                 :class="msg.role === 'user' ? 'user-wrapper' : 'bot-wrapper'">
              <div class="message-bubble" :class="msg.role">
                <span v-if="msg.role === 'user'"><strong>您：</strong></span>
                <span v-if="msg.role === 'bot'"><strong>AI：</strong></span>

                <div v-if="msg.role === 'user'" class="message-content">
                  <pre>{{ msg.content }}</pre>
                </div>

                <div v-else class="message-content bot-message-html" v-html="renderMarkdown(msg, index)"></div>
              </div>
            </div>
          </div>
        </el-scrollbar>

        <div class="input-area">
          <el-input
            v-model="query"
            placeholder="在此输入您的问题..."
            size="large"
            @keydown.enter.prevent="handleSubmitQuery"
            :disabled="isLoading"
          >
            <template #append>
              <el-button @click="handleSubmitQuery" :loading="isLoading" v-if="!isLoading">
                发送
              </el-button>
              <el-button @click="handleStop" type="danger" v-if="isLoading">
                停止
              </el-button>
            </template>
          </el-input>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.rag-page {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  background: #fff;
}
.main-content {
  flex-grow: 1;
  display: flex;
  min-height: 0;
}

/* (*** 关键修复：修改 Flex 比例 ***) */
.context-area {
  flex: 0.8; /* 左侧面板 (33.3%) */
  border-right: 1px solid #dcdfe6;
  display: flex;
  flex-direction: column;
  background-color: #f9fafb;
}
.chat-area {
  flex: 2; /* 右侧面板 (66.7%) */
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}
/* (*** 修复结束 ***) */

.chat-scrollbar, .context-scrollbar {
  flex-grow: 1;
}
.chat-messages, .context-content {
  padding: 20px;
}
.context-content {
  display: flex;
  flex-direction: column;
  gap: 15px;
}
.message-bubble-wrapper {
  display: flex;
  width: 100%;
  margin-bottom: 15px;
}
.message-bubble {
  padding: 10px 15px;
  border-radius: 10px;
  max-width: 90%;
  box-sizing: border-box;
}

/* 用户 (靠右) */
.message-bubble-wrapper.user-wrapper {
  justify-content: flex-end;
}
.message-bubble.user {
  background-color: #ecf5ff;
  border-bottom-right-radius: 0;
}

/* 机器人 (靠左) */
.message-bubble-wrapper.bot-wrapper {
  justify-content: flex-start;
}
.message-bubble.bot {
  background-color: #f0f9eb;
  border-bottom-left-radius: 0;
}

.input-area {
  padding: 15px 20px;
  border-top: 1px solid #dcdfe6;
  background: #ffffff;
  flex-shrink: 0;
}

.message-content {
  word-wrap: break-word;
  font-family: inherit;
  font-size: 14px;
  margin: 0;
}
.message-content pre {
  white-space: pre-wrap;
  font-family: inherit;
  margin: 0;
}
.message-content.bot-message-html {
  white-space: normal;
}
.message-content.bot-message-html :deep(pre) {
  background-color: #f4f4f4;
  padding: 10px;
  border-radius: 5px;
  overflow-x: auto;
  white-space: pre;
}
.message-content.bot-message-html :deep(code) {
  font-family: 'Courier New', Courier, monospace;
  background-color: #f4f4f4;
  padding: 2px 4px;
  border-radius: 3px;
}
.message-content.bot-message-html :deep(pre) > :deep(code) {
  background-color: transparent;
  padding: 0;
}
.message-content.bot-message-html :deep(table) {
  border-collapse: collapse;
  width: auto;
  margin: 10px 0;
}
.message-content.bot-message-html :deep(th),
.message-content.bot-message-html :deep(td) {
  border: 1px solid #ddd;
  padding: 8px;
}
.message-content.bot-message-html :deep(th) {
  background-color: #f2f2f2;
}
.message-content.bot-message-html :deep(ul),
.message-content.bot-message-html :deep(ol) {
  padding-left: 20px;
}
.message-content.bot-message-html :deep(img) {
  max-width: 100%;
  height: auto;
  border-radius: 5px;
}
.empty-chat {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  min-height: 300px;
}
.context-item {
  margin-bottom: 10px;
  padding-bottom: 10px;
  border-bottom: 1px dashed #e4e7ed;
  font-size: 13px;
}
.context-item:last-child {
  border-bottom: none;
}
.context-detail {
  font-size: 12px;
  color: #606266;
  background-color: #fafafa;
  padding: 5px;
  border-radius: 4px;
  white-space: pre-wrap;
  word-wrap: break-word;
  max-height: 100px;
  overflow-y: auto;
  margin-top: 5px;
}
.diag-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
