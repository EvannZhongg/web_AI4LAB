# rag_retriever/views.py
import json
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from django.http import StreamingHttpResponse
from .serializers import RAGQuerySerializer
from .retriever import PgVectorRetriever

logger = logging.getLogger(__name__)

# --- 缓存 Retriever 实例 ---
# Django 启动时，每个 Gunicorn/uWSGI worker 进程会创建
# 一个自己的 retriever_instance 实例，并运行一次 __init__。
try:
    retriever_instance = PgVectorRetriever()
    logger.info("全局 PgVectorRetriever 实例已初始化。")
except Exception as e:
    retriever_instance = None
    logger.error(f"无法初始化全局 PgVectorRetriever: {e}", exc_info=True)


class RAGSearchAPIView(APIView):
    """
    用于 RAG 检索和答案生成的 API 视图。
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        global retriever_instance  # 允许我们重新初始化 (如果需要的话)

        if not retriever_instance:
            # (*** 健壮性修复 ***)
            # 如果启动时初始化失败，尝试在第一次请求时重新初始化
            try:
                retriever_instance = PgVectorRetriever()
                logger.info("RAGSearchAPIView: 已动态重新初始化 PgVectorRetriever。")
            except Exception as e:
                logger.error(f"动态初始化 PgVectorRetriever 失败: {e}", exc_info=True)
                return Response(
                    {"error": "RAG检索器未初始化，请检查服务器日志。"},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )

        serializer = RAGQuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        query = serializer.validated_data['query']
        top_k_chunks = serializer.validated_data['top_k_chunks']
        top_k_paths = serializer.validated_data['top_k_paths']
        mode = serializer.validated_data['mode']

        logger.info(f"RAGSearchAPIView 收到查询: {query}")

        try:
            # --- (*** 关键修复 ***) ---
            #
            # 移除了在每次请求时都调用的:
            # retriever_instance.adj_list, retriever_instance.edge_map = retriever_instance._build_adjacency_list()
            #
            # 我们现在依赖启动时加载的全局 retriever_instance.adj_list
            #
            # --- (*** 修复结束 ***) ---

            results, diagnostics = retriever_instance.search(
                query,
                top_k_chunks=top_k_chunks,
                top_k_paths=top_k_paths
            )

            # 2. 定义流式生成器
            def stream_generator():
                try:
                    # 首先，发送检索到的上下文（用于前端可视化）
                    context_data = {
                        "type": "context",
                        "data": {
                            "top_chunks": results["top_chunks"],
                            "top_paths": results["top_paths"],
                            "diagnostics": diagnostics
                        }
                    }
                    yield f"event: context\ndata: {json.dumps(context_data)}\n\n"

                    # 其次，流式传输LLM的答案
                    answer_stream = retriever_instance.generate_answer(
                        query,
                        results["top_chunks"],
                        results["top_paths"],
                        mode
                    )

                    for chunk in answer_stream:
                        if isinstance(chunk, str):  # 处理错误生成器
                            yield chunk
                            break

                        if chunk.choices:
                            content = chunk.choices[0].delta.content
                            if content:
                                # 发送LLM令牌
                                token_data = {"type": "token", "data": content}
                                yield f"event: token\ndata: {json.dumps(token_data)}\n\n"

                    # 发送流结束信号
                    yield "event: end\ndata: {}\n\n"

                except Exception as e_stream:
                    logger.error(f"在流生成期间发生错误: {e_stream}", exc_info=True)
                    error_data = {"type": "error", "data": f"流生成失败: {e_stream}"}
                    yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

            # 返回流式响应
            response = StreamingHttpResponse(stream_generator(), content_type='text/event-stream')
            response['X-Accel-Buffering'] = 'no'  # 禁用 Nginx 缓冲
            return response

        except Exception as e:
            logger.error(f"RAG 检索阶段失败: {e}", exc_info=True)
            return Response(
                {"error": f"RAG 检索失败: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )