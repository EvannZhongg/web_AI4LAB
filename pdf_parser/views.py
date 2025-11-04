from rest_framework import generics, permissions, status, viewsets
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .models import PDFParsingTask
from .serializers import PDFParsingTaskSerializer
from .tasks import task_pipeline
from django.db import transaction

class PDFTaskViewSet(viewsets.ModelViewSet):
    """
    用于管理 PDF 解析任务的视图集。
    - POST /: 上传新的 PDF 并创建任务
    - GET /: 获取当前用户的所有任务列表
    - GET /<id>/: 获取特定任务的状态
    """
    serializer_class = PDFParsingTaskSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser] # 支持文件上传

    def get_queryset(self):
        # 用户只能看到自己的任务
        return PDFParsingTask.objects.filter(user=self.request.user)

    def create(self, request, *args, **kwargs):
        pdf_file = request.FILES.get('pdf_file')
        if not pdf_file:
            return Response({'error': '未找到 "pdf_file" 字段'}, status=status.HTTP_400_BAD_REQUEST)

        # 1. 创建任务实例
        task = PDFParsingTask(
            user=request.user,
            pdf_file=pdf_file,
            status=PDFParsingTask.Status.PENDING
        )
        task.save()  # 保存以获取 ID

        # 2. 启动 Celery 流水线
        try:
            # <--- 修改：调用新的编排器任务 ---
            transaction.on_commit(lambda: task_pipeline.delay(task.id))
        except Exception as e:
            # (异常处理不变)
            task.status = PDFParsingTask.Status.FAILED
            task.error_message = f"无法启动后台任务: {e}"
            task.save()
            return Response(
                {'error': '无法启动后台处理任务，请联系管理员。'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 3. 立即返回 (不变)
        serializer = self.get_serializer(task)
        return Response(serializer.data, status=status.HTTP_202_ACCEPTED)

    # 禁用 PUT 和 PATCH
    def update(self, request, *args, **kwargs):
        return Response(status=status.HTTP_405_METHOD_NOT_ALLOWED)

    def partial_update(self, request, *args, **kwargs):
        return Response(status=status.HTTP_405_METHOD_NOT_ALLOWED)

    # 允许删除任务
    # destroy 方法 (DELETE /<id>/) 已由 ModelViewSet 默认提供