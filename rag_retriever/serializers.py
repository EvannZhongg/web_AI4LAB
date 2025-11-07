# rag_retriever/serializers.py
from rest_framework import serializers

class RAGQuerySerializer(serializers.Serializer):
    query = serializers.CharField(required=True, allow_blank=False)
    # top_k_chunks, top_k_paths 等可以作为可选参数添加
    top_k_chunks = serializers.IntegerField(default=5, min_value=1, max_value=20)
    top_k_paths = serializers.IntegerField(default=5, min_value=1, max_value=20)
    mode = serializers.ChoiceField(
        choices=["full_context", "chunks_only", "paths_only"],
        default="full_context"
    )