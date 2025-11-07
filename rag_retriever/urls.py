# rag_retriever/urls.py
from django.urls import path
from .views import RAGSearchAPIView

urlpatterns = [
    path('search/', RAGSearchAPIView.as_view(), name='rag-search'),
]