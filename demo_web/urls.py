from django.contrib import admin
from django.urls import path, include
from django.conf import settings               # <--- 添加导入 settings
from django.conf.urls.static import static     # <--- 添加导入 static

urlpatterns = [
    path("admin/", admin.site.urls),
    path('api/', include('api.urls')),
    path('api/pdf_parser/', include('pdf_parser.urls')),
    path('api/rag/', include('rag_retriever.urls')),
]

# 仅在 DEBUG 模式下生效
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)