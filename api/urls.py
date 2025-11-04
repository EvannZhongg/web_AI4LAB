# api/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    UserCreateView, DeviceViewSet, ExperimentViewSet, DamageAssessmentView,
    LinkAssessmentView, SystemFailureProbabilityView, ProbabilityDataSetViewSet,
    DeviceComparisonView, ProfileView
)
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

router = DefaultRouter()
router.register(r'devices', DeviceViewSet, basename='device')
router.register(r'experiments', ExperimentViewSet, basename='experiment') # Router handles create, list, retrieve, update, destroy, grid_data, csv_metadata, upload_csv
router.register(r'probability-datasets', ProbabilityDataSetViewSet, basename='probabilitydataset')

urlpatterns = [
    # --- 手动添加 GET CSV 数据的 URL 模式 ---
    # 匹配 /api/experiments/<experiment_pk>/csv_data/<metadata_id>/
    path(
        'experiments/<int:pk>/csv_data/<int:metadata_id>/',
        ExperimentViewSet.as_view({'get': 'get_csv_data'}), # 将 GET 请求映射到 get_csv_data 方法
        name='experiment-get-csv-data' # 给 URL 起个名字 (可选但推荐)
    ),

    # --- Router 生成的 URLs ---
    # Router 会处理 /api/experiments/, /api/experiments/{pk}/,
    # 以及 upload_csv, grid_data, csv_metadata actions
    path('', include(router.urls)),

    # --- 其他 URLs (保持不变) ---
    path('register/', UserCreateView.as_view(), name='user_register'),
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('assess/damage/', DamageAssessmentView.as_view(), name='assess_damage'),
    path('assess/link/', LinkAssessmentView.as_view(), name='assess_link'),
    path('probability/calculate/', SystemFailureProbabilityView.as_view(), name='calculate_probability'),
    path('compare/', DeviceComparisonView.as_view(), name='device_comparison'),
    path('profile/', ProfileView.as_view(), name='user_profile'),
]