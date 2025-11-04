# api/serializers.py
from rest_framework import serializers
from django.db import transaction
from .models import Device, ProbabilityDataSet, Profile, Experiment, Parameter, DataPoint, CsvFile, safe_float # 导入 CsvFile
from django.contrib.auth.models import User

# --- CsvFile Serializer (修正！) ---
class CsvFileSerializer(serializers.ModelSerializer):
    # 只读字段，显示文件名
    # 移除 source='filename'
    filename = serializers.CharField(read_only=True)
    file_url = serializers.SerializerMethodField()

    class Meta:
        model = CsvFile
        # experiment 字段在创建时不应包含，在读取时应是只读
        fields = ['id', 'experiment', 'metadata_id', 'filename', 'uploaded_at', 'file_url']
        read_only_fields = ['id', 'experiment', 'uploaded_at', 'filename', 'file_url']
        # 在 Meta 中为 write 操作指定字段（如果需要创建 CsvFile 实例）
        # 但在 upload_csv action 中我们是手动创建实例，所以这里不是必须的
        # write_only_fields = ['file', 'metadata_id'] # file 字段实际在 request.FILES 中

    def get_file_url(self, obj):
        # 返回文件的访问 URL (保持不变)
        if obj.file:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.file.url)
            return obj.file.url # Fallback
        return None

# --- DataPoint & Parameter Serializers (保持不变) ---
class DataPointSerializer(serializers.ModelSerializer):
    class Meta: model = DataPoint; fields = ['row_index', 'value_text', 'value_numeric']

class ParameterSerializer(serializers.ModelSerializer):
    datapoints = DataPointSerializer(many=True, read_only=True)
    class Meta: model = Parameter; fields = ['id', 'name', 'column_index', 'unit', 'datapoints']; read_only_fields = ['id']

# --- Experiment Serializer (修改) ---
class ExperimentSerializer(serializers.ModelSerializer):
    parameters = ParameterSerializer(many=True, read_only=True)
    device = serializers.PrimaryKeyRelatedField(queryset=Device.objects.all(), write_only=False, required=False)
    # 嵌套 CsvFileSerializer 用于读取 (可选，取决于是否需要在 Experiment 详情中看到关联文件列表)
    # csv_files = CsvFileSerializer(many=True, read_only=True)

    class Meta:
        model = Experiment
        # 保持 fields 不变，csv_files_metadata 仍然用于前端交互
        fields = ['id', 'device', 'name', 'experiment_type', 'parameters', 'csv_files_metadata', 'created_at']
        read_only_fields = ['id', 'created_at', 'parameters']


# --- UserSerializer (保持不变) ---
class UserSerializer(serializers.ModelSerializer):
    class Meta: model = User; fields = ['username', 'password']; extra_kwargs = {'password': {'write_only': True}}
    def create(self, vd): return User.objects.create_user(**vd)


# --- Device Serializers (保持不变) ---
class DeviceListSerializer(serializers.ModelSerializer):
    test_types_display = serializers.SerializerMethodField()
    class Meta: model = Device; fields = ['id', 'name', 'device_type', 'substrate', 'device_number', 'tech_description', 'created_at', 'test_types_display']
    def get_test_types_display(self, obj): return getattr(obj, '_test_types', None) or ""

class DeviceSerializer(serializers.ModelSerializer):
    experiments = ExperimentSerializer(many=True, read_only=True)
    class Meta: model = Device; fields = ['id', 'name', 'device_type', 'substrate', 'device_number', 'tech_description', 'photo_data', 'created_at', 'experiments']

# --- ProbabilityDataSetSerializer (保持不变) ---
class ProbabilityDataSetSerializer(serializers.ModelSerializer):
     class Meta: model = ProbabilityDataSet; fields = '__all__'

# --- ProfileSerializer (保持不变) ---
class ProfileSerializer(serializers.ModelSerializer):
     class Meta: model = Profile; fields = ['user_configs', 'active_config_id']


# --- Grid Data 更新序列化器 (保持不变) ---
class ParameterInputSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=100, allow_blank=False); column_index = serializers.IntegerField(min_value=0); unit = serializers.CharField(max_length=50, allow_blank=True, allow_null=True, required=False)

class DataPointInputSerializer(serializers.Serializer):
    parameter_col_index = serializers.IntegerField(min_value=0); row_index = serializers.IntegerField(min_value=0); value_text = serializers.CharField(allow_blank=True, allow_null=True, required=False)

class GridDataUpdateSerializer(serializers.Serializer):
    parameters = ParameterInputSerializer(many=True); datapoints = DataPointInputSerializer(many=True)

# --- CsvMetadata 更新序列化器 (保持不变) ---
class CsvMetadataUpdateSerializer(serializers.ModelSerializer):
    class Meta: model = Experiment; fields = ['csv_files_metadata']