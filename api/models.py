# api/models.py
import os # 导入 os 模块
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save, pre_delete # 导入 pre_delete
from django.dispatch import receiver
from django.conf import settings # 导入 settings

# --- Helper function for upload path ---
def experiment_csv_path(instance, filename):
    # 文件将上传到 MEDIA_ROOT/experiment_csvs/<experiment_id>/<filename>
    # 确保 experiment_id 存在
    exp_id = instance.experiment.id if instance.experiment else 'misc'
    return f'experiment_csvs/{exp_id}/{filename}'


class Device(models.Model):
    # ... (name 字段不变) ...
    name = models.CharField(max_length=100, verbose_name="器件名称")

    # (*** 关键修复：允许手动创建时为空 ***)
    device_type = models.CharField(max_length=50, verbose_name="器件类型", db_index=True, blank=True, null=True)
    substrate = models.CharField(max_length=50, verbose_name="衬底材料", blank=True, null=True)
    # (*** 修复结束 ***)

    device_number = models.CharField(max_length=100, unique=True, verbose_name="器件编号")
    tech_description = models.TextField(blank=True, null=True, verbose_name="技术说明")
    photo_data = models.TextField(blank=True, null=True, verbose_name="微观照片 (Base64)")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    def __str__(self):
        return self.device_number

    class Meta:
        verbose_name = "器件"
        verbose_name_plural = verbose_name
        ordering = ['-created_at']

class Experiment(models.Model):
    # ... (保持不变) ...
    device = models.ForeignKey(Device, related_name='experiments', on_delete=models.CASCADE, verbose_name="所属器件")
    name = models.CharField(max_length=200, verbose_name="实验表格名称")
    experiment_type = models.CharField(max_length=100, verbose_name="实验类型", db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # 修改 csv_files_metadata 结构:
    # 示例: [{'metadata_id': <frontend_temp_id>, 'csv_file_id': <CsvFile_id>, 'filename': '...', 'columns': [...]}, ...]
    csv_files_metadata = models.JSONField(default=list, blank=True, verbose_name="关联CSV元数据")

    def __str__(self):
        return f"{self.device.device_number} - {self.name} ({self.experiment_type})"

    class Meta:
        verbose_name = "实验表格"
        verbose_name_plural = verbose_name
        ordering = ['device__device_number', 'name']


# --- 新增：CsvFile 模型 ---
class CsvFile(models.Model):
    """存储与实验关联的 CSV 文件"""
    experiment = models.ForeignKey(Experiment, related_name='csv_files', on_delete=models.CASCADE, verbose_name="所属实验")
    # 使用 FileField 存储文件，指定上传路径
    file = models.FileField(upload_to=experiment_csv_path, verbose_name="CSV 文件")
    # metadata_id 用于前端追踪，确保与 csv_files_metadata 中的记录对应
    metadata_id = models.BigIntegerField(verbose_name="元数据ID (前端生成)", db_index=True, unique=True) # 确保唯一性
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"CSV for Exp {self.experiment.id} - {os.path.basename(self.file.name)} (MetaID: {self.metadata_id})"

    def filename(self):
        return os.path.basename(self.file.name)

    class Meta:
        verbose_name = "CSV 文件"
        verbose_name_plural = verbose_name
        ordering = ['experiment', 'uploaded_at']

# --- Signal: 删除 CsvFile 对象时，同时删除对应的物理文件 ---
@receiver(pre_delete, sender=CsvFile)
def delete_csv_file(sender, instance, **kwargs):
    # 确保文件存在且不为空
    if instance.file:
        # 构建文件的完整路径
        file_path = os.path.join(settings.MEDIA_ROOT, instance.file.name)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted physical file: {file_path}")
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")

class Parameter(models.Model):
    # ... (保持不变) ...
    experiment = models.ForeignKey(Experiment, related_name='parameters', on_delete=models.CASCADE, verbose_name="所属实验")
    name = models.CharField(max_length=100, verbose_name="参数名称", db_index=True)
    column_index = models.PositiveIntegerField(verbose_name="原始列索引")
    unit = models.CharField(max_length=50, blank=True, null=True, verbose_name="单位")

    def __str__(self):
        return f"{self.experiment.name} - {self.name}"

    class Meta:
        verbose_name = "实验参数"
        verbose_name_plural = verbose_name
        unique_together = ('experiment', 'name')
        ordering = ['experiment', 'column_index']

class DataPoint(models.Model):
    # ... (保持不变) ...
    parameter = models.ForeignKey(Parameter, related_name='datapoints', on_delete=models.CASCADE, verbose_name="所属参数")
    row_index = models.PositiveIntegerField(verbose_name="原始行索引")
    value_text = models.TextField(blank=True, null=True, verbose_name="文本值")
    value_numeric = models.FloatField(blank=True, null=True, verbose_name="数值", db_index=True)

    def __str__(self):
        value = self.value_text if self.value_text is not None else self.value_numeric
        return f"Row {self.row_index}, Param {self.parameter.name}: {value}"

    class Meta:
        verbose_name = "数据点"
        verbose_name_plural = verbose_name
        unique_together = ('parameter', 'row_index')
        ordering = ['parameter__experiment', 'row_index', 'parameter__column_index']


# --- ProbabilityDataSet 和 Profile (保持不变) ---
class ProbabilityDataSet(models.Model):
    name = models.CharField(max_length=100, unique=True, verbose_name="数据集名称")
    data = models.JSONField(verbose_name="概率数据")
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self): return self.name
    class Meta:
        verbose_name = "概率数据集"; verbose_name_plural = verbose_name; ordering = ['-created_at']

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    user_configs = models.JSONField(default=list, blank=True, verbose_name="用户自定义API配置列表")
    active_config_id = models.CharField(max_length=50, default='default', blank=True, verbose_name="当前激活的配置ID")
    def __str__(self): return f'{self.user.username} Profile'

# --- Signals (保持不变) ---
@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created: Profile.objects.create(user=instance)

# --- 辅助函数 (保持不变) ---
def safe_float(value):
    try:
        if isinstance(value, str) and not value.strip(): return None
        return float(value)
    except (ValueError, TypeError, AttributeError): return None