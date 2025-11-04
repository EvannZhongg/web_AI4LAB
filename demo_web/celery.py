# evannzhongg/ai4mw_web/AI4MW_Web-b75f2e933ce5eb3d7c9b77393d2d6eec787f7611/demo_web/celery.py
import os
from celery import Celery

# 设置 Django settings 模块
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo_web.settings')

app = Celery('demo_web')

# 使用 Django settings.py 作为 Celery 配置
app.config_from_object('django.conf:settings', namespace='CELERY')

# 自动发现所有 Django app 下的 tasks.py 文件
app.autodiscover_tasks()