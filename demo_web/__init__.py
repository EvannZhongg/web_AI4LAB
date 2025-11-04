# evannzhongg/ai4mw_web/AI4MW_Web-b75f2e933ce5eb3d7c9b77393d2d6eec787f7611/demo_web/__init__.py

# 确保 Celery app 在 Django 启动时被加载
from .celery import app as celery_app

__all__ = ('celery_app',)