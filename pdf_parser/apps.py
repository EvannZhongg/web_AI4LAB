# pdf_parser/apps.py

from django.apps import AppConfig

class PdfParserConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "pdf_parser"

    def ready(self):
        """
        在应用准备就绪时导入信号。
        """
        try:
            import pdf_parser.signals
        except ImportError:
            pass