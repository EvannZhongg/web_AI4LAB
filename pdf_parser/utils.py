import sys
from pathlib import Path
import hashlib
import yaml
import logging
from django.conf import settings # 导入 Django settings

# 导入 docling 相关的模块
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling_core.types.doc import PictureItem, TextItem, SectionHeaderItem, TableItem
    from docling.datamodel.pipeline_options import EasyOcrOptions
    from PIL import Image
except ImportError as e:
    logging.error(f"Docling或Pillow库导入失败: {e}")
    logging.error("请确保在 Django 环境中已安装 'docling-cpu' (或 'docling-gpu') 和 'Pillow'")
    # 抛出异常以便 Celery 捕获
    raise ImportError("Docling 依赖库未安装")

# 使用 Django 的日志系统
logger = logging.getLogger(__name__)

def convert_pdf_to_markdown_with_images(
        pdf_path: Path,
        model_dir: Path,
        output_dir: Path,
        use_ocr: bool = False,
        save_pages: bool = False,
        page_dpi: int = 200
) -> Path:
    """
    转换 PDF 的核心功能函数。
    """
    if not pdf_path.is_file():
        logger.error(f"❌ 错误：找不到输入文件 '{pdf_path}'")
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    absolute_model_path = model_dir.resolve()
    logger.info(f"📦 模型将使用此路径: {absolute_model_path}")
    logger.info(f"📂 输出内容将保存至: {output_dir.resolve()}")

    pipeline_options = PdfPipelineOptions(artifacts_path=str(absolute_model_path))
    pipeline_options.do_ocr = use_ocr
    if use_ocr:
        logger.info("⚙️ 模式: 启用强制全页 OCR (EasyOCR)")
        ocr_options = EasyOcrOptions(force_full_page_ocr=True)
        pipeline_options.ocr_options = ocr_options
    else:
        logger.info("⚙️ 模式: 禁用 OCR")

    pipeline_options.generate_picture_images = True
    pipeline_options.generate_page_images = save_pages
    pipeline_options.images_scale = page_dpi / 72.0
    logger.info(f"⚙️ 图片渲染 DPI 设置为: {page_dpi} (scale: {pipeline_options.images_scale:.2f})")

    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    logger.info(f"\n🚀 开始转换文件: {pdf_path.name}")

    try:
        doc = doc_converter.convert(str(pdf_path)).document
    except Exception as e:
        logger.error(f"❌ 转换过程中发生错误: {e}")
        raise e # 重新抛出异常

    if save_pages:
        logger.info(f"📸 正在保存分页图片...")
        page_output_dir = output_dir / "page"
        page_output_dir.mkdir(exist_ok=True)
        count = 0
        for page_no, page in doc.pages.items():
            if page.image and hasattr(page.image, 'pil_image'):
                page_filename = f"page_{page_no}.png"
                page_save_path = page_output_dir / page_filename
                page.image.pil_image.save(page_save_path, format="PNG")
                count += 1
        logger.info(f"✅ 成功保存 {count} 张分页图片。")

    markdown_parts = []
    image_output_dir = output_dir / "image"
    image_output_dir.mkdir(exist_ok=True)
    logger.info(f"✍️ 正在手动构建 Markdown 内容...")

    for element, level in doc.iterate_items():
        part_md = ""
        try:
            if isinstance(element, PictureItem):
                page_no = -1
                if element.prov and len(element.prov) > 0:
                    page_no = element.prov[0].page_no
                image = element.get_image(doc)
                if image is None: # 添加空检查
                    continue
                image_hash = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = f"page_{page_no}_{image_hash[:16]}.png"
                image_save_path = image_output_dir / image_filename
                image.save(image_save_path, format="PNG")
                # 使用相对路径
                part_md = f"![Image from page {page_no}](image/{image_filename})"
            elif isinstance(element, SectionHeaderItem):
                text = element.text.strip()
                hashes = '#' * (level + 2)
                part_md = f"{hashes} {text}"
            elif isinstance(element, TableItem):
                if hasattr(element, 'export_to_markdown'):
                    part_md = element.export_to_markdown(doc=doc)
            elif hasattr(element, 'text'):
                part_md = element.text
        except Exception as item_e:
            logger.warning(f"处理元素 {type(element)} 时出错: {item_e}")

        if part_md and part_md.strip():
            markdown_parts.append(part_md)

    final_markdown = "\n\n".join(markdown_parts)
    logger.info(f"✅ 成功处理 {len(markdown_parts)} 个内容块。")

    # 将md文件名与原始pdf文件名保持一致
    md_output_path = output_dir / f"{pdf_path.stem}.md"
    md_output_path.write_text(final_markdown, encoding='utf-8')

    logger.info(f"\n✅ 转换全部完成!")
    logger.info(f"📄 Markdown 及相关图片已保存至: {output_dir.resolve()}")

    return md_output_path


def process_pdf_task_logic(pdf_file_path_str: str, output_dir_str: str) -> (Path, Path):
    """
    处理单个 PDF 文件的主入口函数。
    它会读取配置、计算路径并调用核心转换函数。
    返回 (Markdown文件路径, 图片目录路径)
    """
    pdf_path = Path(pdf_file_path_str)
    output_dir = Path(output_dir_str)

    logger.info(f"-> Starting PDF processing logic for: {pdf_path.name}")

    # --- 配置管理 (从 settings.py 读取) ---
    # getattr(settings, 'SETTING_NAME', default_value) 是一种安全的方式
    # 确保在 settings.py 中忘记定义时, 程序仍可使用默认值运行
    should_enable_ocr = getattr(settings, 'PDF_PARSER_ENABLE_OCR', False)
    should_save_pages = getattr(settings, 'PDF_PARSER_SAVE_PAGES', False)
    page_resolution_dpi = getattr(settings, 'PDF_PARSER_PAGE_DPI', 200)

    # Docling 模型目录 (从 settings.py 读取)
    default_model_path = settings.BASE_DIR / "docling_models"
    model_path = getattr(settings, 'PDF_PARSER_MODEL_PATH', default_model_path)

    # 确保 model_path 是一个 Path 对象
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    # --- 配置读取结束 ---

    md_file_path = convert_pdf_to_markdown_with_images(
        pdf_path=pdf_path,
        model_dir=model_path,
        output_dir=output_dir,
        use_ocr=should_enable_ocr,
        save_pages=should_save_pages,
        page_dpi=page_resolution_dpi
    )

    image_dir_path = output_dir / "image"

    return md_file_path, image_dir_path