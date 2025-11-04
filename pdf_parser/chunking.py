import re
import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# --- 辅助函数：从您的代码中提取 ---

def tokenize_text(text: str) -> List[str]:
    """使用正则表达式分词"""
    if not text:
        return []
    return re.findall(r'\S+|\n', text)

def remove_image_markdown(text: str) -> str:
    """从文本中移除 Markdown 图片路由（![](...) 或 ![alt](...)）"""
    return re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', text)

def _clean_pipe_table(table_lines: List[str]) -> List[str]:
    """表格清理子函数"""
    cleaned = []
    for line in table_lines:
        if "|" not in line:
            cleaned.append(line)
            continue
        has_prefix = line.strip().startswith("|")
        has_suffix = line.strip().endswith("|")
        parts = [cell.strip() for cell in line.strip().strip("|").split("|")]
        new_line = "|".join(parts)
        if has_prefix:
            new_line = "|" + new_line
        if has_suffix:
            new_line = new_line + "|"
        cleaned.append(new_line)
    return cleaned

def split_markdown_to_blocks(md_text: str, clean_table_whitespace: bool = False) -> List[str]:
    """将 Markdown 文本分割为逻辑块（标题、段落、表格、图片）"""
    lines = md_text.splitlines()
    blocks = []
    current_block = []
    in_table = False

    def flush_block():
        nonlocal current_block
        if current_block:
            blocks.append("\n".join(current_block).strip())
            current_block = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("![](") or stripped.startswith("!["):
            flush_block()
            blocks.append(stripped)
        elif stripped.startswith("#"):
            flush_block()
            blocks.append(stripped)
        elif stripped.startswith("<!--"):
            flush_block()
            blocks.append(stripped)
        elif "|" in stripped:
            if not in_table:
                flush_block()
                in_table = True
            current_block.append(stripped)
        else:
            if in_table:
                if clean_table_whitespace:
                    current_block = clean_pipe_table(current_block)
                flush_block()
                in_table = False
            if stripped == "":
                flush_block()
            else:
                current_block.append(stripped)
    # 1. 检查文件是否以表格结束
    if in_table:
        if clean_table_whitespace:
            current_block = _clean_pipe_table(current_block)
        flush_block()  # 刷新最后一个表格块

    # 2. 刷新最后一个文本块
    flush_block()

    # 3. 返回所有块
    return blocks

def get_max_block_size(blocks: List[str]) -> int:
    """获取所有块中最大的令牌数"""
    if not blocks:
        return 0
    # (确保 tokenize_text 函数在文件上方已被定义)
    return max(len(tokenize_text(block)) for block in blocks)

def chunk_text_with_dynamic_size(blocks: List[str], dynamic_chunk_size: int) -> List[str]:
    """使用动态块大小将块合并为初步的 Chunks"""
    chunks = []
    current_chunk = ""
    current_chunk_size = 0

    for block in blocks:
        block_tokens = tokenize_text(block)
        block_size = len(block_tokens)

        if current_chunk_size + block_size <= dynamic_chunk_size:
            current_chunk += "\n\n" + block  # 用换行符连接块
            current_chunk_size += block_size
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = block
            current_chunk_size = block_size

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def apply_sliding_window(chunks: List[str], window_size: int) -> List[str]:
    """应用滑动窗口合并 Chunks"""
    final_chunks = []
    for i in range(len(chunks)):
        start = max(0, i - window_size // 2)
        end = min(len(chunks), i + window_size // 2 + 1)
        # 用换行符连接相邻的 Chunks
        merged_chunk = "\n\n".join(chunks[start:end]).strip()
        final_chunks.append(merged_chunk)
    return final_chunks


# --- 新的、符合您要求的 JSON 保存函数 ---

def save_chunks_to_json(chunks: List[str], source_document_name: str, output_path: Path):
    """
    保存最终的 Chunks 到 chunks.json，使用您指定的新格式。
    """
    chunk_objects = []
    for idx, chunk in enumerate(chunks, 1):
        token_count = len(tokenize_text(chunk))
        chunk_objects.append({
            "id": str(idx),  # ID 设为字符串
            "text": chunk,
            "token_count": token_count,
            "source_document_name": source_document_name
        })

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunk_objects, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存 {len(chunk_objects)} 个 chunk 到: {output_path}")
    except Exception as e:
        logger.error(f"保存 chunks.json 失败: {e}")
        raise


def save_basic_chunks_to_json(chunks: List[str], output_path: Path):
    """
    保存基础 Chunks 到 basic_chunk.json，使用旧格式（用于调试或中间步骤）。
    """
    chunk_objects = []
    for idx, chunk in enumerate(chunks, 1):
        token_count = len(tokenize_text(chunk))
        preceding_block_id = idx - 1 if idx > 1 else None
        next_block_id = idx + 1 if idx < len(chunks) else None
        chunk_objects.append({
            "id": idx,
            "text": chunk,
            "token_count": token_count,
            "preceding_block_id": preceding_block_id,
            "next_block_id": next_block_id
        })

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunk_objects, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存 {len(chunk_objects)} 个 basic chunk 到: {output_path}")
    except Exception as e:
        logger.error(f"保存 basic_chunk.json 失败: {e}")
        raise


# --- 主协调函数 (供 Celery 调用) ---

def process_markdown_to_chunks(
        md_file_path: str,
        source_document_name: str,
        config: Dict[str, Any],
        output_dir_path: str
):
    """
    从 Celery 调用的主函数，协调所有分块步骤。
    """
    logger.info(f"开始分块处理: {md_file_path}")

    # 1. 读取配置
    MIN_CHUNK_SIZE = config.get("MIN_CHUNK_SIZE", 100)
    MAX_CHUNK_SIZE = config.get("MAX_CHUNK_SIZE", 500)
    n = config.get("N_FACTOR", 1.5)
    window_size = config.get("WINDOW_SIZE", 2)
    clean_table_whitespace = config.get("CLEAN_TABLE_WHITESPACE", False)

    # 2. 读取 MD 文件
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
    except Exception as e:
        logger.error(f"无法读取 Markdown 文件: {md_file_path} - {e}")
        raise

    # 3. 执行分块逻辑
    # (*** 假设 split_markdown_to_blocks 和 get_max_block_size 存在于此文件上方 ***)
    blocks = split_markdown_to_blocks(markdown_text, clean_table_whitespace)
    max_block_size = get_max_block_size(blocks)
    dynamic_chunk_size = min(max(n * max_block_size, MIN_CHUNK_SIZE), MAX_CHUNK_SIZE)

    # 4. 初步分块
    initial_chunks = chunk_text_with_dynamic_size(blocks, dynamic_chunk_size)

    # 5. 生成 basic_chunks (去除图片)
    # (*** 假设 remove_image_markdown 存在于此文件上方 ***)
    basic_chunks = [remove_image_markdown(chunk).strip() for chunk in initial_chunks]

    # 6. 滑动窗口合并（保留图片）
    final_chunks_with_duplicates = apply_sliding_window(initial_chunks, window_size)

    # --- 6b. 新增：去重处理 (优化点) ---
    # 我们使用一个 set 来跟踪已经添加的 chunk 内容，以保持顺序并去除重复
    seen_chunks = set()
    final_chunks = [] # 这是去重后的列表
    for chunk in final_chunks_with_duplicates:
        if chunk not in seen_chunks:
            final_chunks.append(chunk)
            seen_chunks.add(chunk)
    # --- 去重结束 ---


    logger.info(f"原始块数: {len(blocks)}")
    logger.info(f"最大块大小(令牌数): {max_block_size}")
    logger.info(f"动态 chunk_size = {dynamic_chunk_size}")
    logger.info(f"初步分块数: {len(initial_chunks)}")
    logger.info(f"滑动窗口块数 (去重前): {len(final_chunks_with_duplicates)}")
    logger.info(f"最终分块数 (去重后): {len(final_chunks)}") # <--- 更新日志

    # 7. 保存 JSON 文件到指定的 /results 目录
    output_path = Path(output_dir_path)
    output_path.mkdir(parents=True, exist_ok=True)  # 确保 /results 目录存在

    save_basic_chunks_to_json(basic_chunks, output_path / "basic_chunk.json")
    # 确保保存的是去重后的 final_chunks 列表
    save_chunks_to_json(final_chunks, source_document_name, output_path / "chunks.json")

    logger.info(f"分块 JSON 文件已保存到: {output_path}")