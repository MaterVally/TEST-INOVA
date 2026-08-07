"""
PDF ingestion: text chunking, PyMuPDF / MinerU parsing, image description.
"""
import asyncio
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO

from PIL import Image
from tqdm import tqdm

from ..config import settings as parameter
from ..storage.kv_storage import BaseKVStorage, JsonKVStorage
from ..utils.base import (
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    load_json,
    logger,
)
from .image_utils import (
    compress_image_to_size,
    find_chunk_for_image,
    get_image_description,
)

# ============================================================================
# Text chunking
# ============================================================================

def chunking_by_token_size(content, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"):
    tokens  = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(range(0, len(tokens), max_token_size - overlap_token_size)):
        chunk_tokens  = tokens[start: start + max_token_size]
        chunk_content = decode_tokens_by_tiktoken(chunk_tokens, model_name=tiktoken_model)
        results.append({
            "tokens": min(max_token_size, len(tokens) - start),
            "content": chunk_content.strip(),
            "chunk_order_index": index,
        })
    return results


@dataclass
class TextChunking:
    chunk_func:                           Callable          = chunking_by_token_size
    chunk_token_size:                     int               = 1200
    chunk_overlap_token_size:             int               = 100
    key_string_value_json_storage_cls:    type[BaseKVStorage] = JsonKVStorage
    tiktoken_model_name:                  str               = "gpt-4o"

    def __post_init__(self):
        self.full_docs   = self.key_string_value_json_storage_cls(namespace="full_docs")
        self.text_chunks = self.key_string_value_json_storage_cls(namespace="text_chunks")

    async def text_chunking(self, string_or_strings):
        if isinstance(string_or_strings, str):
            string_or_strings = [string_or_strings]
        try:
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            full_doc_keys     = list(new_docs.keys())
            existing_doc_keys = await self.full_docs.filter_keys(full_doc_keys)
            new_docs_to_insert = {k: v for k, v in new_docs.items() if k in existing_doc_keys}
            if not new_docs_to_insert:
                logger.warning("所有文档已存在")
                return
            logger.info(f"📝 插入新文档: {len(new_docs_to_insert)} 篇")
            inserting_chunks = {}
            for doc_key, doc in new_docs_to_insert.items():
                chunks = self.chunk_func(
                    doc["content"],
                    overlap_token_size=self.chunk_overlap_token_size,
                    max_token_size=self.chunk_token_size,
                    tiktoken_model=self.tiktoken_model_name,
                )
                for chunk in chunks:
                    chunk_id = compute_mdhash_id(chunk["content"], prefix="chunk-")
                    inserting_chunks[chunk_id] = {**chunk, "full_doc_id": doc_key}
            missing_chunk_keys = await self.text_chunks.filter_keys(list(inserting_chunks.keys()))
            final_chunks = {k: v for k, v in inserting_chunks.items() if k in missing_chunk_keys}
            if not final_chunks:
                logger.warning("所有文本块已存在")
                return
            logger.info(f"📄 插入新文本块: {len(final_chunks)} 个")
            await self.full_docs.upsert(new_docs_to_insert)
            await self.text_chunks.upsert(final_chunks)
        finally:
            await self._done()

    async def _done(self):
        await asyncio.gather(
            self.full_docs.index_done_callback(),
            self.text_chunks.index_done_callback()
        )


text_chunking_func = TextChunking


# ============================================================================
# PdfChunking — dual-engine PDF processor (PyMuPDF + MinerU)
# ============================================================================

@dataclass
class PdfChunking:
    pdf_path:    str
    working_dir: str = field(default_factory=lambda: parameter.WORKING_DIR)
    use_mineru:  bool = field(default_factory=lambda: parameter.USE_MINERU)

    def __post_init__(self):
        os.makedirs(self.working_dir, exist_ok=True)

    async def process(self):
        if self.use_mineru and self._mineru_available():
            logger.info("📄 使用 MinerU 解析 PDF")
            return await self._process_mineru()
        logger.info("📄 使用 PyMuPDF 解析 PDF")
        return await self._process_pymupdf()

    def _mineru_available(self):
        import shutil as _shutil
        return _shutil.which("mineru") is not None

    async def _process_pymupdf(self):
        import fitz  # PyMuPDF
        doc          = fitz.open(self.pdf_path)
        texts        = []
        image_data   = {}
        cache_kv     = JsonKVStorage(namespace="multimodel_llm_response_cache",
                                     storage_dir=parameter.CACHE_PATH)

        images_dir   = os.path.join(self.working_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        text_chunks_all = load_json(
            os.path.join(self.working_dir, "kv_store_text_chunks.json")) or {}

        for page_num, page in enumerate(tqdm(doc, desc="📖 解析页面", unit="页")):
            # Extract text
            page_text = page.get_text("text")
            if page_text.strip():
                texts.append(page_text)

            # Extract images
            for img_index, img_ref in enumerate(page.get_images(full=True)):
                xref       = img_ref[0]
                base_image = doc.extract_image(xref)
                img_bytes  = base_image["image"]
                img_name   = f"image_{page_num + 1}_{img_index + 1}"
                img_path   = os.path.join(images_dir, f"{img_name}.jpg")

                pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                compress_image_to_size(pil_img, img_path)

                # Get surrounding text context
                context = page_text[:500] if page_text else ""
                desc, seg = await get_image_description(
                    img_path, caption=[], footnote=[], context=context,
                    hashing_kv=cache_kv
                )

                chunk_id = find_chunk_for_image(text_chunks_all, context)
                chunk_idx = 0
                if chunk_id and chunk_id in text_chunks_all:
                    chunk_idx = text_chunks_all[chunk_id].get("chunk_order_index", 0)

                image_data[img_name] = {
                    "image_path":       img_path,
                    "description":      desc,
                    "segmentation":     seg,
                    "chunk_order_index": chunk_idx,
                }

        # Persist image data
        from ..utils.base import write_json
        write_json(image_data, os.path.join(self.working_dir, "kv_store_image_data.json"))
        await cache_kv.index_done_callback()
        doc.close()
        return texts, list(image_data.keys())

    async def _process_mineru(self):
        pdf_name    = os.path.splitext(os.path.basename(self.pdf_path))[0]
        mineru_dir  = os.path.join(self.working_dir, pdf_name, "auto")
        os.makedirs(mineru_dir, exist_ok=True)

        # Run MinerU CLI
        result = subprocess.run(
            ["mineru", "-p", self.pdf_path, "-o", os.path.join(self.working_dir, pdf_name)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            logger.warning(f"MinerU failed: {result.stderr}. Falling back to PyMuPDF.")
            return await self._process_pymupdf()

        # Read MinerU markdown output
        md_file = os.path.join(mineru_dir, f"{pdf_name}.md")
        if not os.path.exists(md_file):
            logger.warning("MinerU markdown not found, falling back to PyMuPDF.")
            return await self._process_pymupdf()

        with open(md_file, encoding="utf-8") as f:
            full_text = f.read()

        # Read content_list.json for images
        content_list_path = os.path.join(mineru_dir, f"{pdf_name}_content_list.json")
        image_data = {}
        if os.path.exists(content_list_path):
            content_list = load_json(content_list_path) or []
            cache_kv     = JsonKVStorage(namespace="multimodel_llm_response_cache",
                                         storage_dir=parameter.CACHE_PATH)
            text_chunks_all = load_json(
                os.path.join(self.working_dir, "kv_store_text_chunks.json")) or {}

            images_dir = os.path.join(self.working_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            for idx, item in enumerate(tqdm(content_list, desc="🖼️ 处理图像", unit="个")):
                if item.get("type") != "image":
                    continue
                img_path_raw = item.get("img_path", "")
                if not os.path.exists(img_path_raw):
                    continue
                img_name = f"image_{idx}"
                img_dest = os.path.join(images_dir, f"{img_name}.jpg")

                pil_img = Image.open(img_path_raw).convert("RGB")
                compress_image_to_size(pil_img, img_dest)

                caption  = item.get("img_caption", [])
                footnote = item.get("img_footnote", [])
                context  = item.get("text_context", "")

                desc, seg = await get_image_description(
                    img_dest, caption=caption, footnote=footnote,
                    context=context, hashing_kv=cache_kv
                )

                chunk_id  = find_chunk_for_image(text_chunks_all, context)
                chunk_idx = 0
                if chunk_id and chunk_id in text_chunks_all:
                    chunk_idx = text_chunks_all[chunk_id].get("chunk_order_index", 0)

                image_data[img_name] = {
                    "image_path":        img_dest,
                    "description":       desc,
                    "segmentation":      seg,
                    "chunk_order_index": chunk_idx,
                }

            from ..utils.base import write_json
            write_json(image_data, os.path.join(self.working_dir, "kv_store_image_data.json"))
            await cache_kv.index_done_callback()

        return [full_text], list(image_data.keys())


chunking_func_pdf2md = PdfChunking
