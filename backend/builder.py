"""
Pipeline orchestrator — wires all stages together.

Supports multi-format ingestion:
  .pdf                          → PdfChunking
  .docx                         → DocxChunking
  .xlsx / .xls                  → ExcelChunking
  .mp3 / .wav / .m4a / .flac   → AudioChunking
  .png / .jpg / .jpeg / .bmp /
  .webp / .tif / .tiff          → ImageChunking

All processors expose the same interface:
    texts, images = await processor.process()

Everything downstream (TextChunking → Graph → Fusion → Output) is unchanged.
"""
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings as parameter
from .graph.fusion import fusion
from .graph.img2graph import img2graph
from .graph.text2graph import TextEntityExtractor
from .ingestion.audio_preprocessing import AudioChunking
from .ingestion.docx_preprocessing import DocxChunking
from .ingestion.excel_preprocessing import ExcelChunking
from .ingestion.image_preprocessing import ImageChunking
from .ingestion.pdf_preprocessing import PdfChunking, TextChunking
from .utils.base import get_latest_graphml_file, load_json, logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_cache_path = parameter.CACHE_PATH or "data/cache"
os.makedirs(_cache_path, exist_ok=True)
os.environ["CACHE_PATH"] = _cache_path

# ---------------------------------------------------------------------------
# Extension → processor mapping
# ---------------------------------------------------------------------------

_PDF_EXTS   = {".pdf"}
_DOCX_EXTS  = {".docx"}
_EXCEL_EXTS = {".xlsx", ".xls"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

_SUPPORTED_EXTS = _PDF_EXTS | _DOCX_EXTS | _EXCEL_EXTS | _AUDIO_EXTS | _IMAGE_EXTS


def _build_processor(file_path: str, working_dir: str, use_mineru: bool):
    """
    Return the correct preprocessing instance for *file_path*.

    All returned objects expose:  async process() -> (texts, images)
    """
    ext = Path(file_path).suffix.lower()

    if ext in _PDF_EXTS:
        return PdfChunking(
            pdf_path=file_path,
            working_dir=working_dir,
            use_mineru=use_mineru,
        )
    if ext in _DOCX_EXTS:
        return DocxChunking(
            docx_path=file_path,
            working_dir=working_dir,
        )
    if ext in _EXCEL_EXTS:
        return ExcelChunking(
            excel_path=file_path,
            working_dir=working_dir,
        )
    if ext in _AUDIO_EXTS:
        return AudioChunking(
            audio_path=file_path,
            working_dir=working_dir,
        )
    if ext in _IMAGE_EXTS:
        return ImageChunking(
            image_path=file_path,
            working_dir=working_dir,
        )

    raise ValueError(
        f"Unsupported file type '{ext}'. "
        f"Supported: {', '.join(sorted(_SUPPORTED_EXTS))}"
    )


@dataclass
class MMKGBuilder:
    file_path:   str  = field(default_factory=lambda: parameter.INPUT_PDF_PATH)
    working_dir: str  = field(default_factory=lambda: parameter.WORKING_DIR)
    output_dir:  str  = field(default_factory=lambda: parameter.OUTPUT_DIR)
    mmkg_name:   str  = field(default_factory=lambda: parameter.MMKG_NAME)
    use_mineru:  bool = field(default_factory=lambda: parameter.USE_MINERU)

    def __post_init__(self):
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.output_dir,  exist_ok=True)
        os.makedirs(_cache_path,      exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def index(self, file_path: str | None = None):
        file_path = file_path or self.file_path
        logger.info(f"📂 开始处理: {file_path}")

        await self._step_preprocessing(file_path)
        await self._step_text_extraction()
        img_ids = await self._step_image_extraction()
        await self._step_fusion(img_ids)
        self._step_save_output()
        self._step_generate_report()

        logger.info("✅ 知识图谱构建完成")

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    async def _step_preprocessing(self, file_path: str):
        chunks_path = os.path.join(self.working_dir, "kv_store_text_chunks.json")
        if os.path.exists(chunks_path):
            logger.info("⏭️  预处理已完成，跳过")
            return

        ext = Path(file_path).suffix.lower()
        logger.info(f"📄 步骤 1/5 — 文件预处理 [{ext}]")

        processor = _build_processor(file_path, self.working_dir, self.use_mineru)
        texts, _images = await processor.process()

        text_chunking = TextChunking()
        await text_chunking.text_chunking(texts)

    async def _step_text_extraction(self):
        graph_path = os.path.join(self.working_dir, "graph_chunk_entity_relation.graphml")
        if os.path.exists(graph_path):
            logger.info("⏭️  文本实体提取已完成，跳过")
            return

        logger.info("📝 步骤 2/5 — 文本实体提取")
        chunks = load_json(os.path.join(self.working_dir, "kv_store_text_chunks.json")) or {}
        extractor = TextEntityExtractor()
        await extractor.text_entity_extraction(chunks)

    async def _step_image_extraction(self) -> list[str]:
        image_data_path = os.path.join(self.working_dir, "kv_store_image_data.json")
        image_data      = load_json(image_data_path) or {}
        img_ids         = list(image_data.keys())

        if not img_ids:
            logger.info("⏭️  无图像，跳过图像提取")
            return []

        logger.info(f"🖼️  步骤 3/5 — 图像实体提取 ({len(img_ids)} 张)")
        images_dir = os.path.join(self.working_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Collect source images into images_dir
        image_paths = []
        for img_info in image_data.values():
            img_path = img_info.get("image_path", "")
            if os.path.exists(img_path):
                image_paths.append(img_path)

        if image_paths:
            # Run img2graph on the directory containing source images
            src_dir = os.path.dirname(image_paths[0])
            await img2graph(src_dir)

        return img_ids

    async def _step_fusion(self, img_ids: list[str]):
        if not img_ids:
            return

        _existing_graph = get_latest_graphml_file(self.working_dir)
        merged_exists = any(
            f.startswith("graph_merged_") for f in os.listdir(self.working_dir)
        )
        if merged_exists:
            logger.info("⏭️  图谱融合已完成，跳过")
            return

        logger.info(f"🔗 步骤 4/5 — 图谱融合 ({len(img_ids)} 张图像)")
        await fusion(img_ids, working_dir=self.working_dir)

    def _step_save_output(self):
        logger.info("💾 步骤 5a/5 — 保存最终图谱")
        _namespace, src_path = get_latest_graphml_file(self.working_dir)
        if not os.path.exists(src_path):
            logger.warning(f"⚠️  未找到图谱文件: {src_path}")
            return
        dest = os.path.join(self.output_dir, f"{self.mmkg_name}.graphml")
        shutil.copy2(src_path, dest)
        logger.info(f"📦 图谱已保存至: {dest}")

    def _step_generate_report(self):
        logger.info("📊 步骤 5b/5 — 生成报告")
        import networkx as nx
        graph_path = os.path.join(self.output_dir, f"{self.mmkg_name}.graphml")
        if not os.path.exists(graph_path):
            return
        G = nx.read_graphml(graph_path)
        type_counts: dict = {}
        for _, data in G.nodes(data=True):
            etype = data.get("entity_type", "UNKNOWN").strip('"')
            type_counts[etype] = type_counts.get(etype, 0) + 1

        report_lines = [
            f"# Knowledge Graph Build Report — {self.mmkg_name}\n",
            f"- **Nodes**: {G.number_of_nodes()}",
            f"- **Edges**: {G.number_of_edges()}",
            "\n## Entity Type Distribution\n",
        ] + [f"- {k}: {v}" for k, v in sorted(type_counts.items(), key=lambda x: -x[1])]

        report_path = os.path.join(self.output_dir, f"{self.mmkg_name}_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        logger.info(f"📋 报告已保存至: {report_path}")
