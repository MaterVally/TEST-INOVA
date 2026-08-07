"""
Runtime configuration for the Enterprise Compliance Intelligence Platform.
...
"""
import os

from dotenv import load_dotenv

# Load .env BEFORE any os.environ.get() calls so env vars are available
# regardless of import order (uvicorn reloader imports config before main.py).
load_dotenv()

from sentence_transformers import SentenceTransformer

# ============ LLM Configuration ============
# Text LLM — entity extraction, relation building, RAG answers
API_KEY    = os.environ.get("LLM_API_KEY",    "")
API_BASE   = os.environ.get("LLM_API_BASE",   "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-4o")

# Multimodal LLM — image understanding, visual entity extraction, scene graphs
# Uses the same OpenAI key and endpoint; gpt-4o supports vision natively.
MM_API_KEY    = os.environ.get("MM_API_KEY",    "")
MM_API_BASE   = os.environ.get("MM_API_BASE",   "https://api.openai.com/v1")
MM_MODEL_NAME = os.environ.get("MM_MODEL_NAME", "gpt-4o")

# ============ Embedding Model ============
_default_embed_dir = "./models/all-MiniLM-L6-v2" if os.path.exists("./models/all-MiniLM-L6-v2") else "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_DIR = os.environ.get("EMBEDDING_MODEL_DIR", _default_embed_dir)
EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL_DIR, device="cpu")

# ============ Directory Paths ============
INPUT_PDF_PATH = os.environ.get("INPUT_PDF_PATH", "data/input/2020.acl-main.45.pdf")
CACHE_PATH     = os.environ.get("CACHE_PATH",     "data/cache")
WORKING_DIR    = os.environ.get("WORKING_DIR",    "data/working")
OUTPUT_DIR     = os.environ.get("OUTPUT_DIR",     "data/output")
MMKG_NAME      = os.environ.get("MMKG_NAME",      "example_mmkg")

# ============ Processing Parameters ============
ENTITY_EXTRACT_MAX_GLEANING  = int(os.environ.get("ENTITY_EXTRACT_MAX_GLEANING",  "0"))
ENTITY_SUMMARY_MAX_TOKENS    = int(os.environ.get("ENTITY_SUMMARY_MAX_TOKENS",    "500"))
SUMMARY_CONTEXT_MAX_TOKENS   = int(os.environ.get("SUMMARY_CONTEXT_MAX_TOKENS",   "10000"))
USE_MINERU = os.environ.get("USE_MINERU", "true").lower() in ("1", "true", "yes")

# ============ RAG Retrieval Configuration ============
class QueryParam:
    top_k: int = 5
    response_type: str = "Detailed System-like Response"
    local_max_token_for_local_context: int = 4000
    number_of_mmentities: int = 3
    local_max_token_for_text_unit: int = 4000

RETRIEVAL_THRESHOLD: float = 0.2

# ============ Audio / OpenAI Whisper Configuration ============
# Audio transcription uses OpenAI Whisper — same key as LLM_API_KEY.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("LLM_API_KEY", "")

# ============ Auth / Supabase Configuration ============
SUPABASE_URL              = os.environ.get("SUPABASE_URL",              "")
SUPABASE_ANON_KEY         = os.environ.get("SUPABASE_ANON_KEY",         "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_JWT_SECRET       = os.environ.get("SUPABASE_JWT_SECRET",       "")

# Derived automatically — do NOT set this in the environment
SUPABASE_JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"

# Comma-separated list of allowed CORS origins, e.g. "https://app.example.com,https://admin.example.com"
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "")
