import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(os.environ.get("PARSERAG_ROOT", Path.cwd()))
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
PARSED_DIR = DATA_DIR / "parsed"
INDEX_DIR = DATA_DIR / "indexes"
RESULTS_DIR = ROOT / "results"

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
RETRIEVER_K = 20
FUSED_K = 8
RRF_K = 60
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128


def openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return key


def llama_cloud_api_key() -> str:
    key = os.environ.get("LLAMA_CLOUD_API_KEY", "")
    if not key:
        raise RuntimeError("LLAMA_CLOUD_API_KEY is not set")
    return key


def configure_llama_index() -> None:
    from llama_index.core import Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL, api_key=openai_api_key())
    Settings.llm = OpenAI(model=LLM_MODEL, api_key=openai_api_key(), temperature=0)


def pdf_dir(dataset: str) -> Path:
    path = PDF_DIR / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def parsed_dir(dataset: str, doc_id: str) -> Path:
    path = PARSED_DIR / dataset / _safe_id(doc_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def index_dir(dataset: str) -> Path:
    path = INDEX_DIR / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_id(doc_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in doc_id)
