import os
from datetime import datetime
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
LLM_TIMEOUT = 180
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
    Settings.llm = OpenAI(model=LLM_MODEL, api_key=openai_api_key(), temperature=0, timeout=LLM_TIMEOUT)


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


def run_paths(dataset: str, timestamp: str | None = None) -> tuple[Path, Path]:
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / dataset / "run"
    path.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_{timestamp}"
    return path / f"{stem}.jsonl", path / f"{stem}.json"


def latest_run_path(dataset: str) -> Path:
    path = RESULTS_DIR / dataset / "run"
    snapshots = sorted(path.glob(f"{dataset}_*.json"), key=lambda item: item.stat().st_mtime)
    if snapshots:
        return snapshots[-1]
    legacy = path / "predictions.jsonl"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"No run output for {dataset}. Run the dataset first.")


def eval_path(dataset: str, run_path: Path) -> Path:
    path = RESULTS_DIR / dataset / "eval"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{run_path.stem}.metrics.json"


def _safe_id(doc_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in doc_id)
