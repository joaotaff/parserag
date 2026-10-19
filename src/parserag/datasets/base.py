from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GoldPage:
    doc_id: str
    page: int  # 1-indexed


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    pdf_path: Path


@dataclass
class Question:
    question_id: str
    question: str
    gold_pages: list[GoldPage] = field(default_factory=list)
    gold_answers: list[str] = field(default_factory=list)
    domain: str = "unknown"
    metadata: dict = field(default_factory=dict)


class DatasetAdapter(ABC):
    name: str

    @abstractmethod
    def prepare_pdfs(self, limit: int | None = None) -> None:
        """Download or extract PDFs onto disk."""

    @abstractmethod
    def iter_documents(self, limit: int | None = None) -> list[SourceDoc]:
        """Unique source PDFs for this split."""

    @abstractmethod
    def iter_questions(self, limit: int | None = None) -> list[Question]:
        """Evaluation questions. Empty gold_pages are excluded from Recall@k."""
