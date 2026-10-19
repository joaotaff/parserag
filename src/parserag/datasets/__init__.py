from parserag.datasets.base import DatasetAdapter, GoldPage, Question, SourceDoc
from parserag.datasets.dude import DudeAdapter
from parserag.datasets.financebench import FinanceBenchAdapter

ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "financebench": FinanceBenchAdapter,
    "dude": DudeAdapter,
}


def get_adapter(name: str) -> DatasetAdapter:
    try:
        return ADAPTERS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {name!r}. Choose from: {sorted(ADAPTERS)}") from exc


__all__ = [
    "ADAPTERS",
    "DatasetAdapter",
    "DudeAdapter",
    "FinanceBenchAdapter",
    "GoldPage",
    "Question",
    "SourceDoc",
    "get_adapter",
]
