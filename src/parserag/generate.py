from __future__ import annotations

import json

from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from parserag.config import LLM_MODEL, openai_api_key


class Citation(BaseModel):
    doc_id: str
    page: int


class Answer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


PROMPT = """Answer the question using only the retrieved context.
Return JSON with keys:
  answer: string
  citations: list of objects with doc_id and page (integers, 1-indexed)

Question: {question}

Context:
{context}
"""


def generate_answer(question: str, nodes: list[NodeWithScore]) -> dict:
    llm = OpenAI(model=LLM_MODEL, api_key=openai_api_key(), temperature=0)
    prompt = PROMPT.format(question=question, context=_format_context(nodes))
    structured = getattr(llm, "as_structured_llm", None)
    if structured is not None:
        try:
            result = structured(Answer).complete(prompt)
            raw = result.raw if hasattr(result, "raw") else None
            if isinstance(raw, Answer):
                return raw.model_dump()
            text = str(result)
            return Answer.model_validate_json(_json_blob(text)).model_dump()
        except (ValidationError, ValueError, TypeError):
            pass
    text = str(llm.complete(prompt + "\nRespond with JSON only."))
    try:
        return Answer.model_validate_json(_json_blob(text)).model_dump()
    except ValidationError:
        return {"answer": text.strip(), "citations": []}


def _format_context(nodes: list[NodeWithScore]) -> str:
    parts = []
    for i, node in enumerate(nodes, start=1):
        meta = node.node.metadata or {}
        parts.append(
            f"[{i}] doc_id={meta.get('doc_id')} page={meta.get('page')}\n{node.node.get_content()}"
        )
    return "\n\n".join(parts) if parts else "(no retrieved context)"


def _json_blob(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        json.loads(text[start : end + 1])
        return text[start : end + 1]
    return text
