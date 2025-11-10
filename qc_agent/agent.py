"""LangGraph workflow for the EMR QC agent."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict
import json
import logging

from langgraph.graph import StateGraph, START, END

from .config import AgentConfig
from .kimi import KimiClient, ChatMessage, KimiClientError
from .rag import RAGStore, build_query_text, RuleDocument, FewShotExample
from .prompts import build_messages


logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    record_id: str
    fields: Dict[str, Any]
    rules: List[RuleDocument]
    examples: List[FewShotExample]
    llm_response: Dict[str, Any]
    raw_text: str


def _safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
    raise


def build_qc_app(config: AgentConfig):
    rag_store = RAGStore(config.rules_path, config.train_records_path, config.train_split_path)
    kimi = KimiClient(api_key=config.kimi_api_key, base_url=config.kimi_base_url, model=config.kimi_model)

    def prepare_node(state: AgentState) -> AgentState:
        fields = state.get("fields") or {}
        query_text = build_query_text(fields)
        rules = rag_store.retrieve_rules(query_text, top_k=config.rule_top_k, min_score=config.rag_min_score)
        examples = rag_store.retrieve_examples(query_text, top_k=config.fewshot_top_k, min_score=config.rag_min_score)
        return {"rules": rules, "examples": examples}

    def llm_node(state: AgentState) -> AgentState:
        record_id = state["record_id"]
        fields = state.get("fields") or {}
        messages = build_messages(record_id, fields, state.get("rules", []), state.get("examples", []))
        chat_messages = [ChatMessage(**msg) for msg in messages]
        try:
            response = kimi.chat(chat_messages, temperature=config.temperature, max_output_tokens=config.max_output_tokens)
            text = response["text"]
            parsed = _safe_parse_json(text)
        except (KimiClientError, json.JSONDecodeError) as exc:
            logger.error("LLM failed for %s: %s", record_id, exc)
            parsed = {"problems": [], "fixes": [], "error": str(exc)}
            text = ""
        return {"llm_response": parsed, "raw_text": text}

    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare_node)
    graph.add_node("llm", llm_node)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "llm")
    graph.add_edge("llm", END)
    return graph.compile()
