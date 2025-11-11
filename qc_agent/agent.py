
"""LangGraph workflow for the ACE-QC agent."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict
import json
import logging
import re

from langgraph.graph import StateGraph, START, END

from .config import AgentConfig
from .kimi import KimiClient, ChatMessage, KimiClientError
from .gemini import GeminiClient, GeminiClientError  # type: ignore
from .rag import RAGStore, build_query_text, RuleDocument, FewShotExample
from .prompts import build_verdict_messages, build_reflection_prompt


logger = logging.getLogger(__name__)

QUALIFYING_KEYWORDS = ("合格", "无需质控", "不质控", "无需修改", "可保留")
NEGATION_CUES = ("未", "无", "否")
TREATMENT_RE = re.compile(r"(予|给予|口服|静滴|静推|静点|肌注|输液|抗生素|复查|再查|置入|拔除|行.{0,8}术|至.*?(医院|门诊)|在.*?(医院|门诊))")
DEFAULT_LAYER_C_PATCHES = [
    "不得直接生成 problems，系统会从 FAIL verdict 自动派生",
    "诊断遗漏/检查提示但未写入诊断 → IC-RZCB-01-V1",
    "主诉无法导出第一诊断 → DQ-RZ-01-V1",
    "以症状代替诊断 → DQ-RZ-02-V1",
    "辅助检查缺少四要素任一 → EN-FZ-01-V1",
]


class AgentState(TypedDict, total=False):
    record_id: str
    fields: Dict[str, Any]
    rules: List[RuleDocument]
    examples: List[FewShotExample]
    rule_lookup: Dict[str, RuleDocument]
    layer_c: List[str]
    kimi_payload: Dict[str, Any]
    verdicts: List[Dict[str, Any]]
    fixes_raw: List[Any]
    audit: Dict[str, Any]
    raw_text: str
    llm_response: Dict[str, Any]


def _safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _history_has_treatment(text: str) -> bool:
    if not text:
        return False
    for match in TREATMENT_RE.finditer(text):
        prefix = text[max(0, match.start() - 1) : match.start()]
        if prefix and prefix[-1] in NEGATION_CUES:
            continue
        return True
    return False


def _ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _normalize_evidence(spans: Any) -> List[Dict[str, str]]:
    normalized = []
    for span in _ensure_list(spans):
        if isinstance(span, dict):
            section = str(span.get("section") or "")
            text = str(span.get("text") or "")
        else:
            section = ""
            text = str(span)
        if not text:
            continue
        normalized.append({"section": section, "text": text})
    return normalized


def _normalize_verdicts(
    verdicts_raw: List[Any],
    rule_lookup: Dict[str, RuleDocument],
    fields: Dict[str, Any],
) -> List[Dict[str, Any]]:
    history_text = str(fields.get("history_present") or "")
    has_treatment = _history_has_treatment(history_text)
    normalized: List[Dict[str, Any]] = []
    for item in verdicts_raw:
        if not isinstance(item, dict):
            continue
        rule_id = str(item.get("rule_id") or "").strip()
        if not rule_id:
            continue
        meta = rule_lookup.get(rule_id)
        issue_type = item.get("issue_type") or (meta.issue_type if meta else "")
        field = item.get("field") or (meta.field_en if meta else "")
        verdict = (item.get("verdict") or "").upper() or "N/A"
        confidence = item.get("confidence")
        if not isinstance(confidence, (int, float)):
            confidence = 0.6
        desc = str(item.get("short_reason") or item.get("description") or "").strip()
        evidence = _normalize_evidence(item.get("evidence_spans"))
        entry = {
            "rule_id": rule_id,
            "issue_type": issue_type,
            "field": field,
            "verdict": verdict,
            "confidence": float(confidence),
            "short_reason": desc,
            "evidence_spans": evidence,
        }
        if any(keyword in desc for keyword in QUALIFYING_KEYWORDS):
            entry["verdict"] = "PASS"
        if rule_id == "DQ-RZ-01-V1" and any(word in desc for word in ("遗漏", "未写", "缺失")):
            entry["rule_id"] = "IC-RZCB-01-V1"
            entry["issue_type"] = "INFO_COMPLETENESS"
        if rule_id == "IC-XB-01-V1" and verdict == "FAIL" and not has_treatment:
            entry["verdict"] = "N/A"
            entry["short_reason"] = "未出现入院前治疗线索，按规则不质控"
        normalized.append(entry)
    return normalized


def _derive_outputs(
    record_id: str,
    verdicts: List[Dict[str, Any]],
    fixes_raw: List[Any],
    fields: Dict[str, Any],
    config: AgentConfig,
    layer_c_patch: List[str],
    audit: Dict[str, Any],
) -> Dict[str, Any]:
    problems: List[Dict[str, Any]] = []
    notes: List[Dict[str, Any]] = []
    threshold = config.fail_conf_threshold
    for idx, verdict in enumerate(verdicts):
        verdict["index"] = idx
        if verdict["verdict"] == "FAIL" and verdict["confidence"] >= threshold:
            # 为 problems 添加 description 字段（映射自 short_reason）
            problem = dict(verdict)
            problem["description"] = problem.get("short_reason", "")
            problems.append(problem)
        else:
            notes.append(verdict)
    fixes: List[Dict[str, str]] = []
    for item in _ensure_list(fixes_raw):
        if isinstance(item, dict):
            advice = str(item.get("advice") or item.get("text") or "").strip()
            if not advice:
                continue
            fixes.append(
                {
                    "rule_id": str(item.get("rule_id") or ""),
                    "title": str(item.get("title") or "【提示】"),
                    "advice": advice,
                }
            )
        elif isinstance(item, str):
            text = item.strip()
            if text:
                fixes.append({"rule_id": "", "title": "【提示】", "advice": text})
    metrics = {
        "deduction_total": len(problems),
        "summary": "未发现不合格项"
        if not problems
        else f"发现 {len(problems)} 条不合格项：" + ", ".join(sorted({p['rule_id'] for p in problems})),
    }
    return {
        "record_id": record_id,
        "verdicts": verdicts,
        "problems": problems,
        "notes": notes,
        "fixes": fixes,
        "metrics": metrics,
        "audit": {"layer_c_patch": layer_c_patch, **audit},
    }


def _find_suspects(verdicts: List[Dict[str, Any]], config: AgentConfig) -> List[Dict[str, Any]]:
    suspects: List[Dict[str, Any]] = []
    keywords = ("遗漏", "未写", "缺失")
    for verdict in verdicts:
        rule_id = verdict["rule_id"]
        reason = verdict.get("short_reason", "")
        confidence = verdict.get("confidence", 0.0)
        verdict_flag = verdict.get("verdict")
        if rule_id.startswith("DQ-") and any(k in reason for k in keywords):
            suspects.append(verdict)
        elif rule_id in config.reflection_rule_whitelist and verdict_flag == "FAIL":
            suspects.append(verdict)
        elif (
            verdict_flag == "FAIL"
            and config.reflection_conf_min <= confidence <= config.reflection_conf_max
        ):
            suspects.append(verdict)
    return suspects


def _apply_reflections(verdicts: List[Dict[str, Any]], revisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for revision in revisions:
        idx = revision.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(verdicts):
            continue
        merged = verdicts[idx].copy()
        for key in ("rule_id", "issue_type", "field", "verdict", "confidence", "short_reason"):
            if key in revision and revision[key] is not None:
                merged[key] = revision[key]
        if "evidence_spans" in revision:
            merged["evidence_spans"] = _normalize_evidence(revision.get("evidence_spans"))
        verdicts[idx] = merged
    return verdicts


def build_qc_app(config: AgentConfig):
    rag_store = RAGStore(config.rules_path, config.train_records_path, config.train_split_path)
    kimi = KimiClient(
        api_key=config.kimi_api_key,
        base_url=config.kimi_base_url,
        model=config.kimi_model,
        retry=config.kimi_retry,
        timeout=config.kimi_timeout,
        log_path=config.kimi_log_path,
    )

    reflection_runtime: tuple[str, Any] | None = None
    provider = (config.reflection_provider or "gemini").lower()
    if provider == "gemini":
        if config.has_gemini_key:
            try:
                gemini_client = GeminiClient(
                    api_key=config.gemini_api_key,
                    model=config.reflection_model or config.gemini_model,
                    rpm_limit=config.gemini_rpm_limit,
                    max_retries=config.gemini_max_retries,
                )
                reflection_runtime = ("gemini", gemini_client)
            except Exception as exc:  # pragma: no cover
                logger.warning("Gemini client unavailable: %s", exc)
        else:
            logger.warning("Reflection provider set to gemini but GEMINI_API_KEY missing.")
    elif provider in {"kimi", "openai"}:
        api_key = (config.reflection_api_key or config.kimi_api_key).strip()
        if not api_key:
            logger.warning("Reflection provider %s missing API key; skipping reflection.", provider)
        else:
            try:
                reflection_runtime = (
                    "openai",
                    KimiClient(
                        api_key=api_key,
                        base_url=config.reflection_base_url or config.kimi_base_url,
                        model=config.reflection_model or config.kimi_model,
                        retry=config.reflection_retry or config.kimi_retry,
                        timeout=config.reflection_timeout or config.kimi_timeout,
                        log_path=config.reflection_log_path,
                    ),
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Reflection client init failed: %s", exc)

    def prepare_node(state: AgentState) -> AgentState:
        fields = state.get("fields") or {}
        query_text = build_query_text(fields)
        rules = rag_store.retrieve_rules(query_text, top_k=config.rule_top_k, min_score=config.rag_min_score)
        examples = rag_store.retrieve_examples(query_text, top_k=config.fewshot_top_k, min_score=config.rag_min_score)
        layer_c = list(dict.fromkeys(DEFAULT_LAYER_C_PATCHES + state.get("layer_c", [])))
        return {
            "rules": rules,
            "examples": examples,
            "layer_c": layer_c,
            "rule_lookup": rag_store.rule_lookup,
        }

    def kimi_node(state: AgentState) -> AgentState:
        record_id = state["record_id"]
        fields = state.get("fields") or {}
        layer_c = state.get("layer_c", DEFAULT_LAYER_C_PATCHES)
        messages = build_verdict_messages(record_id, fields, state.get("rules", []), layer_c, state.get("examples", []))
        chat_messages = [ChatMessage(**msg) for msg in messages]
        text = ""
        try:
            response = kimi.chat(
                chat_messages,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                stream=True,
            )
            text = response["text"]
            parsed = _safe_parse_json(text)
        except (KimiClientError, json.JSONDecodeError) as exc:
            logger.error("Kimi failed for %s: %s", record_id, exc)
            parsed = {"verdicts": [], "fixes": [], "layer_c_patch": [f"Kimi错误:{exc}"]}
        layer_patch = _ensure_list(parsed.get("layer_c_patch"))
        updated_layer_c = list(dict.fromkeys(layer_c + layer_patch))
        return {
            "kimi_payload": parsed,
            "raw_text": text,
            "layer_c": updated_layer_c,
        }

    def reflection_node(state: AgentState) -> AgentState:
        parsed = state.get("kimi_payload", {})
        fields = state.get("fields") or {}
        verdicts = _normalize_verdicts(_ensure_list(parsed.get("verdicts")), state.get("rule_lookup", {}), fields)
        fixes_raw = _ensure_list(parsed.get("fixes"))
        layer_c = state.get("layer_c", DEFAULT_LAYER_C_PATCHES)
        audit: Dict[str, Any] = {}
        suspects = _find_suspects(verdicts, config)
        if reflection_runtime and suspects:
            prompt = build_reflection_prompt(state["record_id"], suspects, layer_c)
            provider, client = reflection_runtime
            audit["reflection_provider"] = provider
            try:
                if provider == "gemini":
                    reflection = client.generate(prompt)
                    text_payload = reflection.text
                else:
                    messages = [ChatMessage(role="user", content=prompt)]
                    resp = client.chat(messages, temperature=0.05, max_output_tokens=800)
                    text_payload = resp["text"]
                audit["reflection_raw"] = text_payload
                data = _safe_parse_json(text_payload)
                verdicts = _apply_reflections(verdicts, _ensure_list(data.get("revisions")))
                patch_notes = _ensure_list(data.get("patch_notes"))
                if patch_notes:
                    layer_c.extend(str(note) for note in patch_notes if note)
                audit["reflection"] = data.get("why_short", "")
            except (GeminiClientError, KimiClientError, json.JSONDecodeError) as exc:  # pragma: no cover
                audit["reflection_error"] = str(exc)
        return {
            "verdicts": verdicts,
            "fixes_raw": fixes_raw,
            "layer_c": layer_c,
            "audit": audit,
        }

    def compile_node(state: AgentState) -> AgentState:
        result = _derive_outputs(
            record_id=state["record_id"],
            verdicts=state.get("verdicts", []),
            fixes_raw=state.get("fixes_raw", []),
            fields=state.get("fields") or {},
            config=config,
            layer_c_patch=state.get("layer_c", []),
            audit=state.get("audit", {}),
        )
        return {"llm_response": result, "raw_text": state.get("raw_text", "")}

    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare_node)
    graph.add_node("kimi", kimi_node)
    graph.add_node("reflect", reflection_node)
    graph.add_node("compile", compile_node)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "kimi")
    graph.add_edge("kimi", "reflect")
    graph.add_edge("reflect", "compile")
    graph.add_edge("compile", END)
    return graph.compile()
