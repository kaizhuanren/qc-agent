"""Prompt builders for the QC agent."""

from __future__ import annotations

from typing import List, Dict, Any
import json

from .rag import RuleDocument, FewShotExample


SYSTEM_PROMPT = (
    "你是一名熟悉中国住院病历质控规范的医疗质控专家。"
    "需要根据提供的字段内容、标准质控规则以及示例，找出病历中的问题并输出 JSON。"
    "请遵守：\n"
    "1. 严格依据病历文字和规则，不要臆造信息。\n"
    "2. 优先命中最明确、可落地的规则；若信息不足，描述理由并降低置信度。\n"
    "3. 输出 JSON：{\"problems\": [...], \"fixes\": [...]}。每个 problem 含 field, issue_type, rule_id, description, confidence。\n"
    "4. 校正建议（fixes）面向临床医生，只可补写允许的占位词（如“不详/未查/无”），不得捏造检查结果。\n"
)


def _format_rules(rules: List[RuleDocument]) -> str:
    lines = []
    for idx, rule in enumerate(rules, start=1):
        lines.append(f"规则 {idx}:")
        lines.append(rule.text)
        lines.append("")
    return "\n".join(lines).strip()


def _format_examples(examples: List[FewShotExample]) -> str:
    if not examples:
        return "无示例。"
    blocks = []
    for idx, ex in enumerate(examples, start=1):
        output = json.dumps(ex.problems, ensure_ascii=False, indent=2) if ex.problems else "[]"
        blocks.append(
            f"示例 {idx} (record_id={ex.record_id})\n"
            f"输入字段摘要：\n{ex.text}\n"
            f"输出 problems：\n{output}\n"
        )
    return "\n".join(blocks).strip()


def _format_fields(record_id: str, fields: Dict[str, Any]) -> str:
    ordered = [
        ("chief_complaint", "主诉"),
        ("history_present", "现病史"),
        ("history_past", "既往史"),
        ("personal_history", "个人史"),
        ("family_history", "家族史"),
        ("special_exam", "专科检查"),
        ("lab_exam", "辅助检查"),
        ("admission_diagnosis", "入院诊断"),
        ("preliminary_diagnosis", "初步诊断"),
        ("primary_admission_dx", "入院主诊断"),
    ]
    lines = [f"record_id: {record_id}"]
    for key, label in ordered:
        value = fields.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, list):
            text = "；".join(map(str, value))
        else:
            text = str(value)
        lines.append(f"{label}：{text}")
    return "\n".join(lines)


def build_messages(
    record_id: str,
    fields: Dict[str, Any],
    rules: List[RuleDocument],
    examples: List[FewShotExample],
) -> List[Dict[str, str]]:
    rule_block = _format_rules(rules) or "无命中规则，按常规质控判断。"
    example_block = _format_examples(examples)
    field_block = _format_fields(record_id, fields)
    user_content = (
        "【任务】\n"
        "依据“规则知识库”审阅“待质控病历”，列出所有需要提示的问题以及整改建议。\n\n"
        f"【规则知识库】\n{rule_block}\n\n"
        f"【示例】\n{example_block}\n\n"
        f"【待质控病历】\n{field_block}\n\n"
        "【输出要求】\n"
        "1. 只输出 JSON；字段顺序不限。\n"
        "2. problems 中若无法确定某条规则是否命中，可在 description 中说明原因并设定 confidence 为 0.4 以下。\n"
        "3. fixes 至少给出一条操作性建议；若无问题可提示\"不需修改\"。\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
