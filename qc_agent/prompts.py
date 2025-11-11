"""Prompt builders for the ACE-QC agent."""

from __future__ import annotations

from typing import List, Dict, Any
import json

from .rag import RuleDocument, FewShotExample


CONTRACT_SNIPPET = (
    "【ACE-QC 输出契约】
"
    "- 严格输出 JSON；不得直接生成 problems。
"
    "- verdicts: 每条候选规则给 PASS/FAIL/N/A，含 rule_id/field/issue_type/confidence/evidence/short_reason。
"
    "- problems: 系统层将 verdicts 中 FAIL 派生；模型不可写。
"
    "- notes: PASS 或 N/A 的说明。
"
    "- fixes: 可执行建议，允许使用‘未详/不详/未予特殊处理’等占位词。
"
    "- metrics: 至少给出 deduction_total 与 summary。
"
    "- 证据优先：无证据不得 FAIL；引用原文片段或段落名。
"
    "- rule_id 必须来自系统列出的白名单。
"
)

LAYER_C_BASE = [
    "不得直接生成 problems，需由系统从 FAIL verdicts 派生",
    "诊断遗漏/检查提示但未写入诊断 → IC-RZCB-01-V1",
    "主诉无法导出第一诊断 → DQ-RZ-01-V1",
    "以症状代替诊断 → DQ-RZ-02-V1",
    "辅助检查缺少时间/地点/项目/结果任一 → EN-FZ-01-V1",
    "现病史出现治疗但缺时间/地点/方式任一 → IC-XB-01-V1",
]


def _format_rule_catalog(rules: List[RuleDocument]) -> str:
    lines = []
    for idx, rule in enumerate(rules, start=1):
        lines.append(
            f"{idx}. rule_id={rule.rule_id} | issue_type={rule.issue_type} | field={rule.field_en}
"
            f"   摘要：{rule.text.splitlines()[3] if rule.text else ''}"
        )
    return "
".join(lines) if lines else "无匹配规则"


def _format_examples(examples: List[FewShotExample]) -> str:
    if not examples:
        return "无示例"
    blocks = []
    for idx, ex in enumerate(examples, start=1):
        output = json.dumps(ex.problems, ensure_ascii=False, indent=2) if ex.problems else "[]"
        blocks.append(
            f"示例 {idx} (record_id={ex.record_id})
"
            f"字段摘要：
{ex.text}
"
            f"历史 problems：{output}
"
        )
    return "
".join(blocks)


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
        text = "；".join(map(str, value)) if isinstance(value, list) else str(value)
        lines.append(f"{label}：{text}")
    return "
".join(lines)


def build_verdict_messages(
    record_id: str,
    fields: Dict[str, Any],
    rules: List[RuleDocument],
    layer_c_patches: List[str],
    examples: List[FewShotExample] | None = None,
) -> List[Dict[str, str]]:
    rule_block = _format_rule_catalog(rules)
    example_block = _format_examples(examples or [])
    field_block = _format_fields(record_id, fields)
    layer_c = LAYER_C_BASE + layer_c_patches
    user_content = (
        "【Layer A：任务契约】
" + CONTRACT_SNIPPET + "

"
        "【Layer B：记录事实】
" + field_block + "

"
        "【Layer C：最新补丁】
" + "
".join(layer_c) + "

"
        "【候选规则白名单】
" + rule_block + "

"
        "【示例参考】
" + example_block + "

"
        "【任务】
"
        "逐条评审上述候选规则，输出 JSON：{"verdicts": [...], "fixes": [...], "layer_c_patch": [...]}。
"
        "- verdicts 仅包含 PASS/FAIL/N/A；不得生成 problems。
"
        "- 每条 verdict 字段：rule_id, field, issue_type, verdict, confidence, evidence_spans[], short_reason。
"
        "- confidence 范围 0~1；证据需引用原文片段或段落名。
"
        "- 建议 (fixes) 可为字符串或对象，需对应 rule_id。
"
        "- 可在 layer_c_patch 中追加新的最小补丁提醒（可为空）。
"
    )
    return [
        {"role": "system", "content": CONTRACT_SNIPPET},
        {"role": "user", "content": user_content},
    ]


def build_reflection_prompt(
    record_id: str,
    suspect_verdicts: List[Dict[str, Any]],
    layer_c_patches: List[str],
) -> str:
    payload = json.dumps(suspect_verdicts, ensure_ascii=False, indent=2)
    layer_c = LAYER_C_BASE + layer_c_patches
    return (
        "你是 ACE-QC 反思器，负责纠错与补丁。
"
        "输入是同一病历可疑 verdicts。请：
"
        "1) 如判错，给出 revised verdict（rule_id/field/verdict/confidence/evidence/short_reason）。
"
        "2) 如果仅需备注，写 reason 并维持 verdict。
"
        "3) 输出 JSON：{"revisions": [...], "patch_notes": [...], "why_short": "..."}
"
        "4) 只允许使用白名单 rule_id，不得生成 problems。
"
        "Layer C 补丁（供参考）：
" + "
".join(layer_c) + "

"
        f"record_id: {record_id}
"
        f"suspect_verdicts: {payload}
"
    )
