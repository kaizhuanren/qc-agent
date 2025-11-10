"""Naive RAG helpers: load rule docs + few-shot examples and retrieve them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import math
import re


TOKEN_RE = re.compile(r"[A-Za-z]+|[\u4e00-\u9fff]|\d+")


def tokenize(text: str | None) -> List[str]:
    if not text:
        return []
    lowered = text.lower()
    return TOKEN_RE.findall(lowered)


@dataclass
class RuleDocument:
    rule_id: str
    issue_type: str
    field_en: str
    field_ch: str
    text: str


@dataclass
class FewShotExample:
    record_id: str
    text: str
    problems: List[Dict[str, Any]]


class BM25Retriever:
    def __init__(self, docs: List[str]) -> None:
        self.docs = docs
        self.tokens = [tokenize(doc) for doc in docs]
        self.doc_freq: Dict[str, int] = {}
        for tokens in self.tokens:
            for token in set(tokens):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        self.avg_len = sum(len(toks) for toks in self.tokens) / len(self.tokens) if self.tokens else 0.0
        self.k1 = 1.5
        self.b = 0.75

    def _idf(self, token: str) -> float:
        n_docs = len(self.tokens)
        df = self.doc_freq.get(token, 0)
        return math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

    def _score_doc(self, doc_idx: int, query_tokens: List[str]) -> float:
        tokens = self.tokens[doc_idx]
        if not tokens:
            return 0.0
        freq: Dict[str, int] = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        doc_len = len(tokens)
        score = 0.0
        for token in query_tokens:
            tf = freq.get(token, 0)
            if tf == 0:
                continue
            idf = self._idf(token)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_len or 1))
            score += idf * numerator / denominator
        return score

    def search(self, query_text: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if not self.docs:
            return []
        query_tokens = tokenize(query_text)
        if not query_tokens:
            return []
        scores = [(idx, self._score_doc(idx, query_tokens)) for idx in range(len(self.docs))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_split_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[obj["record_id"]] = obj
    return rows


def _format_fields(fields: Dict[str, Any]) -> str:
    keys = [
        "chief_complaint",
        "history_present",
        "history_past",
        "special_exam",
        "lab_exam",
        "admission_diagnosis",
        "preliminary_diagnosis",
    ]
    parts = []
    for key in keys:
        value = fields.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            text = "；".join(map(str, value))
        else:
            text = str(value)
        parts.append(f"{key}: {text}")
    return "\n".join(parts)


class RAGStore:
    def __init__(self, rules_path: Path, train_records_path: Path, train_split_path: Path) -> None:
        self.rules = self._load_rules(rules_path)
        self.rule_retriever = BM25Retriever([rule.text for rule in self.rules])
        self.examples = self._load_examples(train_records_path, train_split_path)
        self.example_retriever = BM25Retriever([ex.text for ex in self.examples]) if self.examples else None

    def _load_rules(self, path: Path) -> List[RuleDocument]:
        docs: List[RuleDocument] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                block = "\n".join(
                    [
                        f"rule_id: {obj.get('rule_id')}",
                        f"issue_type: {obj.get('issue_type')}",
                        f"field: {obj.get('field_ch')} ({obj.get('field_en')})",
                        f"definition: {obj.get('definition') or '无'}",
                        f"qc_logic: {obj.get('qc_logic') or '无'}",
                        f"typical_defects: {obj.get('typical_defects') or '无'}",
                    ]
                )
                docs.append(
                    RuleDocument(
                        rule_id=obj.get("rule_id", ""),
                        issue_type=obj.get("issue_type", ""),
                        field_en=obj.get("field_en", ""),
                        field_ch=obj.get("field_ch", ""),
                        text=block,
                    )
                )
        return docs

    def _load_examples(self, train_records_path: Path, train_split_path: Path) -> List[FewShotExample]:
        records = _load_json(train_records_path)
        split_rows = _load_split_jsonl(train_split_path)
        rows: List[FewShotExample] = []
        for record in records:
            rid = record.get("record_id")
            split_fields = split_rows.get(rid, {}).get("fields", {})
            field_block = _format_fields(split_fields)
            problems = record.get("problems", [])
            text = f"record_id: {rid}\n{field_block}"
            rows.append(FewShotExample(record_id=rid, text=text, problems=problems))
        return rows

    def retrieve_rules(self, text: str, top_k: int = 5, min_score: float = 0.0) -> List[RuleDocument]:
        hits = self.rule_retriever.search(text, top_k=top_k)
        results = []
        for idx, score in hits:
            if score < min_score:
                continue
            results.append(self.rules[idx])
        return results

    def retrieve_examples(self, text: str, top_k: int = 3, min_score: float = 0.0) -> List[FewShotExample]:
        if not self.example_retriever:
            return []
        hits = self.example_retriever.search(text, top_k=top_k)
        results = []
        for idx, score in hits:
            if score < min_score:
                continue
            results.append(self.examples[idx])
        return results


def build_query_text(fields: Dict[str, Any]) -> str:
    chunks = []
    for key in ["chief_complaint", "history_present", "special_exam", "lab_exam", "admission_diagnosis", "preliminary_diagnosis"]:
        value = fields.get(key)
        if not value:
            continue
        if isinstance(value, list):
            value = "；".join(map(str, value))
        chunks.append(f"{key}:{value}")
    return "\n".join(chunks)
