"""Configuration helpers for the EMR QC agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass(frozen=True)
class AgentConfig:
    """Holds configurable paths and default hyper parameters."""

    data_dir: Path = field(default_factory=lambda: Path("data"))
    train_records_path: Path = field(init=False)
    test_records_path: Path = field(init=False)
    train_split_path: Path = field(init=False)
    test_split_path: Path = field(init=False)
    rules_path: Path = field(init=False)

    kimi_api_key: str = field(default_factory=lambda: os.getenv("KIMI_API_KEY", ""))
    kimi_base_url: str = field(default_factory=lambda: os.getenv("KIMI_API_BASE", "https://api.moonshot.cn/v1"))
    kimi_model: str = field(default_factory=lambda: os.getenv("KIMI_MODEL", "moonshot-v1-32k"))

    rule_top_k: int = 6
    fewshot_top_k: int = 3
    temperature: float = 0.15
    max_output_tokens: int = 1200
    rag_min_score: float = 0.08

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_records_path", self.data_dir / "train.json")
        object.__setattr__(self, "test_records_path", self.data_dir / "test.json")
        object.__setattr__(self, "train_split_path", self.data_dir / "train_emr_split.json")
        object.__setattr__(self, "test_split_path", self.data_dir / "test_emr_split.json")
        object.__setattr__(self, "rules_path", self.data_dir / "rules.jsonl")

    @property
    def has_kimi_key(self) -> bool:
        return bool(self.kimi_api_key.strip())
