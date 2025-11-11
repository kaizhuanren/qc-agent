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
    kimi_retry: int = 3
    kimi_timeout: int = 120

    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_rpm_limit: int = 1
    gemini_max_retries: int = 3

    rule_top_k: int = 10
    fewshot_top_k: int = 3
    temperature: float = 0.15
    max_output_tokens: int = 1600
    rag_min_score: float = 0.08
    fail_conf_threshold: float = 0.55
    reflection_conf_min: float = 0.45
    reflection_conf_max: float = 0.6
    reflection_rule_whitelist: tuple[str, ...] = (
        "IC-RZCB-01-V1",
        "DQ-RZ-01-V1",
        "DQ-RZ-02-V1",
        "CO-XB-01-V1",
        "CO-XB-04-V1",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_records_path", self.data_dir / "train.json")
        object.__setattr__(self, "test_records_path", self.data_dir / "test.json")
        object.__setattr__(self, "train_split_path", self.data_dir / "train_emr_split.json")
        object.__setattr__(self, "test_split_path", self.data_dir / "test_emr_split.json")
        object.__setattr__(self, "rules_path", self.data_dir / "rules.jsonl")

    @property
    def has_kimi_key(self) -> bool:
        return bool(self.kimi_api_key.strip())

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key.strip())
