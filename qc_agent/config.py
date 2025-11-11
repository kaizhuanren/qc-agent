"""Configuration helpers for the EMR QC agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
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
    kimi_model: str = field(default_factory=lambda: os.getenv("KIMI_MODEL", "kimi-k2-0905-preview"))
    kimi_retry: int = 3
    kimi_timeout: int = 120

    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_rpm_limit: int = 1
    gemini_max_retries: int = 3

    reflection_provider: str = field(default_factory=lambda: os.getenv("REFLECTION_PROVIDER", "gemini"))
    reflection_model: str = field(default_factory=lambda: os.getenv("REFLECTION_MODEL", "kimi-k2-thinking"))
    reflection_api_key: str = field(default_factory=lambda: os.getenv("REFLECTION_API_KEY", ""))
    reflection_base_url: str = field(default_factory=lambda: os.getenv("REFLECTION_BASE_URL", "https://api.moonshot.cn/v1"))
    reflection_timeout: int = field(default_factory=lambda: int(os.getenv("REFLECTION_TIMEOUT", "120")))
    reflection_retry: int = field(default_factory=lambda: int(os.getenv("REFLECTION_RETRY", "3")))

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
    model_config_path: Path = field(default_factory=lambda: Path(os.getenv("MODEL_CONFIG_PATH", "config/models.json")))
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", "logs")))
    kimi_log_path: Path = field(init=False)
    reflection_log_path: Path = field(init=False)

    def __post_init__(self) -> None:
        model_config_path = Path(self.model_config_path)
        object.__setattr__(self, "model_config_path", model_config_path)
        self._apply_model_overrides()
        log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "log_dir", log_dir)
        object.__setattr__(self, "kimi_log_path", log_dir / "kimi_requests.log")
        object.__setattr__(self, "reflection_log_path", log_dir / "reflection_requests.log")
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

    @property
    def has_reflection_key(self) -> bool:
        provider = (self.reflection_provider or 'gemini').lower()
        if provider == 'gemini':
            return self.has_gemini_key
        if provider in {'kimi', 'openai'}:
            key = (self.reflection_api_key or self.kimi_api_key).strip()
            return bool(key)
        return False

    def _apply_model_overrides(self) -> None:
        config_path = self.model_config_path
        if not config_path.exists():
            return
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return
        request_model = str(data.get("request_model") or "").strip()
        if request_model:
            object.__setattr__(self, "kimi_model", request_model)
        reflection_model = str(data.get("reflection_model") or "").strip()
        if reflection_model:
            object.__setattr__(self, "reflection_model", reflection_model)
        reflection_provider = str(data.get("reflection_provider") or "").strip()
        if reflection_provider:
            object.__setattr__(self, "reflection_provider", reflection_provider)

