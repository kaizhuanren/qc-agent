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

    request_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY") or os.getenv("KIMI_API_KEY", "")
    )
    request_base_url: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_BASE_URL")
        or os.getenv("KIMI_API_BASE")
        or "https://openrouter.ai/api/v1"
    )
    request_model: str = field(
        default_factory=lambda: os.getenv("REQUEST_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or os.getenv("KIMI_MODEL")
        or "openai/gpt-4o-mini"
    )
    request_retry: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_RETRY", os.getenv("OPENROUTER_RETRY", "3")))
    )
    request_timeout: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", os.getenv("OPENROUTER_TIMEOUT", "300")))
    )
    request_referer: str = field(default_factory=lambda: os.getenv("OPENROUTER_HTTP_REFERER", ""))
    request_app_name: str = field(default_factory=lambda: os.getenv("OPENROUTER_APP_NAME", ""))

    reflection_model: str = field(
        default_factory=lambda: os.getenv("REFLECTION_MODEL")
        or os.getenv("OPENROUTER_REFLECTION_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or "openai/gpt-4o-mini"
    )
    reflection_api_key: str = field(
        default_factory=lambda: os.getenv("REFLECTION_API_KEY") or os.getenv("OPENROUTER_REFLECTION_API_KEY", "")
    )
    reflection_base_url: str = field(
        default_factory=lambda: os.getenv("REFLECTION_BASE_URL")
        or os.getenv("OPENROUTER_REFLECTION_BASE")
        or os.getenv("OPENROUTER_BASE_URL")
        or os.getenv("KIMI_API_BASE")
        or "https://openrouter.ai/api/v1"
    )
    reflection_retry: int = field(
        default_factory=lambda: int(os.getenv("REFLECTION_RETRY", os.getenv("OPENROUTER_RETRY", "3")))
    )
    reflection_timeout: int = field(
        default_factory=lambda: int(os.getenv("REFLECTION_TIMEOUT", os.getenv("OPENROUTER_TIMEOUT", "300")))
    )

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
    request_log_path: Path = field(init=False)
    reflection_log_path: Path = field(init=False)

    def __post_init__(self) -> None:
        model_config_path = Path(self.model_config_path)
        object.__setattr__(self, "model_config_path", model_config_path)
        self._apply_model_overrides()
        log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "log_dir", log_dir)
        object.__setattr__(self, "request_log_path", log_dir / "openrouter_requests.log")
        object.__setattr__(self, "reflection_log_path", log_dir / "reflection_requests.log")
        object.__setattr__(self, "train_records_path", self.data_dir / "train.json")
        object.__setattr__(self, "test_records_path", self.data_dir / "test.json")
        object.__setattr__(self, "train_split_path", self.data_dir / "train_emr_split.json")
        object.__setattr__(self, "test_split_path", self.data_dir / "test_emr_split.json")
        object.__setattr__(self, "rules_path", self.data_dir / "rules.jsonl")

    @property
    def has_request_key(self) -> bool:
        return bool(self.request_api_key.strip())

    @property
    def reflection_api_token(self) -> str:
        return (self.reflection_api_key or self.request_api_key).strip()

    @property
    def has_reflection_model(self) -> bool:
        return bool(self.reflection_model.strip())

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
            object.__setattr__(self, "request_model", request_model)
        reflection_model = str(data.get("reflection_model") or "").strip()
        if reflection_model:
            object.__setattr__(self, "reflection_model", reflection_model)
