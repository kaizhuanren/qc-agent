# 🏥 RAG-Based EMR Quality Control Agent

本项目构建了一个结合 **规则 + LLM** 的混合式「病历质控校正 Agent」，主要功能为对电子病历（EMR）进行自动质控与校正，输出包含问题识别、修改建议及可选修正版病历的结果。

---

## 💡 功能概述

该 Agent 基于以下要素构建：

* **结构化数据**：`train/test` 病历与标注数据
* **规则文件**：`data/rules.jsonl`（定义质控规则与约束）
* **LangGraph 流水线**：RAG 检索 → few-shot 示例 → Kimi API 调用 → JSON 解析

最终可批量输出以下字段：

* `problems`（发现的问题）
* `fixes`（修复建议）
* `corrected_emr`（可选修正版）

> ⚠️ 注意：所有 LLM 调用依赖 Kimi（Moonshot）API。运行前需确保拥有合法的 API Key，并符合单位内网安全要求。

---

## 📂 目录结构

```
rag_med/
├── qc_agent/                 # Agent 核心代码
│   ├── config.py             # 全局配置（路径、Kimi 参数等）
│   ├── kimi.py               # Moonshot API 客户端
│   ├── rag.py                # 规则/示例检索逻辑
│   ├── prompts.py            # 提示词模板
│   └── agent.py              # LangGraph 主流程（prepare → llm）
│
├── data/                     # 数据与规则
│   ├── train.json / test.json
│   ├── train_emr_split.json / test_emr_split.json
│   └── rules.jsonl
│
├── run_agent.py              # CLI 启动入口
└── README.md
```

---

## ⚙️ 环境准备

### 1. Python 环境

* **Python 3.9+**（macOS 自带版本即可）

### 2. 安装依赖

```bash
pip3 install langgraph langchain-core langchain-text-splitters numpy scikit-learn requests
```

### 3. 设置 Kimi 环境变量

```bash
export KIMI_API_KEY="sk-xxx"

# 可选项：自定义域名与模型（默认模型为 kimi-k2-0905-preview）
export KIMI_API_BASE="https://api.moonshot.cn/v1"
export KIMI_MODEL="kimi-k2-0905-preview"
```

### 🔧 使用 `config/models.json` 快速切换模型

不想频繁改环境变量时，可在 `config/models.json` 中配置请求与反思模型，例如：

```json
{
  "request_model": "kimi-k2-turbo-preview",
  "reflection_model": "kimi-k2-turbo-preview",
  "reflection_provider": "kimi"
}
```

该文件优先级高于默认值，也可通过 `MODEL_CONFIG_PATH` 指向自定义路径；若需回到 Gemini 反思，把 `reflection_provider` 改为 `gemini` 即可。

---

## 🚀 运行 Agent

脚本会自动执行以下步骤：

1. 读取指定数据集（`train_emr_split.json` 或 `test_emr_split.json`）
2. 用 **BM25** 检索最相关的规则与 few-shot 示例
3. 构建 prompt 并调用 Kimi API
4. 解析并保存 JSON 输出至 `outputs/*.jsonl`

### 示例：运行前 5 条测试数据

```bash
python3 run_agent.py \
  --dataset test \
  --limit 5 \
  --output outputs/test_pred.jsonl
```

### 查看输出结果

```bash
head outputs/test_pred.jsonl
```

---

## 📊 输出结果示例

```json
{
  "record_id": "MED_QC_ADM_0001",
  "problems": [
    {
      "field": "history_present",
      "issue_type": "CONSISTENCY",
      "rule_id": "CO-XB-02-V1",
      "description": "...",
      "confidence": 0.82
    }
  ],
  "fixes": [
    "【病史一致性】若现病史写了手术/置管 ..."
  ],
  "raw_text": "{LLM 原始 JSON 字符串}"
}
```

> 你可以基于 `raw_text` 字段进行二次解析，追加 `corrected_emr` 输出。

---

## 🧩 自定义与调试

* **RAG 检索参数**：可在 `qc_agent/config.py` 中修改

  * `rule_top_k`
  * `fewshot_top_k`
  * `rag_min_score`
* **Prompt 扩展**：`qc_agent/prompts.py` 中可加入医学知识包或约束
* **语料补充**：在 `data/rules.jsonl` 与 `data/train.json` 中添加更多规则与示例
* **本地调试（无 API Key）**：
  在 `qc_agent/kimi.py` 中 mock `KimiClient.chat` 返回固定 JSON，即可快速跑通流程

---

## 🧠 常见问题

| 问题                        | 解决方法                                    |
| ------------------------- | --------------------------------------- |
| `KIMI_API_KEY is missing` | 确认已设置环境变量或在 `AgentConfig` 中填写           |
| SSL / LibreSSL 警告         | macOS 自带 Python 触发，可忽略或升级 OpenSSL       |
| JSON 解析失败                 | `agent.py` 已内置容错；可在 prompt 强调“仅输出 JSON” |

---

## 🔮 后续优化方向

1. **融合本地规则引擎**：降低 LLM 幻觉率
2. **优化输出结构**：将 `fixes` 拆为「整改要点 + 模板文稿」
3. **基于标注集误差分析**：计算命中率 / 误报率，迭代 prompt 与检索语料


## ACE-QC (Agentic Context Engineering) Pipeline

- **Layer A (契约层)：** 在提示中锁定输出契约、规则白名单与禁止事项。
- **Layer B (事实层)：** 按病历裁剪的字段/结构化事实。
- **Layer C (补丁层)：** 反思器产出的最小修复提示，逐批演化。
- **多代理协同：** Kimi 负责抽取/规则评审/生成 verdicts；Gemini 仅在疑难或家族越界时反思与纠错。
- **产出契约：** `verdicts` → `problems/notes` 由系统派生，永不让 LLM 直接写 problems。

## Dual-LLM Setup

```bash
export KIMI_API_KEY="sk-..."
export GEMINI_API_KEY="ya29...."
pip3 install google-genai
```

默认 rpm<=1：Gemini 客户端带速率限制与指数回退；Kimi 客户端自带空响应/超时重试。可在 `qc_agent/config.py` 调整模型名与重试策略。


### Reflection Provider选项
- `REFLECTION_PROVIDER`: `gemini` (默认) 或 `kimi`/`openai`，用于指定反思器模型来源。
- `REFLECTION_MODEL`: 反思模型名，例如 `gemini-2.5-flash` 或 `kimi-k2-thinking`。
- `REFLECTION_API_KEY`/`REFLECTION_BASE_URL`: 当使用 openai/kimi 兼容接口时指定；默认为主 Kimi 配置。
- 运行示例：
```bash
export REFLECTION_PROVIDER=kimi
export REFLECTION_MODEL=kimi-k2-thinking
export REFLECTION_API_KEY="$KIMI_API_KEY"
```
反思层会在低置信/家族越界案件上调用该模型做二次校对。

### 🪵 请求日志

默认会将每次 Kimi 调用记录到 `logs/kimi_requests.log`（反思阶段写入 `logs/reflection_requests.log`），其中包含模型、请求摘要、响应片段及错误，便于定位“返回为空/超时”等问题。可通过 `LOG_DIR` 或 `MODEL_CONFIG_PATH` 指向自定义目录。

