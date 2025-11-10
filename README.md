# RAG-Based EMR Quality Control Agent

This project builds a hybrid规则+LLM「病历质控校正」Agent，结合：

- 已结构化的 train/test EMR 数据、规则说明 (`data/rules.jsonl`)
- LangGraph 流水线（RAG → few-shot → Kimi API 调整）

最终可批量输出 `problems`、`fixes`、可选 `corrected_emr` 等质控结果。

> ⚠️ 所有 LLM 调用依赖 Kimi（Moonshot）API。运行前请确保具备合法的 API KEY，并遵循单位内网合规要求。

---

## 目录结构

rag_med/
├── qc_agent/ # Agent 代码包
│ ├── config.py # 全局配置（路径、Kimi 参数等）
│ ├── kimi.py # Moonshot API 客户端
│ ├── rag.py # 规则/示例检索
│ ├── prompts.py # 系统+用户提示词模板
│ └── agent.py # LangGraph 流程（prepare → llm）
├── data/ # 已给定的 train/test、规则文件
│ ├── train.json / test.json # 原始记录 + 标注
│ ├── train_emr_split.json / test_emr_split.json # 已切分字段
│ └── rules.jsonl # 质控规则描述
├── run_agent.py # CLI 入口
└── README.md



---

## 环境准备

1. **Python 3.9+**（macOS 自带即可）
2. 安装依赖：
   ```bash
   pip3 install langgraph langchain-core langchain-text-splitters numpy scikit-learn requests
设置 Kimi 环境变量（终端）：
bash

export KIMI_API_KEY="sk-xxx"
# 可选：自定义域名/模型
export KIMI_API_BASE="https://api.moonshot.cn/v1"
export KIMI_MODEL="moonshot-v1-32k"
运行 Agent
脚本会：

根据 train_emr_split.json 或 test_emr_split.json 读取字段；
用 BM25 检索最相关的规则、few-shot 示例，嵌入 prompt；
调用 Kimi，解析 JSON 结果；
将预测写入 outputs/*.jsonl。
示例（跑前 5 条测试数据）：

bash

python3 run_agent.py \
  --dataset test \
  --limit 5 \
  --output outputs/test_pred.jsonl
查看输出：

bash

head outputs/test_pred.jsonl
结果格式
每条结果包含：

json

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
你可以基于 raw_text 做二次解析、追加 corrected_emr 输出。

自定义 & 调试
RAG 检索参数：在 qc_agent/config.py 中调整 rule_top_k、fewshot_top_k、rag_min_score 等。
Prompt 扩展：qc_agent/prompts.py 中可插入诊断知识包、更多约束。
规则/示例语料：data/rules.jsonl、data/train.json 可继续补充，以提升匹配。
本地调试：若暂时没有 Kimi key，可在 qc_agent/kimi.py 里 mock KimiClient.chat 返回固定 JSON，先走通流程。
常见问题
KIMI_API_KEY is missing：确保已导出环境变量，或直接修改 AgentConfig.
SSL/LibreSSL 警告：macOS 自带 Python 触发 urllib3 的提醒，可忽略或自行升级 OpenSSL。
JSON 解析失败：agent.py 已做容错；若频繁失败，可在 prompt 中加强“只输出 JSON”约束。
下一步建议
进一步补全规则实现（可在 LLM 结果接入本地规则引擎，降低 hallucination）。
将 fixes 拆分为“整改要点 + 模板文稿”两段输出，直接用于质控系统。
结合 train 标注进行误差分析（命中率/误报率），迭代 prompt 与 RAG 语料。s