# -*- coding: utf-8 -*-
"""
EMR splitter using local Qwen model (default) with a regex fallback.

- Loads a causal LLM from a local path (e.g., /share/StS/Qwen3-8B)
- Prompts it to return STRICT JSON containing common admission note fields
- If the LLM output is not valid JSON, falls back to regex-based section splitting
- Supports batch processing of a JSON/JSONL input file that contains records with an 'emr_content' field
- Writes results as JSONL, one parsed record per line

Usage:
    python emr_splitter.py --input data.jsonl --output parsed.jsonl --model /share/StS/Qwen3-8B
    python emr_splitter.py --input data.json --output parsed.jsonl --model /share/StS/Qwen3-8B --mode llm
    python emr_splitter.py --input data.jsonl --output parsed.jsonl --mode regex

Dependencies:
    pip install "transformers>=4.41" accelerate torch tqdm

Author: ChatGPT
"""
import argparse
import ast
import io
import json
import os
import re
from typing import Dict, List, Any, Optional

try:
    import torch  # type: ignore
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
    )  # type: ignore
except Exception:
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


CANONICAL_FIELDS = [
    "chief_complaint",           # 主诉
    "history_present",           # 现病史
    "history_past",              # 既往史
    "personal_history",          # 个人史
    "family_history",            # 家族史
    "menstrual_history",         # 月经史
    "marriage_childbearing",     # 婚育史/婚姻史
    "physical_exam",             # 体格检查
    "special_exam",              # 专科检查
    "lab_exam",                  # 辅助检查/检查检验
    "admission_diagnosis",       # 入院诊断
    "preliminary_diagnosis",     # 初步诊断/初始诊断
    "primary_admission_dx",      # 入院主诊断/初步主诊断/修正主诊断
    "revised_diagnosis",         # 修正诊断
    "narrator",                  # 病史陈述者
]

HEADER_MAP: Dict[str, str] = {
    "主诉": "chief_complaint",
    "现病史": "history_present",
    "既往史": "history_past",
    "既往病史": "history_past",
    "个人史": "personal_history",
    "家族史": "family_history",
    "月经史": "menstrual_history",
    "婚育史": "marriage_childbearing",
    "婚姻史": "marriage_childbearing",
    "体格检查": "physical_exam",
    "专科检查": "special_exam",
    "辅助检查": "lab_exam",
    "检查": "lab_exam",
    "实验室检查": "lab_exam",
    "入院诊断": "admission_diagnosis",
    "初步诊断": "preliminary_diagnosis",
    "初始诊断": "preliminary_diagnosis",
    "入院主诊断": "primary_admission_dx",
    "修正主诊断": "primary_admission_dx",
    "初步主诊断": "primary_admission_dx",
    "修正诊断": "revised_diagnosis",
    "病史陈述者": "narrator",
}

HEADER_REGEX = re.compile(
    r"(主诉|现病史|既往史|既往病史|个人史|家族史|月经史|婚育史|婚姻史|体格检查|专科检查|辅助检查|检查|实验室检查|入院诊断|初步诊断|初始诊断|入院主诊断|修正主诊断|初步主诊断|修正诊断|病史陈述者)\s*[：:]",
    flags=re.M | re.U,
)

BRACKETED_LIST_RE = re.compile(r"：\s*(\[[\s\S]*?\])")

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if text.startswith("{") or text.startswith("["):
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    out = []
                    if "train" in data and isinstance(data["train"], list):
                        out.extend(data["train"])
                    if "test" in data and isinstance(data["test"], list):
                        out.extend(data["test"])
                    if out:
                        return out
                    return [data]
                if isinstance(data, list):
                    return data
            except Exception:
                pass

    # JSONL fallback
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                rows.append({"emr_content": s})
    return rows


def try_parse_bracketed_list(text: str) -> Optional[List[str]]:
    m = BRACKETED_LIST_RE.search(text)
    if not m:
        return None
    try:
        return list(ast.literal_eval(m.group(1)))
    except Exception:
        s = m.group(1).replace("“", '"').replace("”", '"').replace("'", '"')
        try:
            return list(json.loads(s))
        except Exception:
            return None


def regex_split_sections(emr: str) -> Dict[str, Any]:
    sections: Dict[str, Any] = {k: "" for k in CANONICAL_FIELDS}
    positions = []

    for m in HEADER_REGEX.finditer(emr):
        header = m.group(1)
        start = m.end()
        positions.append((header, start))

    end_pos = len(emr)
    for idx, (hdr, start) in enumerate(positions):
        end = positions[idx + 1][1] - 1 if idx + 1 < len(positions) else end_pos
        chunk = emr[start:end].strip()
        key = HEADER_MAP.get(hdr)
        if not key:
            continue

        if key in ("admission_diagnosis", "preliminary_diagnosis", "primary_admission_dx"):
            maybe_list = try_parse_bracketed_list(emr[start-10:end])
            if maybe_list:
                sections[key] = [x.strip() for x in maybe_list if str(x).strip()]
            else:
                items = re.split(r"[，,；;、\n]+", chunk)
                items = [i.strip() for i in items if i.strip()]
                sections[key] = items
        else:
            sections[key] = chunk

    for zh, key in [("入院诊断", "admission_diagnosis"),
                    ("初步诊断", "preliminary_diagnosis"),
                    ("入院主诊断", "primary_admission_dx"),
                    ("修正诊断", "revised_diagnosis")]:
        if not sections.get(key):
            pat = re.compile(fr"{zh}\s*[：:]\s*(.+)")
            m = pat.search(emr)
            if m:
                tail = m.group(0)
                maybe = try_parse_bracketed_list(tail)
                if maybe:
                    sections[key] = [x.strip() for x in maybe if str(x).strip()]
                else:
                    text = m.group(1).strip()
                    items = [p.strip() for p in re.split(r"[，,；;、\n]+", text) if p.strip()]
                    sections[key] = items

    return sections


def build_prompt(emr: str) -> str:
    fields_cn = {
        "chief_complaint": "主诉",
        "history_present": "现病史",
        "history_past": "既往史",
        "personal_history": "个人史",
        "family_history": "家族史",
        "menstrual_history": "月经史",
        "marriage_childbearing": "婚育史/婚姻史",
        "physical_exam": "体格检查",
        "special_exam": "专科检查",
        "lab_exam": "辅助检查/检查检验",
        "admission_diagnosis": "入院诊断（数组）",
        "preliminary_diagnosis": "初步诊断（数组）",
        "primary_admission_dx": "入院主诊断/修正主诊断/初步主诊断（数组）",
        "revised_diagnosis": "修正诊断（数组）",
        "narrator": "病史陈述者",
    }
    schema_lines = [f'  "{k}": ""  # {v}' for k, v in fields_cn.items()]
    schema = "{\n" + ",\n".join(schema_lines) + "\n}"

    instructions = f"""
你是“入院记录结构化助手”。给你一段入院记录原文，请把内容拆分到固定字段，并以**严格的 JSON**返回，**不要**输出任何解释或多余文本。

要求：
1) 键名固定且全部返回（即便没有信息也给空字符串 "" 或空数组 []）。
2) 诊断类字段如果能拆成多条，请输出 **JSON 数组**，例如 ["血尿","泌尿系感染"]。
3) 文本中若出现多次同名小节，合并到同一键；保持原始表述，去掉多余空格与换行。
4) 仅返回 UTF-8 JSON，对象最外层键顺序可以任意。

字段与含义（示例 schema，注意只参考键名，不要把注释带入结果）：
{schema}

下面是待解析的入院记录原文，用三个反引号包裹：

按上述 schema 直接输出 JSON。
"""
    return instructions.strip()


class JsonStopper(StoppingCriteria):
    """
    看到结尾出现 '}'（可能后面跟少量空白）就停止，避免模型继续无意义补写。
    """
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self._pat = re.compile(r"}\s*$", flags=re.S)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True, errors="ignore")
        return bool(self._pat.search(text))


class QwenExtractor:
    def __init__(self, model_path: str,
                 gpus: Optional[str] = "0,1",
                 dtype: Optional[str] = "bf16",
                 device_map: Optional[str] = "auto"):
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise RuntimeError("transformers/torch is not available; install them or use --mode regex.")

        # 显式限定 GPU：默认使用 cuda:0 与 cuda:1
        if gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus  # e.g., "0,1"
            if tqdm:
                tqdm.write(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            # 对大多数 Qwen 权重，pad 与 eos 复用可避免生成时的 warning
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # dtype
        kwargs: Dict[str, Any] = dict(trust_remote_code=True)
        torch_dtype = None
        if dtype:
            d = dtype.lower()
            if d == "bf16":
                torch_dtype = getattr(torch, "bfloat16")
            elif d == "fp16":
                torch_dtype = getattr(torch, "float16")
            elif d == "fp32":
                torch_dtype = getattr(torch, "float32")
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype

        # 多 GPU 自动切分（在可见设备上做模块级切片）
        if device_map:
            kwargs["device_map"] = device_map  # "auto" 会在 CUDA:0/1 间平衡放置
        else:
            kwargs["device_map"] = "auto"

        # 更少 CPU 内存占用
        kwargs["low_cpu_mem_usage"] = True

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.eval()

        # 打印设备映射，确认确实把权重分布在 cuda:0 / cuda:1
        try:
            dm = getattr(self.model, "hf_device_map", None)
            if tqdm:
                tqdm.write(f"Model loaded from {model_path}")
                tqdm.write(f"Device map: {dm}")
                tqdm.write(f"torch_dtype: {self.model.dtype}")
            else:
                print(f"Model loaded from {model_path}")
                print(f"Device map: {dm}")
                print(f"torch_dtype: {self.model.dtype}")
        except Exception:
            pass

    def extract(self, emr_text: str, max_new_tokens: int = 512) -> Dict[str, Any]:
        with torch.inference_mode():
            prompt = build_prompt(emr_text)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.model.device)

            # 仅使用贪心解码；不传 temperature/top_k/top_p —— 彻底消除相关 warning
            stoppers = StoppingCriteriaList([JsonStopper(self.tokenizer)])
            gen_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                stopping_criteria=stoppers,
            )
            out = self.tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        json_obj = self._safe_json(out)
        if json_obj is None:
            return regex_split_sections(emr_text)
        return json_obj

    @staticmethod
    def _safe_json(text: str) -> Optional[Dict[str, Any]]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start:end+1]
        try:
            obj = json.loads(candidate)
            return normalize_result(obj)
        except Exception:
            repaired = candidate.replace("“", '"').replace("”", '"').replace("'", '"')
            repaired = re.sub(r",\s*}", "}", repaired)
            repaired = re.sub(r",\s*]", "]", repaired)
            try:
                obj = json.loads(repaired)
                return normalize_result(obj)
            except Exception:
                return None


def normalize_result(obj: Dict[str, Any]) -> Dict[str, Any]:
    cn_to_key = {
        "主诉": "chief_complaint",
        "现病史": "history_present",
        "既往史": "history_past",
        "个人史": "personal_history",
        "家族史": "family_history",
        "月经史": "menstrual_history",
        "婚育史": "marriage_childbearing",
        "婚姻史": "marriage_childbearing",
        "体格检查": "physical_exam",
        "专科检查": "special_exam",
        "辅助检查": "lab_exam",
        "检查": "lab_exam",
        "入院诊断": "admission_diagnosis",
        "初步诊断": "preliminary_diagnosis",
        "初始诊断": "preliminary_diagnosis",
        "入院主诊断": "primary_admission_dx",
        "修正主诊断": "primary_admission_dx",
        "初步主诊断": "primary_admission_dx",
        "修正诊断": "revised_diagnosis",
        "病史陈述者": "narrator",
    }
    normalized: Dict[str, Any] = {k: "" for k in CANONICAL_FIELDS}
    for k, v in obj.items():
        key = cn_to_key.get(k, k)
        if key not in normalized:
            normalized[key] = v
        else:
            normalized[key] = v

    for k in ("admission_diagnosis", "preliminary_diagnosis", "primary_admission_dx", "revised_diagnosis"):
        v = normalized.get(k, "")
        if isinstance(v, str):
            parts = [p.strip() for p in re.split(r"[，,；;、\n]+", v) if p.strip()]
            normalized[k] = parts
        elif v is None:
            normalized[k] = []
        elif isinstance(v, list):
            normalized[k] = [str(x).strip() for x in v if str(x).strip()]
        else:
            normalized[k] = [str(v).strip()] if str(v).strip() else []

    for k in CANONICAL_FIELDS:
        if k not in normalized:
            normalized[k] = "" if k not in ("admission_diagnosis","preliminary_diagnosis","primary_admission_dx","revised_diagnosis") else []
        elif normalized[k] is None:
            normalized[k] = "" if k not in ("admission_diagnosis","preliminary_diagnosis","primary_admission_dx","revised_diagnosis") else []

    return normalized


def process_records(rows: List[Dict[str, Any]], mode: str = "llm",
                    model_path: Optional[str] = None, max_new_tokens: int = 512,
                    gpus: Optional[str] = "0,1", dtype: Optional[str] = "bf16") -> List[Dict[str, Any]]:
    extractor = None
    if mode == "llm":
        if not model_path:
            raise ValueError("--model is required in llm mode")
        extractor = QwenExtractor(model_path, gpus=gpus, dtype=dtype)

    results = []
    iterable = tqdm(rows, desc="Processing", unit="rec") if tqdm else rows
    for row in iterable:
        emr = row.get("emr_content", "")
        rid = row.get("record_id", None)
        if not emr:
            results.append({"record_id": rid, "error": "empty emr_content"})
            continue

        if mode == "regex":
            parsed = regex_split_sections(emr)
        else:
            try:
                parsed = extractor.extract(emr, max_new_tokens=max_new_tokens)  # type: ignore
            except Exception as e:
                parsed = regex_split_sections(emr)
                warning = f"llm_failed: {e}"
                parsed["_warning"] = warning
                try:
                    if tqdm:
                        tqdm.write(f"Warning for record {rid}: {warning}")
                    else:
                        print(f"Warning for record {rid}: {warning}")
                except Exception:
                    pass

        results.append({"record_id": rid, "fields": parsed})
    return results


def main():
    parser = argparse.ArgumentParser(description="Split EMR admission note into fields using local Qwen or regex.")
    parser.add_argument("--input", required=True, help="Path to JSON/JSONL input containing records with 'emr_content'")
    parser.add_argument("--output", required=True, help="Path to JSONL output")
    parser.add_argument("--mode", choices=["llm", "regex"], default="llm", help="Use local model or regex fallback only")
    parser.add_argument("--model", default="/share/StS/Qwen3-8B", help="Local path to Qwen model")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens generated by the model")
    # 新增：明确 GPU 与 dtype，默认使用 cuda:0,1 + BF16
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma separated GPU ids to use, e.g. '0,1'")
    parser.add_argument("--dtype", choices=["bf16","fp16","fp32"], default="bf16", help="Inference dtype")
    args = parser.parse_args()

    rows = load_json_or_jsonl(args.input)
    results = process_records(
        rows, mode=args.mode, model_path=args.model, max_new_tokens=args.max_new_tokens,
        gpus=args.gpus, dtype=args.dtype
    )

    with open(args.output, "w", encoding="utf-8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
