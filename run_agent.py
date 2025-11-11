#!/usr/bin/env python3
"""
Run the LangGraph-based EMR QC agent on train/test records.

Usage:
    python run_agent.py --dataset train --limit 2 --output outputs/train_pred.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from qc_agent import AgentConfig, build_qc_app


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_split_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[obj["record_id"]] = obj.get("fields", {})
    return rows


def build_dataset(records_file: Path, split_file: Path) -> List[Dict[str, Any]]:
    records = load_json(records_file)
    split_map = load_split_jsonl(split_file)
    dataset = []
    for record in records:
        rid = record["record_id"]
        fields = split_map.get(rid, {})
        if not fields:
            continue
        dataset.append(
            {
                "record_id": rid,
                "fields": fields,
                "problems": record.get("problems"),
            }
        )
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EMR QC agent.")
    parser.add_argument("--dataset", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=5, help="Number of records to process.")
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AgentConfig()
    if not config.has_kimi_key:
        raise SystemExit("Missing KIMI_API_KEY environment variable.")

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records_file = config.train_records_path if args.dataset == "train" else config.test_records_path
    split_file = config.train_split_path if args.dataset == "train" else config.test_split_path

    dataset = build_dataset(records_file, split_file)
    if args.limit > 0:
        dataset = dataset[: args.limit]

    app = build_qc_app(config)
    with output_path.open("w", encoding="utf-8") as writer:
        for record in dataset:
            state = {"record_id": record["record_id"], "fields": record["fields"]}
            result = app.invoke(state)
            llm_payload = result.get("llm_response", {})
            payload = {
                "record_id": record["record_id"],
                **llm_payload,
                "raw_text": result.get("raw_text", ""),
            }
            writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(
                f"Processed {record['record_id']} -> {len(llm_payload.get('problems', []))} problems, "
                f"{len(llm_payload.get('notes', []))} notes"
            )


if __name__ == "__main__":
    main()
