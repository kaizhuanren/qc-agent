# -*- coding: utf-8 -*-
"""

- 仅校验 pred.json 的格式；
"""

import json
import os
import sys
import zipfile
import shutil
import glob
import logging
from typing import Dict, List, Tuple, Set


# ------------------------ 只校验 pred.json ------------------------

def _validate_pred_format(pred_json) -> Tuple[bool, str]:
    """
    仅对 pred.json 做格式校验：
      - 顶层必须 list
      - 每条包含 record_id(str)、problems(list)
      - problems[*] 必含非空字符串：field、issue_type、rule_id、description
    """
    if not isinstance(pred_json, list):
        return False, "pred.json 顶层必须为数组(list)。"

    for i, rec in enumerate(pred_json):
        if not isinstance(rec, dict):
            return False, f"pred.json 第{i}条应为对象。"
        rid = rec.get("record_id")
        probs = rec.get("problems")
        if not isinstance(rid, str) or not rid.strip():
            return False, f"第{i}条 record_id 缺失或不是非空字符串。"
        if not isinstance(probs, list):
            return False, f"第{i}条 problems 缺失或不是数组。"
        for j, p in enumerate(probs):
            if not isinstance(p, dict):
                return False, f"第{i}条 problems[{j}] 应为对象。"
            for key in ("field", "issue_type", "rule_id", "description"):
                val = p.get(key)
                if not isinstance(val, str) or not val.strip():
                    print (i,rec,p)
                    return False, f"第{i}条 problems[{j}].{key} 缺失或不是非空字符串。"
    return True, ""

# ------------------------ 主入口 ------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if len(sys.argv) != 2:
        out = _err_out(
            "校验程序入参错误：需要 pred.json",
            "Usage: python validate_pred_file.py pred.json"
        )
        return

    pred_path = sys.argv[1]
    
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_cases = json.load(f)
    
    ok, msg = _validate_pred_format(pred_cases)
    if not ok:
        print (msg)
    else:
    	print ('file format is ok.')

if __name__ == "__main__":
    main()
