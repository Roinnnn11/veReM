import os
from datasets import load_dataset
from typing import List


GSM8K_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "At the end, write your final answer after '####'.\n\n"
    "Problem: {question}"
)

# 本地数据集路径，可通过环境变量 GSM8K_LOCAL_PATH 覆盖
_LOCAL_PATH = os.environ.get("GSM8K_LOCAL_PATH", "/mnt/data/gsm8k")


def _load_local_or_remote(config: str, split: str):
    local_dir = os.path.join(_LOCAL_PATH, config)
    if os.path.isdir(local_dir):
        return load_dataset("parquet", data_files={split: os.path.join(local_dir, f"{split}-*.parquet")}, split=split)
    # 无本地数据时才尝试网络
    return load_dataset("gsm8k", config, split=split)


def load_gsm8k(split: str = "test", num_samples: int = None, seed: int = 42) -> List[dict]:
    ds = _load_local_or_remote("main", split=split)
    if num_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))

    examples = []
    for i, item in enumerate(ds):
        examples.append({
            "id": f"gsm8k_{split}_{i}",
            "prompt": GSM8K_PROMPT_TEMPLATE.format(question=item["question"]),
            "gold": item["answer"],
            "question": item["question"],
        })
    return examples
