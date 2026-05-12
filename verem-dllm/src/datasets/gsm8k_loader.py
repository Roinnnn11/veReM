from datasets import load_dataset
from typing import List


GSM8K_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "At the end, write your final answer after '####'.\n\n"
    "Problem: {question}"
)


def load_gsm8k(split: str = "test", num_samples: int = None, seed: int = 42) -> List[dict]:
    ds = load_dataset("gsm8k", "main", split=split)
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
