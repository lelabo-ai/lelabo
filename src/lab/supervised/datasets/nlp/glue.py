from __future__ import annotations

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from ..base import DataBundle, make_loader
from ..registry import register_dataset

GLUE_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


@register_dataset("glue")
def make_glue_dataset(
    *,
    glue_task: str,
    hf_model: str,
    batch_size: int = 16,
    max_length: int = 128,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
    **_: object,
) -> DataBundle:
    task_name = str(glue_task).lower()
    if task_name not in GLUE_KEYS:
        raise ValueError(f"Unknown GLUE task: {task_name}")

    raw = load_dataset("glue", task_name)
    tok = AutoTokenizer.from_pretrained(hf_model, use_fast=True)
    s1, s2 = GLUE_KEYS[task_name]

    def tokenize(examples):
        if s2 is None:
            return tok(examples[s1], truncation=True, max_length=max_length)
        return tok(examples[s1], examples[s2], truncation=True, max_length=max_length)

    encoded = raw.map(tokenize, batched=True)

    def rename_label(ex):
        ex["labels"] = ex["label"]
        return ex

    encoded = encoded.map(rename_label)
    encoded = encoded.with_format("python")

    is_regression = (task_name == "stsb")
    num_labels = 1 if is_regression else raw["train"].features["label"].num_classes

    pad_collator = DataCollatorWithPadding(tokenizer=tok, return_tensors="pt")

    def collate_fn(batch):
        labels_list = [ex["labels"] for ex in batch]
        if is_regression:
            labels = torch.tensor(labels_list, dtype=torch.float32)
        else:
            labels = torch.tensor(labels_list, dtype=torch.long)

        allowed = {"input_ids", "attention_mask", "token_type_ids"}
        features = []
        for ex in batch:
            features.append({k: ex[k] for k in ex.keys() if k in allowed})

        out = pad_collator(features)
        out["labels"] = labels
        return out

    train_loader = make_loader(
        encoded["train"],
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        seed_scope=f"glue.{task_name}.train",
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loaders = {}
    if task_name == "mnli":
        for split in ["validation_matched", "validation_mismatched"]:
            val_loaders[split] = make_loader(
                encoded[split],
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
                seed_scope=f"glue.{task_name}.{split}",
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
    else:
        val_loaders["validation"] = make_loader(
            encoded["validation"],
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            seed_scope=f"glue.{task_name}.validation",
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return DataBundle(
        train_loader=train_loader,
        val_loader=None,
        test_loader=None,
        num_classes=num_labels,
        meta={
            "val_loaders": val_loaders,
            "num_labels": num_labels,
            "is_regression": is_regression,
        },
    )
