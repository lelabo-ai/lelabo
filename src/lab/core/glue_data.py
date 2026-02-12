# lab/core/glue_data.py
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

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

def make_glue_loaders(
    task_name: str,
    model_name: str,
    batch_size: int = 16,
    max_length: int = 128,
    seed: int = 42,
    num_workers: int = 0,  # keep 0 by default for max compatibility
):
    """
    Returns:
      train_loader, val_loaders(dict), num_labels, is_regression
    Batch format returned by loaders:
      {"input_ids":..., "attention_mask":..., (token_type_ids):..., "labels":...}
    """
    task_name = task_name.lower()
    if task_name not in GLUE_KEYS:
        raise ValueError(f"Unknown GLUE task: {task_name}")

    raw = load_dataset("glue", task_name)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    s1, s2 = GLUE_KEYS[task_name]

    def tokenize(examples):
        if s2 is None:
            return tok(examples[s1], truncation=True, max_length=max_length)
        return tok(examples[s1], examples[s2], truncation=True, max_length=max_length)

    encoded = raw.map(tokenize, batched=True)

    # Rename label -> labels (keep as Python scalars)
    def rename_label(ex):
        ex["labels"] = ex["label"]
        return ex

    encoded = encoded.map(rename_label)

    # Force python formatting to avoid NumPy2 / torch_formatter path
    encoded = encoded.with_format("python")

    is_regression = (task_name == "stsb")
    num_labels = 1 if is_regression else raw["train"].features["label"].num_classes

    # Collator pads input_ids/attention_mask/(token_type_ids)
    pad_collator = DataCollatorWithPadding(tokenizer=tok, return_tensors="pt")

    def collate_fn(batch):
        # labels
        labels_list = [ex["labels"] for ex in batch]
        if is_regression:
            labels = torch.tensor(labels_list, dtype=torch.float32)
        else:
            labels = torch.tensor(labels_list, dtype=torch.long)

        # Keep ONLY model input fields (avoid raw text columns)
        allowed = {"input_ids", "attention_mask", "token_type_ids"}
        features = []
        for ex in batch:
            d = {k: ex[k] for k in ex.keys() if k in allowed}
            features.append(d)

        out = pad_collator(features)  # pads to same length + creates tensors
        out["labels"] = labels
        return out


    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        encoded["train"],
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loaders = {}
    if task_name == "mnli":
        for split in ["validation_matched", "validation_mismatched"]:
            val_loaders[split] = DataLoader(
                encoded[split],
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
            )
    else:
        val_loaders["validation"] = DataLoader(
            encoded["validation"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loaders, num_labels, is_regression
