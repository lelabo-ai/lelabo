# lab/models/bert_glue.py
from transformers import AutoModelForSequenceClassification

def build_bert_for_glue(model_name: str, num_labels: int):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
