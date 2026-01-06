# core/glue_task.py
import torch
import evaluate

class GLUETask:
    def __init__(self, task_name: str, is_regression: bool, num_labels: int):
        self.task_name = task_name.lower()
        self.is_regression = is_regression
        self.num_labels = num_labels
        self.metric = evaluate.load("glue", self.task_name)

    @torch.no_grad()
    def evaluate(self, model, loader, device):
        model.eval()
        metric = evaluate.load("glue", self.task_name)

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            inputs = {k: v for k, v in batch.items() if k != "labels"}

            out = model(**inputs)
            logits = out.logits

            if self.is_regression:
                preds = logits.squeeze(-1)
            else:
                preds = logits.argmax(dim=-1)

            metric.add_batch(
                predictions=preds.detach().cpu(),
                references=labels.detach().cpu()
            )

        raw = metric.compute()

        # aggregation rule (exactly like you wrote)
        if self.task_name == "cola":
            agg_metric, name = raw["matthews_correlation"], "MCC"
        elif self.task_name == "stsb":
            agg_metric, name = (raw["pearson"] + raw["spearmanr"]) / 2.0, "Pearson+Spearman"
        elif self.task_name in ["mrpc", "qqp"]:
            agg_metric, name = (raw["accuracy"] + raw["f1"]) / 2.0, "Acc+F1"
        else:
            agg_metric, name = raw["accuracy"], "Accuracy"

        out = {"agg": float(agg_metric), "agg_name": name}
        out.update({k: float(v) for k, v in raw.items()})
        return out
