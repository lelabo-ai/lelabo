import torch
import torch.nn.functional as F

class ClassificationTask:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def loss(self, logits, y):
        return F.cross_entropy(logits, y)

    @torch.no_grad()
    def metrics(self, logits, y):
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        return {"acc": acc.item()}
