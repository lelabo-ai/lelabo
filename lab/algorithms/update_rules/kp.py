# algorithms/kp1.py
import torch
from .base import UpdateRule
from collections.abc import Mapping

class KP(UpdateRule):
    def __init__(self, learning_rate=0.05):
        super().__init__()
        self.lr = learning_rate

    @torch.no_grad()
    def train_step(self, model, task, batch, device):
        model.train()

        # ---- capture (x, z) for each Linear layer ----
        layer_cache = []  # list of (x, z) per Linear
        hooks = []

        def hook_fn(module, inputs, output):
            x = inputs[0]          # (B, in)
            z = output             # (B, out)  pre-activation
            if x.dim() == 2 and z.dim() == 2:  # keep it simple (MLP case)
                layer_cache.append((x.detach(), z.detach()))

        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                hooks.append(m.register_forward_hook(hook_fn))

        # ---- forward (supports tuple or dict batches) ----
        if isinstance(batch, Mapping):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            y_true = batch["labels"]
        else:
            x, y_true = batch
            x, y_true = x.to(device), y_true.to(device)
            logits = model(x)

        # remove hooks ASAP
        for h in hooks:
            h.remove()

        # ---- reinforcement r: +1 correct, -1 wrong ----
        preds = logits.argmax(dim=-1)
        correct = (preds == y_true).float()
        r = torch.where(correct > 0, 1.0, -1.0).view(-1, 1)  # (B,1)

        # ---- apply your local rule on each Linear ----
        # find the Linear modules again in the same order as hooks fired
        linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        B = logits.shape[0]

        for l_idx, (layer, (x_l, z_l)) in enumerate(zip(linear_layers, layer_cache)):
            is_last = (l_idx == len(linear_layers) - 1)

            alpha = 0.7
            lr_l = self.lr * (alpha ** (len(linear_layers) - 1 - l_idx))

            if is_last:
                # ---- classification rule for last layer ----
                # y_true -> target in {-1, +1}
                num_classes = layer.weight.shape[0]
                t = torch.full((B, num_classes), 0, device=z_l.device)
                t.scatter_(1, y_true.view(-1, 1), 1.0)  # +1 for true class
                s = r * t                          # (B, C)
                x_term = (2 * x_l - 1)             # (B, in)
                dW = (s.T @ x_term) / B                # (C, in)

            else:
                # ---- hidden layer rule (unchanged) ----
                y_bin = (z_l > 0).float()
                post = 2.0 * y_bin - 0.00               # {-1, +1}
                s = r * post                           # (B, out)
                dW = (s.T @ x_l) / B                   # (out, in)

            layer.weight.add_(lr_l * dW)

            if layer.bias is not None:
                db = s.mean(dim=0)
                layer.bias.add_(lr_l * db)


        acc = correct.mean().item()
        self.global_step += 1
        return {"acc": acc, "loss": float(1.0 - acc)}
