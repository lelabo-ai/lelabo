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
        layer_cache = []
        hooks = []

        def hook_fn(module, inputs, output):
            x = inputs[0]
            z = output
            if x.dim() == 2 and z.dim() == 2:
                layer_cache.append((x.detach(), z.detach()))

        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                hooks.append(m.register_forward_hook(hook_fn))

        # ---- forward ----
        if isinstance(batch, Mapping):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            y_true = batch["labels"]
        else:
            x, y_true = batch
            x, y_true = x.to(device), y_true.to(device)
            logits = model(x)

        for h in hooks:
            h.remove()

        # ---- reinforcement signal ----
        preds = logits.argmax(dim=-1)
        correct = (preds == y_true).float()
        r = torch.where(correct > 0, 1.0, -1.0).view(-1, 1)

        linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        B = logits.shape[0]
        num_classes = logits.shape[1]

        # ---- learning ----
        for l_idx, (layer, (x_l, z_l)) in enumerate(zip(linear_layers, layer_cache)):
            is_last = (l_idx == len(linear_layers) - 1)

            alpha = 0.7
            lr_l = self.lr * (alpha ** (len(linear_layers) - 1 - l_idx))

            # =========================
            # INITIALISATION MÉMOIRE
            # =========================
            if not hasattr(layer, "class_usage"):
                layer.class_usage = torch.zeros(
                    layer.weight.shape[0],
                    layer.weight.shape[1],
                    num_classes,
                    device=layer.weight.device
                )

            # =========================
            # DERNIÈRE COUCHE (inchangée)
            # =========================
            if is_last:
                t = torch.zeros((B, num_classes), device=z_l.device)
                t.scatter_(1, y_true.view(-1, 1), 1.0)

                s = r * t
                x_term = (2 * x_l - 1)
                dW = (s.T @ x_term) / B

            # =========================
            # COUCHES CACHÉES AVEC MÉMOIRE
            # =========================
            else:
                y_bin = (z_l > 0).float()
                post = 2.0 * y_bin - 1.0

                # mémoire de classe pour la classe vraie
                usage = layer.class_usage[..., y_true]          # (out, in, B)
                usage = usage.mean(dim=2)                        # (out, in)

                # facteur de protection (plus utilisé → moins puni)
                protection = torch.exp(-usage)

                s = r * post
                dW = (s.T @ x_l) / B
                dW = dW * protection

                # ---- mise à jour mémoire SI correct ----
                for b in range(B):
                    if correct[b] > 0:
                        layer.class_usage[:, :, y_true[b]] += (
                            y_bin[b].unsqueeze(1) *
                            x_l[b].unsqueeze(0)
                        )

            # ---- update poids ----
            layer.weight.add_(lr_l * dW)

            if layer.bias is not None:
                layer.bias.add_(lr_l * s.mean(dim=0))

        acc = correct.mean().item()
        self.global_step += 1
        return {"acc": acc, "loss": float(1.0 - acc)}
