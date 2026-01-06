# core/robustness.py
import numpy as np
import torch

@torch.no_grad()
def _accuracy_on_tensors(model, x, y, device="cpu", batch_size=2048):
    model_was_training = model.training
    model.eval()

    x = x.to(device)
    y = y.to(device)

    correct = 0
    total = 0

    for start in range(0, x.size(0), batch_size):
        xb = x[start:start+batch_size]
        yb = y[start:start+batch_size]
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()

    if model_was_training:
        model.train()

    return correct / max(1, total)


@torch.no_grad()
def test_with_noise(
    model,
    x,
    y,
    sigma: float,
    mode: str = "input_noise",
    device: str = "cpu",
    trials: int = 5,
    batch_size: int = 2048,
):
    """
    Test accuracy under different noise models.

    Returns:
      mean_acc, ci95_half_width

    Modes:
      - "input_noise":          x -> x + sigma * N(0, I)
      - "relative_input_noise": x -> x + sigma * |x| * N(0, I)
      - "weight_noise":         p -> p + sigma * N(0, I) for all parameters
    """
    assert mode in ["input_noise", "relative_input_noise", "weight_noise"]

    accs = []

    # Keep a copy of the original parameters for restore (only used for weight_noise)
    params = list(model.parameters())
    saved = None

    for _ in range(trials):
        if mode == "input_noise":
            x_pert = x + sigma * torch.randn_like(x)
            acc = _accuracy_on_tensors(model, x_pert, y, device=device, batch_size=batch_size)

        elif mode == "relative_input_noise":
            x_pert = x + sigma * torch.abs(x) * torch.randn_like(x)
            acc = _accuracy_on_tensors(model, x_pert, y, device=device, batch_size=batch_size)

        else:  # weight_noise
            if saved is None:
                saved = [p.data.clone() for p in params]

            # perturb in-place
            for p in params:
                p.data.add_(sigma * torch.randn_like(p))

            acc = _accuracy_on_tensors(model, x, y, device=device, batch_size=batch_size)

            # restore
            for p, orig in zip(params, saved):
                p.data.copy_(orig)

        accs.append(acc)

    accs = np.array(accs, dtype=np.float64)
    mean_acc = float(accs.mean())
    std_acc = float(accs.std(ddof=1)) if trials > 1 else 0.0
    ci95 = 1.96 * std_acc / np.sqrt(trials) if trials > 1 else 0.0
    return mean_acc, float(ci95)
