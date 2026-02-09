import torch
import matplotlib.pyplot as plt
import os

# lab/core/data.py
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _split_train_val(n: int, val_frac: float, seed: int):
    """
    Returns train_idx, val_idx as torch.LongTensor.
    Deterministic split using a local generator.
    """
    val_frac = float(val_frac)
    if val_frac <= 0.0:
        return torch.arange(n), torch.empty(0, dtype=torch.long)

    val_size = int(round(n * val_frac))
    val_size = max(1, min(val_size, n - 1))  # ensure non-empty train/val

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    return train_idx, val_idx


def make_iris_loaders(batch_size=32, seed=42, val_frac: float = 0.0, input_noise_dataset: float = 0.0, noise_on_test: bool = False):
    data = load_iris()
    X, y = data.data, data.target
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split train+val vs test first
    Xtrv, Xte, ytrv, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    Xtrv = torch.tensor(Xtrv, dtype=torch.float32)
    ytrv = torch.tensor(ytrv, dtype=torch.long)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    yte = torch.tensor(yte, dtype=torch.long)

    # add input noise if specified
    if input_noise_dataset > 0.0:
        # Relative noise: scale per element by a percentage of its absolute value
        noise_trv = torch.randn_like(Xtrv) * input_noise_dataset
        Xtrv = Xtrv + noise_trv

        if noise_on_test:
            noise_te = torch.randn_like(Xte) * input_noise_dataset
            Xte = Xte + noise_te
    # split train vs val
    n = Xtrv.size(0)
    tr_idx, va_idx = _split_train_val(n, val_frac, seed)

    train_ds = TensorDataset(Xtrv[tr_idx], ytrv[tr_idx])
    val_ds = TensorDataset(Xtrv[va_idx], ytrv[va_idx]) if va_idx.numel() > 0 else None
    test_ds = TensorDataset(Xte, yte)

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    in_dim = Xtrv.shape[1]
    num_classes = int(ytrv.max().item() + 1)

    return train_loader, val_loader, test_loader, Xtrv, ytrv, Xte, yte, in_dim, num_classes


def make_mnist_loaders(batch_size=128, seed=42, flatten=True, val_frac: float = 0.0, input_noise_dataset: float = 0.0, noise_on_test: bool = False):
    """
    Returns:
      train_loader, val_loader, test_loader, Xtr, ytr, Xte, yte, in_dim_or_shape, num_classes
    """
    from torchvision.datasets import MNIST
    from torchvision import transforms

    # Standard MNIST normalization
    tfm = transforms.Compose([
        transforms.ToTensor(),  # [0,1], shape [1,28,28]
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds_full = MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = MNIST(root="./data", train=False, download=True, transform=tfm)

    # Materialize into tensors (simple + robust-eval friendly)
    def to_tensors(ds):
        xs, ys = [], []
        for i in range(len(ds)):
            x, y = ds[i]
            xs.append(x)
            ys.append(y)
        X = torch.stack(xs, dim=0)          # [N,1,28,28]
        y = torch.tensor(ys, dtype=torch.long)
        return X, y

    Xtr_full, ytr_full = to_tensors(train_ds_full)
    Xte, yte = to_tensors(test_ds)

    # add input noise if specified
    if input_noise_dataset > 0.0:
        # Relative noise: scale per element by a percentage of its absolute value
        scale_tr = input_noise_dataset * torch.abs(Xtr_full)
        noise_tr = torch.randn_like(Xtr_full) * scale_tr
        Xtr_full = Xtr_full + noise_tr

        if noise_on_test:
            scale_te = input_noise_dataset * torch.abs(Xte)
            noise_te = torch.randn_like(Xte) * scale_te
            Xte = Xte + noise_te
            
    if flatten:
        Xtr_full = Xtr_full.view(Xtr_full.size(0), -1)  # [N,784]
        Xte = Xte.view(Xte.size(0), -1)
        in_dim_or_shape = Xtr_full.shape[1]
    else:
        in_dim_or_shape = tuple(Xtr_full.shape[1:])  # (1,28,28)

    # split train vs val on the 60k train
    n = Xtr_full.size(0)
    tr_idx, va_idx = _split_train_val(n, val_frac, seed)

    Xtr = Xtr_full[tr_idx]
    ytr = ytr_full[tr_idx]
    Xva = Xtr_full[va_idx] if va_idx.numel() > 0 else None
    yva = ytr_full[va_idx] if va_idx.numel() > 0 else None

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch_size,
        shuffle=True,
        generator=g
    )

    val_loader = None
    if Xva is not None:
        val_loader = DataLoader(
            TensorDataset(Xva, yva),
            batch_size=batch_size,
            shuffle=False
        )

    test_loader = DataLoader(
        TensorDataset(Xte, yte),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, Xtr, ytr, Xte, yte, in_dim_or_shape, 10


def make_cifar_loaders(dataset="cifar10", batch_size=128, seed=42, flatten=False, num_workers=2, val_frac: float = 0.0, input_noise_dataset: float = 0.0, noise_on_test: bool = False):
    """
    dataset: "cifar10" or "cifar100"
    flatten: True -> returns [N, 3072] (MLP)
             False -> returns [N, 3, 32, 32] (CNN)

    Returns:
      train_loader, val_loader, test_loader, in_dim_or_shape, num_classes
    """
    from torchvision import datasets, transforms

    assert dataset in ["cifar10", "cifar100"]

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    tfms = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if flatten:
        tfms.append(transforms.Lambda(lambda t: t.view(-1)))

    transform = transforms.Compose(tfms)

    if dataset == "cifar10":
        train_full = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        num_classes = 10
    else:
        train_full = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        num_classes = 100

    # add input noise if specified
    if input_noise_dataset > 0.0:
        def add_noise_to_dataset(ds):
            for i in range(len(ds)):
                x, y = ds[i]
                scale = input_noise_dataset * torch.abs(x)
                noise = torch.randn_like(x) * scale
                x_noisy = x + noise
                ds.data[i] = x_noisy if flatten else x_noisy.view(3, 32, 32)
        add_noise_to_dataset(train_full)
        if noise_on_test:
            add_noise_to_dataset(test_ds)
                
    n = len(train_full)
    tr_idx, va_idx = _split_train_val(n, val_frac, seed)

    train_ds = Subset(train_full, tr_idx.tolist())
    val_ds = Subset(train_full, va_idx.tolist()) if va_idx.numel() > 0 else None

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=g,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    in_dim_or_shape = 3072 if flatten else (3, 32, 32)
    return train_loader, val_loader, test_loader, in_dim_or_shape, num_classes


def make_breast_cancer_loaders(batch_size, seed=42, flatten=True, val_frac: float = 0.0, input_noise_dataset: float = 0.0, noise_on_test: bool = False):
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X, y = data.data, data.target  # X: (N, 30), y: (N,) with 0/1

    # split train+val vs test first (avoid leakage)
    Xtrv, Xte, ytrv, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    # add input noise if specified
    if input_noise_dataset > 0.0:
        # Relative noise: scale per element by a percentage of its absolute value
        Xtrv_tensor = torch.tensor(Xtrv, dtype=torch.float32)
        scale_trv = input_noise_dataset * torch.abs(Xtrv_tensor)
        noise_trv = torch.randn_like(Xtrv_tensor) * scale_trv
        Xtrv_tensor = Xtrv_tensor + noise_trv
        Xtrv = Xtrv_tensor.numpy()

        if noise_on_test:
            Xte_tensor = torch.tensor(Xte, dtype=torch.float32)
            scale_te = input_noise_dataset * torch.abs(Xte_tensor)
            noise_te = torch.randn_like(Xte_tensor) * scale_te
            Xte_tensor = Xte_tensor + noise_te
            Xte = Xte_tensor.numpy()

    scaler = StandardScaler()
    Xtrv = scaler.fit_transform(Xtrv)
    Xte = scaler.transform(Xte)

    Xtrv = torch.tensor(Xtrv, dtype=torch.float32)
    ytrv = torch.tensor(ytrv, dtype=torch.long)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    yte = torch.tensor(yte, dtype=torch.long)

    # split train vs val
    n = Xtrv.size(0)
    tr_idx, va_idx = _split_train_val(n, val_frac, seed)

    train_ds = TensorDataset(Xtrv[tr_idx], ytrv[tr_idx])
    val_ds = TensorDataset(Xtrv[va_idx], ytrv[va_idx]) if va_idx.numel() > 0 else None
    test_ds = TensorDataset(Xte, yte)

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    in_dim = Xtrv.shape[1]
    num_classes = int(ytrv.max().item() + 1)

    return train_loader, val_loader, test_loader, Xtrv, ytrv, Xte, yte, in_dim, num_classes


# dossier de sortie
os.makedirs("noise_examples", exist_ok=True)

def get_one_five(input_noise):
    _, _, _, Xtr, ytr, _, _, in_dim, _ = make_mnist_loaders(
        batch_size=128,
        flatten=False,
        input_noise_dataset=input_noise,
        noise_on_test=False
    )

    # trouver le premier "5"
    idx = (ytr == 5).nonzero(as_tuple=True)[0][0]
    x = Xtr[idx]  # shape [1, 28, 28]
    return x.squeeze(0)  # [28,28]


# niveaux de bruit
noise_levels = {
    "clean": 0.0,
    "noise_5pct": 5,
    "noise_20pct": 20,
}

import random
import matplotlib.pyplot as plt
import torch
import os

os.makedirs("noise_examples", exist_ok=True)

# niveaux de bruit (attention : ton loader attend des proportions)
noise_levels = [0.0, 5, 20]
noise_names = ["Noise = 0%", "Noise = 5%", "Noise = 20%"]

fig, axes = plt.subplots(
    nrows=3,
    ncols=10,
    figsize=(18, 6)
)

for row, (noise, noise_name) in enumerate(zip(noise_levels, noise_names)):
    _, _, _, Xtr, ytr, _, _, _, _ = make_mnist_loaders(
        flatten=False,
        input_noise_dataset=noise,
        noise_on_test=False
    )

    # prendre UN exemple par chiffre (0–9)
    indices = []
    for digit in range(10):
        idx = (ytr == digit).nonzero(as_tuple=True)[0][0]
        indices.append(idx)

    # mélanger l'ordre des chiffres pour cette ligne
    random.shuffle(indices)

    for col, idx in enumerate(indices):
        img = Xtr[idx].squeeze(0)
        label = ytr[idx].item()

        ax = axes[row, col]
        ax.imshow(img, cmap="gray")
        ax.set_title(str(label), fontsize=10)
        ax.axis("off")

    # label de la ligne (à gauche)
    axes[row, 0].set_ylabel(noise_name, fontsize=12)

plt.tight_layout()

# sauvegarde UNE SEULE IMAGE
plt.savefig(
    "noise_examples/mnist_noise_grid.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()
