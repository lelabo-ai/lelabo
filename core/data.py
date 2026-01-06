# core/data.py
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def make_iris_loaders(batch_size=32, seed=42):
    data = load_iris()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.long)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    yte = torch.tensor(yte, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, Xtr, ytr, Xte, yte, Xtr.shape[1], len(set(y))


def make_mnist_loaders(batch_size=128, seed=42, flatten=True):
    """
    Returns:
      train_loader, test_loader, Xtr, ytr, Xte, yte, in_dim_or_shape, num_classes
    """
    from torchvision.datasets import MNIST
    from torchvision import transforms

    g = torch.Generator().manual_seed(seed)

    # Standard MNIST normalization
    tfm = transforms.Compose([
        transforms.ToTensor(),  # [0,1], shape [1,28,28]
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = MNIST(root="./data", train=False, download=True, transform=tfm)

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

    Xtr, ytr = to_tensors(train_ds)
    Xte, yte = to_tensors(test_ds)

    if flatten:
        Xtr = Xtr.view(Xtr.size(0), -1)     # [N,784]
        Xte = Xte.view(Xte.size(0), -1)
        in_dim_or_shape = Xtr.shape[1]
    else:
        in_dim_or_shape = tuple(Xtr.shape[1:])  # (1,28,28) for CNNs later

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch_size,
        shuffle=True,
        generator=g
    )
    test_loader = DataLoader(
        TensorDataset(Xte, yte),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, Xtr, ytr, Xte, yte, in_dim_or_shape, 10

def make_cifar_loaders(dataset="cifar10", batch_size=128, seed=42, flatten=False, num_workers=2):
    """
    dataset: "cifar10" or "cifar100"
    flatten: True -> returns [N, 3072] (MLP)
             False -> returns [N, 3, 32, 32] (CNN)
    """
    from torchvision import datasets, transforms

    assert dataset in ["cifar10", "cifar100"]

    # CIFAR normalization (standard)
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
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        num_classes = 10
    else:
        train_ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        test_ds  = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        num_classes = 100

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=g,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    in_dim_or_shape = 3072 if flatten else (3, 32, 32)
    return train_loader, test_loader, in_dim_or_shape, num_classes

def make_breast_cancer_loaders(batch_size, seed=42, flatten=True):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X, y = data.data, data.target  # X: (N, 30), y: (N,) with 0/1

    # split first (avoid leakage)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.long)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    yte = torch.tensor(yte, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)

    in_dim = Xtr.shape[1]
    num_classes = int(y.max() + 1)  # 2

    return train_loader, test_loader, Xtr, ytr, Xte, yte, in_dim, num_classes
