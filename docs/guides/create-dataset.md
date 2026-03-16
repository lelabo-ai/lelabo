# Create a Dataset

Register a custom dataset loader so it can be selected by name from a config or the CLI.

## When to use this

Use a dataset extension when you need custom loading, splitting, preprocessing, or metadata that the built-in datasets don't provide.

## Where to edit

Inside your capsule: `datasets/example.py`

## Builder shape

```python
from lelabo.supervised.datasets.registry import register_dataset
from lelabo.supervised.datasets.bundle import DataBundle
from torch.utils.data import TensorDataset
import torch


@register_dataset("my_dataset")
def make_my_dataset(**kwargs):
    # kwargs contains all params from [dataset.params] in the config
    n_samples = kwargs.get("n_samples", 1000)
    n_features = kwargs.get("n_features", 20)
    n_classes = kwargs.get("n_classes", 3)

    # Build your data here
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))

    # Split into train/val/test
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)

    train_ds = TensorDataset(X[:n_train], y[:n_train])
    val_ds = TensorDataset(X[n_train:n_train+n_val], y[n_train:n_train+n_val])
    test_ds = TensorDataset(X[n_train+n_val:], y[n_train+n_val:])

    return DataBundle(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        num_classes=n_classes,
        in_dim=n_features,
        input_shape=(n_features,),
    )
```

## `DataBundle` fields

| Field | Type | Description |
|---|---|---|
| `train` | `Dataset` | Training split |
| `val` | `Dataset \| None` | Validation split |
| `test` | `Dataset \| None` | Test split |
| `num_classes` | `int` | Number of output classes |
| `in_dim` | `int` | Flattened input dimension |
| `input_shape` | `tuple` | Raw input shape |

!!! warning "Always set `num_classes` and `in_dim`"
    The model builder receives these values through `ModelContext`. If they are missing, model construction will fail.

## Config snippet

```toml
[dataset]
name = "my_dataset"

[dataset.params]
n_samples = 2000
n_features = 32
n_classes = 5
```

## Verify

```bash
lelabo list datasets
# my_dataset should appear

lelabo train supervised \
  --dataset my_dataset \
  --model mlp \
  --rule bp
```

## Common mistakes

- Returning something other than a `DataBundle`
- Not setting `num_classes` or `in_dim` — this breaks model construction
- Reading params from somewhere other than `**kwargs`
- Trying to access a context object — dataset builders receive plain kwargs

## Image dataset example

```python
from lelabo.supervised.datasets.registry import register_dataset
from lelabo.supervised.datasets.bundle import DataBundle
from torchvision import datasets, transforms


@register_dataset("my_image_dataset")
def make_my_image_dataset(**kwargs):
    root = kwargs.get("root", "./data")
    img_size = kwargs.get("img_size", 32)

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_ds = datasets.ImageFolder(f"{root}/train", transform=transform)
    val_ds = datasets.ImageFolder(f"{root}/val", transform=transform)

    num_classes = len(train_ds.classes)
    in_dim = 3 * img_size * img_size

    return DataBundle(
        train=train_ds,
        val=val_ds,
        num_classes=num_classes,
        in_dim=in_dim,
        input_shape=(3, img_size, img_size),
    )
```
