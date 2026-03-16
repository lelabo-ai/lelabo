# Built-in Components

All components that ship with LeLabo and are available by name in configs and the CLI.

---

## Models

### `mlp`

Multi-layer perceptron. Designed for tabular and flattened input.

| Param | Type | Default | Description |
|---|---|---|---|
| `hidden` | int | `128` | Hidden layer size |
| `layers` | int | `2` | Number of hidden layers |
| `dropout` | float | `0.0` | Dropout probability |
| `activation` | str | `"relu"` | Activation: `"relu"`, `"tanh"`, `"gelu"` |

Compatible with all update rules, including local rules.

---

### `cnn`

Small convolutional network. Designed for vision input (MNIST, CIFAR-10).

| Param | Type | Default | Description |
|---|---|---|---|
| `channels` | list | `[32, 64]` | Out-channels per conv layer |
| `kernel` | int | `3` | Kernel size |
| `hidden` | int | `256` | FC hidden size |
| `dropout` | float | `0.0` | Dropout probability |

Compatible with all update rules, including local rules.

---

### `resnet18`

Standard ResNet-18 backbone with a custom classification head.

| Param | Type | Default | Description |
|---|---|---|---|
| `pretrained` | bool | `false` | Load ImageNet pretrained weights |
| `freeze_backbone` | bool | `false` | Freeze backbone, train head only |

Local-rule compatibility: partial (standard blocks only; custom residual paths are not declared).

---

### `bert`

BERT base model with a classification head. Designed for GLUE/NLP tasks.

| Param | Type | Default | Description |
|---|---|---|---|
| `pretrained` | str | `"bert-base-uncased"` | HuggingFace model identifier |
| `freeze_encoder` | bool | `false` | Freeze encoder, train head only |
| `dropout` | float | `0.1` | Head dropout |

Requires `transformers` to be installed.

---

### `deephebb`

Deep Hebbian network. Architecture designed for SoftHebb-style local learning rules.

| Param | Type | Default | Description |
|---|---|---|---|
| `hidden` | int | `256` | Hidden layer size |
| `layers` | int | `3` | Number of layers |

Best used with the `softhebb` update rule.

---

## Update rules

### `bp`

Standard backpropagation. Uses the configured optimizer and loss.

No rule-specific params.

---

### `dfa` — Direct Feedback Alignment

Trains with direct random feedback connections from the output layer to each hidden layer, bypassing the chain rule.

| Param | Type | Default | Description |
|---|---|---|---|
| `feedback_scale` | float | `1.0` | Scale for feedback projection |

Requires the model to implement `declare_blocks()`.

---

### `fa` — Feedback Alignment

Replaces the transpose weight matrix in backprop with a fixed random matrix.

No rule-specific params. Requires `declare_blocks()`.

---

### `drtp` — Deep Random Target Projections

Projects target labels as local targets for each layer.

| Param | Type | Default | Description |
|---|---|---|---|
| `projection_scale` | float | `1.0` | Target projection scale |

Requires `declare_blocks()`.

---

### `dni` — Decoupled Neural Interfaces

Uses synthetic gradient modules to decouple layer updates.

| Param | Type | Default | Description |
|---|---|---|---|
| `dni_hidden` | int | `256` | Hidden size of the synthetic gradient network |
| `dni_lr` | float | `0.001` | Learning rate for synthetic gradient modules |

---

### `scl` — Symmetric Cross-Layer

Computes symmetric updates across layer pairs.

No rule-specific params. Requires `declare_blocks()`.

---

### `softhebb` — Soft Hebbian Learning

Competitive Hebbian learning with soft winner-take-all dynamics.

| Param | Type | Default | Description |
|---|---|---|---|
| `t` | float | `0.1` | Temperature for soft competition |
| `lr_hebb` | float | `0.01` | Hebbian learning rate |

Best used with the `deephebb` model.

---

## Datasets

### `iris`

Fisher's Iris dataset. 150 samples, 4 features, 3 classes.

No params. Loaded from scikit-learn.

| | |
|---|---|
| `in_dim` | 4 |
| `num_classes` | 3 |
| `input_shape` | `(4,)` |

---

### `mnist`

MNIST handwritten digits. 60k train / 10k test, 28×28 grayscale.

| Param | Type | Default | Description |
|---|---|---|---|
| `root` | str | `"./data"` | Download/cache directory |
| `flatten` | bool | `true` | Flatten to 784-dim vector |

| | |
|---|---|
| `in_dim` | 784 (flattened) or 1×28×28 |
| `num_classes` | 10 |

---

### `cifar10`

CIFAR-10. 50k train / 10k test, 32×32 RGB.

| Param | Type | Default | Description |
|---|---|---|---|
| `root` | str | `"./data"` | Download/cache directory |
| `augment` | bool | `false` | Random crop + horizontal flip on train |

| | |
|---|---|
| `in_dim` | 3072 (flattened) or 3×32×32 |
| `num_classes` | 10 |

---

### `glue/{task}`

GLUE benchmark tasks via HuggingFace Datasets. Supported tasks: `sst2`, `mrpc`, `qqp`, `mnli`, `qnli`, `rte`, `wnli`.

| Param | Type | Default | Description |
|---|---|---|---|
| `task` | str | `"sst2"` | GLUE task name |
| `model_name` | str | `"bert-base-uncased"` | Tokenizer model name |
| `max_length` | int | `128` | Max token sequence length |

Requires `transformers` and `datasets`.

---

## Optimizers

### `sgd`

| Param | Type | Default |
|---|---|---|
| `lr` | float | `0.01` |
| `momentum` | float | `0.0` |
| `weight_decay` | float | `0.0` |
| `nesterov` | bool | `false` |

### `adam`

| Param | Type | Default |
|---|---|---|
| `lr` | float | `0.001` |
| `betas` | list | `[0.9, 0.999]` |
| `eps` | float | `1e-8` |
| `weight_decay` | float | `0.0` |

### `adamw`

Same params as `adam`. Uses decoupled weight decay.

---

## Losses

| Name | Use case |
|---|---|
| `ce` | Multi-class classification (cross-entropy) |
| `bce` | Binary classification |
| `mse` | Regression |

---

## Metrics

| Name | Task | Description |
|---|---|---|
| `acc` | Classification | Top-1 accuracy |
| `f1` | Classification | Macro F1 score |
| `precision` | Classification | Macro precision |
| `recall` | Classification | Macro recall |
| `mse` | Regression | Mean squared error |
| `mae` | Regression | Mean absolute error |
| `r2` | Regression | R² coefficient |

---

## Schedulers

| Name | Description |
|---|---|
| `none` | No scheduling (default) |
| `step` | Multiply LR by `gamma` every `step_size` epochs |
| `exponential` | Multiply LR by `gamma` every epoch |
| `cosine` | Cosine annealing from initial LR to `eta_min` |
| `reduce_on_plateau` | Reduce LR when monitored metric stops improving |

---

## Callbacks

### `earlystopping`

| Param | Type | Default | Description |
|---|---|---|---|
| `monitor` | str | `"val.acc"` | Scalar to monitor |
| `mode` | str | `"auto"` | `"max"`, `"min"`, or `"auto"` |
| `patience` | int | `5` | Epochs without improvement before stopping |
| `min_delta` | float | `0.0` | Minimum change to count as improvement |
| `warmup` | int | `3` | Epochs before monitoring starts |
| `restore_best` | bool | `true` | Restore best model at end of training |

See [Config schema](config-schema.md#callbacks) for the full config syntax.
