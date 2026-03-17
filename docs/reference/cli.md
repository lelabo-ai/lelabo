# CLI Reference

```
lelabo <command> [args]
```

---

## `lelabo train supervised`

Run a supervised experiment.

```bash
lelabo train supervised [--config PATH] [OPTIONS]
```

### Config and overrides

| Flag | Type | Description |
|---|---|---|
| `--config PATH` | str | Path to a TOML config file. Auto-detected if omitted. |
| `--set KEY=VALUE` | repeatable | Nested override, e.g. `--set model.params.hidden=512` |

### Component selection

| Flag | Description |
|---|---|
| `--dataset NAME` | Override `dataset.name` |
| `--model NAME` | Override `model.name` |
| `--rule NAME` | Override `update_rule.name` |
| `--optimizer NAME` | Override `optimizer.name` |
| `--loss NAME` | Override `loss.name` |
| `--initializer NAME` | Override `initializer.name` |

### Hyperparameters

| Flag | Type | Description |
|---|---|---|
| `--lr FLOAT` | float | Override `optimizer.params.lr` |
| `--weight-decay FLOAT` | float | Override `optimizer.params.weight_decay` |
| `--epochs INT` | int | Override `train.epochs` |
| `--batch INT` | int | Override `train.batch` |
| `--metrics NAMES` | str | Comma-separated metric names, e.g. `acc,f1` |

### Runtime

| Flag | Type | Default | Description |
|---|---|---|---|
| `--device STR` | str | `auto` | `cpu`, `cuda`, `mps`, or `auto` |
| `--seed INT` | int | — | Override `runtime.seed` |
| `--determinism STR` | str | `relaxed` | `off`, `relaxed`, or `strict` |
| `--display STR` | str | `compact` | `none`, `compact`, or `rich` |
| `--run-dir PATH` | str | — | Write run artifacts to this directory |
| `--save-checkpoints` | flag | false | Write `checkpoints/last.pt` and `checkpoints/best.pt` |

### Config auto-detection

If `--config` is omitted, LeLabo looks for a config file in this order:

```
./train.supervised.toml
./train.toml
./configs/train/supervised.toml
./configs/train.toml
```

---

## `lelabo train rl`

Run a reinforcement learning experiment.

```bash
lelabo train rl [--config PATH] [OPTIONS]
```

| Flag | Description |
|---|---|
| `--env ENV_ID` | Gymnasium environment ID, e.g. `CartPole-v1` |
| `--algo NAME` | RL algorithm name |
| `--rule NAME` | Override `update_rule.name` |
| `--optimizer NAME` | Override `optimizer.name` |
| `--lr FLOAT` | Override `optimizer.params.lr` |
| `--rl-steps INT` | Total training steps |
| `--rl-eval-episodes INT` | Evaluation episodes |
| `--rl-param KEY=VALUE` | Repeatable RL-specific param override |

Runtime flags (`--device`, `--seed`, `--run-dir`, etc.) are the same as for `supervised`.

!!! warning "RL is not yet the primary stable surface"
    See the [RL page](../rl.md) for current status.

---

## `lelabo list`

List registered components.

```bash
lelabo list [CATEGORY]
```

| Category | What it lists |
|---|---|
| `models` | Registered model builders |
| `update-rules` | Registered update rule builders |
| `datasets` | Registered dataset builders |
| `optimizers` | Registered optimizer builders |
| `losses` | Registered loss builders |
| `metrics` | Registered metric builders |
| `schedulers` | Registered scheduler builders |
| `initializers` | Registered initializer builders |

With no category, lists all categories.

---

## `lelabo capsule`

Manage capsules.

```bash
lelabo capsule <subcommand> [args]
```

### `init`

```bash
lelabo capsule init <name> [--dir PATH] [--force]
```

Create a new capsule scaffold.

| Flag | Default | Description |
|---|---|---|
| `name` | required | Capsule folder name |
| `--dir PATH` | `.` | Parent directory |
| `--force` | false | Create even if directory exists |

### `stash`

```bash
lelabo capsule stash [SOURCE] [--alias NAME] [--all] [--json]
```

Move a local capsule into the local capsule store.

| Flag | Description |
|---|---|
| `SOURCE` | Capsule root to stash (defaults to active capsule in cwd) |
| `--alias NAME` | Optional alias in the store |
| `--all` | Stash all child capsule folders in SOURCE |

### `checkout`

```bash
lelabo capsule checkout <id_or_alias> [DESTINATION] [--json]
```

Restore a stored capsule into a local workspace.

### `pack`

```bash
lelabo capsule pack --from SOURCE [--out PATH] [--id ID] [--with-code-snapshot]
```

Build a shareable capsule bundle.

| Flag | Description |
|---|---|
| `--from SOURCE` | Source run dir, sweep dir, or config file (required) |
| `--out PATH` | Output bundle path (`.tar.gz` or `.tar.zst`) |
| `--id ID` | Optional capsule ID |
| `--with-code-snapshot` | Embed `src/` snapshot in the bundle |

### `install`

```bash
lelabo capsule install <bundle> [--alias NAME] [--json]
```

Import a capsule bundle into the local store.

### `list`

```bash
lelabo capsule list [--json]
```

List stored capsules.

### `show`

```bash
lelabo capsule show <id_or_alias> [--json]
```

Show one stored capsule entry.

### `remove`

```bash
lelabo capsule remove <id_or_alias> [--keep-files] [-rf] [--json]
```

Remove a capsule from the store.

| Flag | Description |
|---|---|
| `--keep-files` | Remove from registry only, keep files on disk |
| `-rf` | Allow deleting files outside the capsule cache |

---

## `lelabo sweep`

Run parameter sweeps.

```bash
lelabo sweep <subcommand> [args]
```

### `run`

Execute a grid sweep from a YAML config file.

```bash
lelabo sweep run --config PATH [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config PATH` | str | required | YAML config describing base args + grid |
| `--outdir PATH` | str | `outputs/runs` | Parent directory for sweep outputs |
| `--name NAME` | str | from config | Override experiment name (also used as W&B group) |
| `--max-parallel N` | int | `1` | Number of concurrent jobs |
| `--gpus IDS` | str | — | GPU IDs for round-robin assignment, e.g. `0,1,2` |
| `--dry-run` | flag | — | Print commands without executing them |

Each job runs as an independent `lelabo train` subprocess with its own run directory. See the [sweep guide](../guides/sweeps.md) for config format and usage.

**Examples:**

```bash
# Preview commands
lelabo sweep run --config sweep.yaml --dry-run

# Run 4 jobs in parallel across 2 GPUs
lelabo sweep run --config sweep.yaml --max-parallel 4 --gpus 0,1

# Custom output directory and name
lelabo sweep run --config sweep.yaml --outdir results/ --name my_experiment
```

---

## `lelabo audit`

Run update rule audit tests.

```bash
lelabo audit [--all] [--modes supervised,rl] [OPTIONS]
```

Used to verify that registered update rules behave correctly across modes and datasets.
