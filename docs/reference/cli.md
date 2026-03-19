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

## `lelabo config`

Manage user-level settings for GitHub and capsule defaults.

```bash
lelabo config <subcommand> [args]
```

| Subcommand | Description |
|---|---|
| `path` | Show global config path and discovered local override |
| `show` | Show effective merged settings (global + local) |
| `get <key>` | Read one dotted key |
| `set <key> <value>` | Write one dotted key |
| `edit` | Open config file in `$EDITOR` |

Examples:

```bash
lelabo config path
lelabo config show
lelabo config set github.owner your-org
lelabo config set capsules.default_checkout_dir ./workbench
lelabo config get capsules.install_checkout
```

!!! warning "Security"
    Do not store GitHub tokens or secrets in `lelabo config`. Authentication is handled by `gh auth`.

---

## `lelabo gitspace`

Manage multi-capsule Git workspaces.

```bash
lelabo gitspace <subcommand> [args]
```

### `init`

```bash
lelabo gitspace init [PATH]
```

Create `.lelabo/gitspace.toml` in `PATH` or the current directory.

### `show`

```bash
lelabo gitspace show [PATH]
```

Show the detected gitspace and its manifest location.

### `list`

```bash
lelabo gitspace list [PATH]
```

List capsules declared in the gitspace manifest.

### `add`

```bash
lelabo gitspace add <capsule_path> [PATH]
```

Register an existing local capsule in the gitspace manifest without moving files.

Human output is rendered as compact status blocks; use `--json` when you need stable machine-readable output.

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

### `attach`

```bash
lelabo capsule attach [SOURCE] [--alias NAME] [--rename-to CAPSULE_ID] [--force-replace] [--json]
```

Link a local capsule into the LeLabo registry without moving or copying files.

| Flag | Description |
|---|---|
| `SOURCE` | Capsule root to attach (defaults to active capsule in cwd) |
| `--alias NAME` | Optional alias in the store registry |
| `--rename-to CAPSULE_ID` | Register under a different capsule id |
| `--force-replace` | Replace an existing registry entry with the same capsule id |

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
lelabo capsule install <source> [--alias NAME] [--rename-to CAPSULE_ID] [--force-replace] [--capsule ID ...] [--all] [--ref REF] [--checkout [DEST]] [--json]
```

Import a capsule into the local store.

`<source>` supports:
- local bundle path (`.tar.gz` / `.tar.zst`)
- GitHub gitspace URL (`https://github.com/<owner>/<repo>` or `.git`)

Local capsule directories must use `lelabo capsule attach <capsule_dir>`.
GitHub repos without `.lelabo/gitspace.toml` are not supported.

Install is store-first. Use `--checkout` to move the installed capsule into a workspace after install.

| Flag | Description |
|---|---|
| `--alias NAME` | Optional alias in the store |
| `--rename-to CAPSULE_ID` | Install under a different capsule id to avoid id conflicts |
| `--force-replace` | Replace an existing capsule that already uses the same capsule id |
| `--capsule ID` | Select one capsule from a multi-capsule gitspace (repeatable) |
| `--all` | Install all capsules declared in the gitspace |
| `--ref REF` | Branch / tag / commit for GitHub installs |
| `--checkout [DEST]` | Checkout after install (`DEST` optional, defaults to config checkout dir) |

Install conflict policy:
- same capsule id + same fingerprint in current workspace: no-op (`install_action = "already_present_workspace"`)
- same capsule id + same source: no-op (`install_action = "unchanged"`)
- same capsule id + different source: error with explicit resolution (`--force-replace` or `--rename-to`)
- multi-capsule gitspace without `--capsule` or `--all`: interactive checkbox picker in TTY (`↑/↓`, `space`, `a`, `enter`), error otherwise

Human mode prints short progress updates such as cloning, resolving, and installing. `--json` stays silent except for the final payload.

### `share`

```bash
lelabo capsule share [capsule_ref] [--mode github|local] [--owner OWNER] [--repo REPO] [--branch BRANCH] [--public|--private] [--out PATH] [--yes] [--json]
```

Share a capsule with mode-based backends.

- Default: `--mode github`
- Local export: `--mode local`
- `capsule_ref` can be a local path or a stored capsule id/alias (so you can share from a multi-capsule workspace)
- GitHub share works through the capsule's gitspace, not through a standalone capsule repo

Preconditions:
- run from an active capsule root
- `gh` is installed and authenticated (`gh auth status`)
- worktree is clean

Behavior:
- no auto-commit
- with `--mode local`, exports a `.tar.gz` bundle in the current directory by default (or `--out PATH`) and does not require git/gh
- with `--mode github`, if the capsule is not part of any gitspace, LeLabo launches a guided bootstrap flow
- with `--mode github`, LeLabo can initialize git + first commit after confirmation (`--yes` for non-interactive mode)
- optional repo auto-creation remains controlled by `github.create_repo_if_missing`
- with `--json`, bootstrap is non-interactive only (`--yes` required when bootstrap is needed)

The interactive bootstrap flow shows:
- the detected GitHub account
- the proposed gitspace root
- the gitspace/GitHub settings to review before publishing

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

Discover and run workspace sweeps.

```bash
lelabo sweep
lelabo sweep run [--config PATH | --capsule CAPSULE_ID --sweep SWEEP_NAME] [OPTIONS]
```

Behavior:
- `lelabo sweep` scans descendant capsules under the current workspace
- in a TTY, it opens an interactive capsule -> sweep picker
- outside a TTY, it prints a compact listing instead of showing help
- named sweeps are discovered only from `sweeps/*.yaml` and `sweeps/*.yml`

### `run`

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config PATH` | str | — | Direct path to a sweep YAML config |
| `--capsule CAPSULE_ID` | str | — | Capsule id resolved from the current workspace |
| `--sweep SWEEP_NAME` | str | — | Named sweep from `sweeps/<name>.yaml` |
| `--outdir PATH` | str | `outputs/runs` | Parent directory for sweep outputs |
| `--name NAME` | str | from config | Override experiment name (also used as W&B group) |
| `--max-parallel N` | int | `1` | Number of concurrent jobs |
| `--gpus IDS` | str | — | GPU IDs for round-robin assignment, e.g. `0,1,2` |
| `--dry-run` | flag | — | Print commands without executing them |

Each job runs as an independent `lelabo train` subprocess with its own run directory. See the [sweep guide](../guides/sweeps.md) for config format and usage.

**Examples:**

```bash
# Discover sweeps in the current workspace
lelabo sweep

# Run a named sweep from a discovered capsule
lelabo sweep run --capsule demo_capsule --sweep example

# Preview commands from a config path
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
