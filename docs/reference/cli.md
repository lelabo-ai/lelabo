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

## `lelabo push`

Publish one capsule to a workspace repo or a configured remote target.

```bash
lelabo push [capsule_ref] [--target NAME] [--all-targets] [-m MESSAGE] [--preview] [--yes] [--json]
```

- `capsule_ref` can be a local path or a stored capsule id/alias
- if the capsule already lives in a git repo with `origin`, LeLabo uses the implicit `workspace` target
- otherwise LeLabo can bootstrap a GitHub target and remember it for next time

| Flag | Description |
|---|---|
| `--target NAME` | Push to one named target |
| `--all-targets` | Push to every configured target |
| `-m MESSAGE` | Commit message override |
| `--preview` | Show the resolved publish plan without pushing |
| `--yes` | Accept bootstrap defaults non-interactively |

`lelabo push` auto-stages and auto-commits only the selected capsule scope. It never does a repo-wide `git add -A`.

---

## `lelabo targets`

Manage GitHub publish targets for capsules. Most users only need `lelabo push` — use `lelabo targets` when you need multiple publish destinations or to manage which capsules are attached to a shared repo.

```bash
lelabo targets <subcommand> [args]
```

### `list`

```bash
lelabo targets list [CAPSULE_REF] [--json]
```

List configured GitHub targets. With a `CAPSULE_REF`, shows targets for that specific capsule.

### `create`

```bash
lelabo targets create [--target OWNER/REPO] [--public | --private] [--json]
```

Create or register a shared GitHub target repo.

### `attach`

```bash
lelabo targets attach REPO_REF CAPSULE_REF [--json]
```

Attach a capsule to an existing configured target. `REPO_REF` is `owner/repo`.

### `detach`

```bash
lelabo targets detach REPO_REF CAPSULE_REF [--json]
```

Detach a capsule from a target (local config only — does not delete remote files).

### `defaults`

```bash
lelabo targets defaults list CAPSULE_REF [--json]
lelabo targets defaults add CAPSULE_REF REPO_REF [--json]
lelabo targets defaults remove CAPSULE_REF REPO_REF [--json]
```

Manage which targets are marked as default for a given capsule.

### `import`

```bash
lelabo targets import REPO_REF [--capsule CAPSULE_ID] [--json]
```

Import capsule(s) discovered in a target repo into the local store.

### `remove-capsule`

```bash
lelabo targets remove-capsule REPO_REF CAPSULE_REF [--json]
```

Remove a capsule from a target repo and detach it locally.

### `delete`

```bash
lelabo targets delete REPO_REF [--json]
```

Delete a target entry from the local config and clean its attachments.

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
- GitHub LeLabo repo URL (`https://github.com/<owner>/<repo>` or `.git`)

Local capsule directories must use `lelabo capsule attach <capsule_dir>`.
GitHub repos must be LeLabo-compatible repos (created or managed via `lelabo targets` or `lelabo push`).

Install is store-first. Use `--checkout` to move the installed capsule into a workspace after install.

| Flag | Description |
|---|---|
| `--alias NAME` | Optional alias in the store |
| `--rename-to CAPSULE_ID` | Install under a different capsule id to avoid id conflicts |
| `--force-replace` | Replace an existing capsule that already uses the same capsule id |
| `--capsule ID` | Select one capsule from a multi-capsule LeLabo repo (repeatable) |
| `--all` | Install all capsules declared in the repo manifest |
| `--ref REF` | Branch / tag / commit for GitHub installs |
| `--checkout [DEST]` | Checkout after install (`DEST` optional, defaults to config checkout dir) |

Install conflict policy:
- same capsule id + same fingerprint in current workspace: no-op (`install_action = "already_present_workspace"`)
- same capsule id + same source: no-op (`install_action = "unchanged"`)
- same capsule id + different source: error with explicit resolution (`--force-replace` or `--rename-to`)
- multi-capsule repo without `--capsule` or `--all`: interactive checkbox picker in TTY (`↑/↓`, `space`, `a`, `enter`), error otherwise

Human mode prints short progress updates such as cloning, resolving, and installing. `--json` stays silent except for the final payload.

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
