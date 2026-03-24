# Push a Capsule to GitHub

LeLabo can publish capsules to a private GitHub repository so you can share them, reinstall them on another machine, or make them available to collaborators.

## Prerequisites

- [`gh`](https://cli.github.com/) installed and authenticated (`gh auth status`)
- A GitHub account

## Quick start

### Step 1 — Set your GitHub username

```bash
lelabo config set github.owner your-username
```

This tells LeLabo where to create the default capsule repository. You only need to do this once.

### Step 2 — Push the capsule

From inside a capsule directory (or from any directory with a capsule id):

```bash
lelabo push
```

On the first push, LeLabo bootstraps automatically:

- It suggests `your-username/lelabo-capsules` as the destination (private repository)
- It creates the repo on GitHub if it doesn't exist yet (using `gh`)
- It pushes the capsule code and remembers the target for next time

Future pushes go directly to the same destination — no prompts.

### Step 3 — Preview before pushing

```bash
lelabo push --preview
```

Shows the resolved publish plan without actually pushing anything.

### Step 4 — Install from GitHub

Anyone with access to the repo (or you, on another machine) can install:

```bash
lelabo capsule install https://github.com/your-username/lelabo-capsules
```

If the repo contains multiple capsules, LeLabo shows an interactive picker. To install a specific one:

```bash
lelabo capsule install https://github.com/your-username/lelabo-capsules --capsule my_capsule
```

To install and immediately check out into a workspace:

```bash
lelabo capsule install https://github.com/your-username/lelabo-capsules \
  --capsule my_capsule \
  --checkout ./workbench
```

---

## Full example

```bash
# One-time setup
lelabo config set github.owner alice

# Create and develop a capsule
lelabo capsule init dfa_paper
cd dfa_paper
# ... implement your method ...

# Push to GitHub (bootstraps on first run)
lelabo push

# On another machine, install it
lelabo capsule install https://github.com/alice/lelabo-capsules \
  --capsule dfa_paper \
  --checkout ./workbench
```

---

## Configuration

`lelabo config` controls the push behavior:

| Key | Default | Description |
|---|---|---|
| `github.owner` | — | Your GitHub username or org. Used as the default push destination. |
| `github.default_visibility` | `"private"` | Visibility for auto-created repos. |
| `github.create_repo_if_missing` | `true` | Auto-create the GitHub repo if it doesn't exist. |
| `capsules.default_checkout_dir` | `"."` | Default directory for `--checkout` when installing. |
| `capsules.install_checkout` | `false` | Auto-checkout after every `lelabo capsule install`. |

View or edit your config:

```bash
lelabo config show
lelabo config set github.owner your-username
lelabo config get capsules.default_checkout_dir
```

---

## Targets (advanced)

When you run `lelabo push`, LeLabo creates a **target** — a named, persisted link between your capsule and a GitHub repository. Targets are stored locally in `~/.cache/lelabo/publish_targets.json`.

By default, one target is created and used automatically. You don't need to think about this for the standard workflow.

If you need to push the same capsule to multiple repos, or manage a shared repo with multiple capsules, use `lelabo targets`:

```bash
# List configured targets
lelabo targets list

# Attach a capsule to an additional repo
lelabo targets attach owner/another-repo my_capsule

# Manage which targets are used by default
lelabo targets defaults list my_capsule
lelabo targets defaults add my_capsule owner/another-repo
```

See the [CLI reference](../reference/cli.md#lelabo-targets) for the full `lelabo targets` command reference.
