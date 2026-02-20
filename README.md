# 🧪 Le Labo

> **Personal research laboratory for modular experimentation with learning algorithms in deep learning.**

---

## 🔍 Overview

**Le Labo** is a personal research workspace dedicated to experimenting with learning algorithms in deep learning.

The repository is designed as a **flexible and modular laboratory**, where models, tasks, and learning rules can be combined freely in order to study and compare different training paradigms under controlled conditions.

This is not intended to be a production-ready framework, but rather a **research-oriented environment** optimized for exploration, iteration, and understanding.

---

## 🎯 Goals

The main objectives of this project are to:

- Experiment with **optimization algorithms** (SGD variants, adaptive methods, etc.)
- Explore **alternative learning rules**, including local and biologically inspired approaches
- Compare different **learning paradigms** on shared tasks
- Enable rapid prototyping without unnecessary boilerplate
- Serve as a long-term experimental playground for deep learning research

The focus is on **learning dynamics**, not on benchmarks or performance alone.

---

## 🧠 Design Philosophy

This repository follows a few guiding principles:

- **Modularity first**  
  Algorithms, models, tasks, and training logic are cleanly decoupled.

- **Minimal abstractions**  
  Only abstractions that help experimentation are introduced.

- **Explicit over implicit**  
  Research code should be readable, inspectable, and easy to reason about.

- **Algorithm-centric view**  
  Learning rules are treated as first-class objects, not implementation details.

- **Flexibility over stability**  
  APIs may evolve as research questions change.

---

## 🧩 High-level Structure

The repository is now split by responsibility:

- **`src/lab/`**  
  Importable research code: models, datasets, algorithms, trainer/runner logic.

- **`experiments/configs/`**  
  YAML experiment definitions (grids, baselines, demos).

- **`experiments/launchers/`**  
  Orchestration scripts to run many jobs from config files.

- **`tools/`**  
  Utility scripts for post-processing and plotting.

- **`outputs/`**  
  Generated artifacts (`outputs/runs/`, `outputs/figures/`).

- **`scripts/` and `plots/`**  
  Backward-compatible wrappers pointing to the new locations.

The structure is designed to make it easy to answer questions like:

> *What happens if I change only the learning rule while keeping the model and task fixed?*

---

## 🚀 Usage

This repository is intended for **personal research use**.

Typical workflow:
1. Define or modify a **model / algorithm** in `src/lab/`
2. Create or update a sweep config in `experiments/configs/`
3. Launch runs with `experiments/launchers/launch_grid.py`
4. Analyze results with scripts in `tools/`

Capsule workflow (share/install/rerun experiments):
1. `lelabo capsule pack --from <run_dir> --out outputs/exports/capsules/my_run.tar.gz`
2. `lelabo capsule install outputs/exports/capsules/my_run.tar.gz --name my_baseline`
3. `lelabo capsule list`
4. `lelabo capsule show my_baseline`
5. `lelabo capsule rerun my_baseline --env current`

By default, installed capsules are stored under `.lelabo/capsules/` and can be overridden with
`--capsules-dir` or `LELABO_CAPSULES_DIR`.

Quick examples:

```bash
pip install -e .
```

```bash
lelabo --help
```

```bash
python experiments/launchers/launch_grid.py \
  --config experiments/configs/demo.yaml \
  --max-parallel 4
```

```bash
python tools/plot_sweep_table.py \
  --config experiments/configs/demo.yaml \
  --metric eval.test.acc
```

---

## 🚧 Status

This project is under **active development**.

- Code structure may change
- APIs are not guaranteed to be stable
- Refactoring is expected and intentional

This flexibility is a feature, not a limitation.

---

## 📖 Citation

If you use ideas, code, or experiments from this repository, please cite:

**Adrien Kegreisz**, *Le Labo – Personal Research Laboratory for Learning Algorithms in Deep Learning*.

```bibtex
@misc{kegreisz_le_labo,
  author       = {Adrien Kegreisz},
  title        = {Le Labo: Personal Research Laboratory for Learning Algorithms in Deep Learning},
  year         = {2026},
  note         = {Private research repository}
}

```
## 📜 License

This project is licensed under the **MIT License**.

If this repository is made public in the future, **attribution is required** for any use or derivative work.

---

## 📝 Notes

**Le Labo** serves as a long-term research workspace.

Clarity, experimental flexibility, and conceptual soundness are prioritized over backward compatibility or polish.
