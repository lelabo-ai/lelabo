# Installation

## Requirements

- Python 3.11 or 3.12
- pip

## Install

!!! tip "Use a virtual environment"
    We recommend installing LeLabo in a dedicated virtual environment to avoid dependency conflicts:
    ```bash
    python -m venv .venv
    source .venv/bin/activate   # on Windows: .venv\Scripts\activate
    ```

```bash
pip install lelabo
```

That's it. All dependencies — PyTorch, scikit-learn, Hugging Face, Gymnasium — are included.

## Install from source

If you want the latest development version:

```bash
git clone https://github.com/lelabo-ai/lelabo.git
cd lelabo
pip install -e "."
```

The `-e` flag installs in editable mode, so local changes are picked up immediately.

## Install dev tools

To also install the documentation and test tools:

```bash
pip install -e ".[dev]"
```

## Verify

```bash
lelabo --help
```

You should see the list of available commands: `train`, `list`, `capsule`, `audit`.

```bash
lelabo list models
lelabo list update-rules
lelabo list datasets
```

If these return a list of built-in components, your installation is working.

!!! tip "GPU support"
    LeLabo uses PyTorch under the hood. If you want GPU support, install the CUDA-enabled version of PyTorch after installing LeLabo. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for instructions.
