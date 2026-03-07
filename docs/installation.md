# Installation

## Requirements

- Python `>= 3.11`
- `pip`

## Install from source

Minimal install:

```bash
pip install -e .
```

Supervised workflows:

```bash
pip install -e ".[supervised]"
```

Supervised + docs + tests:

```bash
pip install -e ".[supervised,dev,test]"
```

RL workflows:

```bash
pip install -e ".[rl]"
```

NLP / Hugging Face workflows:

```bash
pip install -e ".[nlp]"
```

Everything typically needed for local development:

```bash
pip install -e ".[supervised,rl,nlp,dev,test]"
```

## Verify the install

```bash
lelabo --help
```

```bash
lelabo list
```

## Serve the documentation locally

```bash
mkdocs serve
```

Then open `http://127.0.0.1:8000`.

## Notes

- `pip install -e .` is the most practical setup while the project is evolving quickly.
- Some commands require optional dependencies. For example, CNN training needs the `supervised` extra, and BERT / GLUE experiments need the `nlp` extra.
