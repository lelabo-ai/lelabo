# Installation

## Requirements

- Python `>= 3.11`
- `pip`

## Extras matrix

- minimal: `pip install lelabo`
- supervised: `pip install "lelabo[supervised]"`
- supervised + NLP: `pip install "lelabo[supervised,nlp]"`
- RL: `pip install "lelabo[rl]"`
- all public extras: `pip install "lelabo[supervised,rl,nlp]"`

## Recommended installs

If you want the official supervised paths:

```bash
pip install "lelabo[supervised]"
```

If you also want GLUE / BERT:

```bash
pip install "lelabo[supervised,nlp]"
```

If you want all public optional surfaces:

```bash
pip install "lelabo[supervised,rl,nlp]"
```

## Verify the install

```bash
lelabo --help
```

```bash
lelabo list
```

```bash
lelabo train supervised --config configs/train/supervised.quickstart.toml
```

## Optional notes

- CNN and dataset-heavy supervised workflows need the `supervised` extra.
- GLUE / BERT workflows need the `nlp` extra.
- RL entrypoints need the `rl` extra.

## Serve the docs locally

```bash
mkdocs serve
```
