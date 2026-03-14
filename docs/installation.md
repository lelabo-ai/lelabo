# Installation

## Requirements

- Python `>= 3.11`
- `pip`

## Extras matrix

- minimal: `pip install -e .`
- supervised: `pip install -e ".[supervised]"`
- supervised + NLP: `pip install -e ".[supervised,nlp]"`
- RL: `pip install -e ".[rl]"`
- local development: `pip install -e ".[supervised,rl,nlp,dev,test]"`

## Recommended installs

If you want the official supervised paths:

```bash
pip install -e ".[supervised]"
```

If you also want GLUE / BERT:

```bash
pip install -e ".[supervised,nlp]"
```

If you are working on the repo itself:

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
