## Initial Pool JSON

Use this folder to store hand-crafted seed rules for GA warm start.

Enable from config:

```yaml
ga:
  initial_pool:
    enabled: true
    path: genetic_algorithm/pools/iris_seed_pool.json
    max_items: 0
    strict: false
```

Accepted JSON formats:

1. A list of genomes:

```json
[
  {"local_lr": 0.001, "update_expr": {"t": "term", "name": "u"}},
  {"local_lr": 0.0005, "update_expr": {"t": "binary", "op": "outer", "a": {"t": "term", "name": "u"}, "b": {"t": "term", "name": "x"}}}
]
```

2. An object with `pool` or `rules`:

```json
{"pool": [ ... ]}
```

Each entry is merged over a randomly sampled genome to fill missing fields.
