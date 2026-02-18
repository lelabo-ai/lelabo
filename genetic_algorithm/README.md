# Genetic Algorithm (Hors-Serie)

Module standalone a la racine, non relie au CLI `lelabo`.

## Objectif de ce mode

- Architecture verrouillee: `MLP` avec `5` couches cachees de `256` neurones.
- Training en 2 phases configurable:
  - phase 1: update locale creee (hidden layers)
  - phase 2: backprop supervise sur la derniere matrice (head uniquement)
- Couches cachees: regle locale evoluee par GA a partir des signaux:
  - `x`: entree de couche
  - `u`: pre-activation
  - `y`: sortie apres activation
  - `label`
  - `mistake` (pred != label)
  - `layer`, `layer_ratio` (position de la couche)
- La regle est un `expression_tree` (ops unaires/binaires), incluant
  `+`, `-`, `*`, `/`, `outer`, `dot`, `normalize`, etc.
- Inclut aussi `scale` pour multiplier n'importe quel sous-arbre par un coeff (ex: `0.2`).
- Profondeur de l'arbre bornable dans le YAML:
  - `search_space.update_expr.min_depth`
  - `search_space.update_expr.max_depth`

## Run

```bash
python -m genetic_algorithm.run --config genetic_algorithm/configs/iris_ga.yaml
```

Options:

```bash
python -m genetic_algorithm.run \
  --config genetic_algorithm/configs/iris_ga.yaml \
  --output-dir genetic_algorithm/outputs \
  --max-evals max \
  --workers 8

# disable tqdm
python -m genetic_algorithm.run --config genetic_algorithm/configs/iris_ga.yaml --no-progress
```

## Config YAML

- `fixed`: params fixes (dataset, epochs, batch, etc.)
- `fixed.local_phase_epochs` / `fixed.head_phase_epochs`: duree des 2 phases
- `search_space`: genes de la regle locale (expression tree + lrs)
- `fitness.objective`: `accuracy`, `loss`, `bp_cosine_epoch`, `bp_sign_match_epoch` ou `bp_update_gap_epoch`
- `bp_cosine_epoch`: moyenne simple des cosines batch-par-batch (updates GA hidden vs direction BP `-grad`) puis moyenne par epoch local
- `bp_sign_match_epoch`: moyenne simple, batch par batch, du pourcentage de composantes ou `sign(ΔW_GA) == sign(ΔW_BP)`
- `bp_update_gap_epoch`: moyenne simple du ratio `||ΔW_GA - ΔW_BP||_2 / (||ΔW_BP||_2 + eps)` (a minimiser; le GA maximise `-gap`)
- `fitness.split`: split de fitness (`train|val|test`, recommande `val`)
- `ga.top_k`: nombre de rules exportees
- `ga.max_evals`: budget global optionnel (`max`/`none` = pas de limite)
- `ga.workers`: parallelisme (0=auto)
- `ga.show_progress`: active tqdm
- `ga.eval_repeats`: nombre d'evals par individu (moyenne des scores)
- `ga.eval_seed_stride`: increment entre seeds de repetition
- `ga.memetic.enabled`: flag legacy (pruning per-generation desactive; conserve pour compatibilite)
- `ga.memetic.score_tol`: parametre legacy (non utilise pendant la selection RTR classique)
- `ga.memetic.immigrant_rate`: quota d'individus full-random injectes par generation (branche classique)
- `ga.initial_pool.enabled`: active un warm-start depuis un JSON de seeds
- `ga.initial_pool.path`: chemin du JSON de pool (liste de genomes)
- `ga.initial_pool.max_items`: limite d'entrees lues (0 = toutes)
- `ga.initial_pool.strict`: fail hard si fichier/format invalide

Exemple:

```yaml
ga:
  initial_pool:
    enabled: true
    path: genetic_algorithm/pools/iris_seed_pool.json
```

## Outputs

Chaque run cree:

- `genetic_algorithm/outputs/<run_name>_<timestamp>/manifest.json`
- `genetic_algorithm/outputs/<run_name>_<timestamp>/rules/ga_rule_top*.py`

Les `.py` exportes sont pluggables:

1. Copier le fichier dans `src/lab/algorithms/update_rules/`
2. Lancer un train avec `--algo <RULE_NAME>`

## Pruner / Degraisser les rules

Pour simplifier automatiquement les rules evoluees (ablation de params + simplification de `update_expr`) tout en preservant le score:

```bash
python -m genetic_algorithm.prune \
  --manifest genetic_algorithm/outputs/<run_name_timestamp>/manifest.json \
  --top-k 5 \
  --score-tol 0.005
```

Sorties:

- `prune_report.json`: avant/apres (score, complexite, etapes acceptees)
- `pruned_rules/*.py`: versions simplifiees pretes a etre pluggees

Options utiles:

- `--max-iters`: nombre max d'iterations de simplification par winner
- `--max-candidates-per-iter`: budget d'ablations teste a chaque iteration
- `--eval-repeats`, `--eval-seed-stride`: override de la robustesse d'eval
- `--no-export-rules`: ecrire seulement le report
