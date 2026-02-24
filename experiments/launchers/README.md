Batch/sweep launchers live here.

Primary entrypoint:
- `launch_grid.py`: builds the Cartesian product from YAML and executes runs.
  - Default run module: `lelabo.cli.main` (equivalent to `lelabo` command).

Default outputs:
- Run logs and metadata in `outputs/runs/`.
