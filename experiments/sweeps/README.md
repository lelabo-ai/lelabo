Experiment configuration files (`.yaml`) live here.

Conventions:
- `base`: fixed arguments passed to the training CLI (`lelabo`)
- `grid`: hyperparameters to sweep
- `name`: experiment folder name under `outputs/runs/`

Typical usage:
`python experiments/launchers/launch_grid.py --config experiments/sweeps/demo.yaml`
