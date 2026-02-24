"""Minimal config usage hints for this capsule."""

# Prefer TOML files in this same folder (see train.*.toml templates + README).
# Include top-level version fields in your TOML files:
#   config_version = "auto"
#   lelabo_version = "auto"
#
# Example commands:
#   lelabo train supervised --config configs/train.supervised.quickstart.toml
#   lelabo train supervised --config configs/train.supervised.detailed.toml --dataset iris
#   lelabo train supervised --set model.params.hidden=1024
