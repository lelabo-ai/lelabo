from __future__ import annotations

import argparse
import warnings


RL_ONLY_FLAGS = {
    "rl_steps",
    "rl_eval_episodes",
    "rl_algo",
    "rl_param",
}


SUPERVISED_ONLY_FLAGS = {
    "metrics",
    "bp_alignment_every",
    "bp_alignment_eps",
    "input_noise_training",
    "input_noise_dataset",
    "noise_on_test",
    "val_frac",
    "early_stop",
    "no_early_stop",
    "early_monitor",
    "early_patience",
    "early_min_delta",
    "early_warmup",
    "robustness",
    "noise_trials",
    "glue_task",
    "hf_model",
    "hf_trust_remote_code",
    "max_length",
}


def _render_flags(flags: set[str]) -> str:
    return ", ".join(f"--{name.replace('_', '-')}" for name in sorted(flags))


def validate_train_args(args: argparse.Namespace) -> None:
    task = str(getattr(args, "task", "")).strip().lower()
    if task not in {"supervised", "rl"}:
        raise ValueError(f"Invalid resolved task '{task}'. Expected 'supervised' or 'rl'.")

    provided = set(getattr(args, "_provided_flags", set()))

    if task == "supervised":
        invalid = provided & RL_ONLY_FLAGS
        if invalid:
            raise ValueError(
                "The following flags are not valid for task 'supervised': "
                f"{_render_flags(invalid)}."
            )
        return

    invalid = provided & SUPERVISED_ONLY_FLAGS
    if invalid:
        raise ValueError(
            "The following flags are not valid for task 'rl': "
            f"{_render_flags(invalid)}."
        )

    if "model" in provided:
        warnings.warn(
            "Model selection for RL is not implemented yet; '--model' is currently ignored.",
            UserWarning,
            stacklevel=2,
        )
