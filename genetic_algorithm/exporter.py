from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import RuntimeConfig
from .evolved_rule import EvolvedRuleParams, render_generated_rule_source
from .ga import GenerationLog, ScoredIndividual


def create_run_dir(*, base_output_dir: str, run_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base_output_dir) / f"{run_name}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def export_results(
    *,
    run_dir: Path,
    runtime_cfg: RuntimeConfig,
    winners: list[ScoredIndividual],
    history: list[GenerationLog],
    total_evaluations: int,
) -> list[Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    exported_files: list[Path] = []

    rules_dir = run_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "config_name": runtime_cfg.name,
        "seed": runtime_cfg.seed,
        "device": runtime_cfg.device,
        "objective": runtime_cfg.fitness.objective,
        "fitness_split": runtime_cfg.fitness.split,
        "eval_repeats": runtime_cfg.ga.eval_repeats,
        "eval_seed_stride": runtime_cfg.ga.eval_seed_stride,
        "total_evaluations": int(total_evaluations),
        "history": [h.__dict__ for h in history],
        "winners": [],
    }

    for rank, individual in enumerate(winners, start=1):
        fname = f"ga_rule_top{rank}.py"
        rule_path = rules_dir / fname
        rule_name = _rule_name(runtime_cfg.name, rank)
        params = EvolvedRuleParams.from_genome(individual.result.evolved_params)
        source = render_generated_rule_source(
            rule_name=rule_name,
            params=params,
            score=individual.result.score,
            objective_value=individual.result.objective_value,
        )
        rule_path.write_text(source, encoding="utf-8")
        exported_files.append(rule_path)

        manifest["winners"].append(
            {
                "rank": rank,
                "rule_name": rule_name,
                "file": str(rule_path),
                "score": individual.result.score,
                "objective_value": individual.result.objective_value,
                "metrics": individual.result.metrics,
                "params": individual.result.merged_params,
                "evolved_params": individual.result.evolved_params,
                "replicate_seeds": individual.result.replicate_seeds,
                "replicate_objectives": individual.result.replicate_objectives,
                "score_std": individual.result.score_std,
                "generation": individual.generation,
                "ok": individual.result.ok,
                "error": individual.result.error,
            }
        )

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    (run_dir / "README.txt").write_text(
        "\n".join(
            [
                "Generated GA update-rules (pluggable):",
                "",
                "1) Copy the desired file(s) from `rules/` into:",
                "   src/lab/algorithms/update_rules/",
                "2) Keep the generated `@register_update_rule(...)` name.",
                "3) Launch training with --algo <generated_name>.",
                "",
                "These generated rules implement an evolved local update on hidden layers",
                "with supervised backprop on the final head layer.",
            ]
        ),
        encoding="utf-8",
    )

    return exported_files


def _rule_name(run_name: str, rank: int) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in run_name.lower()).strip("_")
    safe = safe or "ga"
    return f"ga_{safe}_top{rank}"
