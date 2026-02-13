from __future__ import annotations

import copy
import os
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from .config import GASettings
from .evaluator import EvaluationResult, SupervisedEvaluator
from .prune import simplify_expr_tree
from .search_space import SearchSpace

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


@dataclass
class ScoredIndividual:
    genome: dict[str, Any]
    result: EvaluationResult
    generation: int


@dataclass
class GenerationLog:
    generation: int
    evaluated: int
    best_score: float
    best_objective: float | None
    best_ok: bool


def _evaluate_genome_payload(payload: dict[str, Any]) -> tuple[str, EvaluationResult]:
    evaluator = SupervisedEvaluator(
        fixed_params=payload["fixed_params"],
        objective=payload["objective"],
        fitness_split=payload["fitness_split"],
        device=payload["device"],
        eval_repeats=payload["eval_repeats"],
        eval_seed_stride=payload["eval_seed_stride"],
    )
    fp = str(payload["fp"])
    genome = payload["genome"]
    result = evaluator.evaluate(genome)
    return fp, result


class GeneticSearch:
    def __init__(
        self,
        *,
        search_space: SearchSpace,
        evaluator: SupervisedEvaluator,
        settings: GASettings,
        seed: int,
        initial_pool: list[dict[str, Any]] | None = None,
    ):
        self.search_space = search_space
        self.evaluator = evaluator
        self.settings = settings
        self.rng = random.Random(int(seed))
        self._cache: dict[str, EvaluationResult] = {}
        self.evaluation_count = 0

        self.workers = self._resolve_workers(settings.workers)
        self.show_progress = bool(settings.show_progress)
        self.parallel_backend = "none"
        self.memetic_enabled = bool(settings.memetic_enabled)
        self._parent_prune_cache: dict[str, tuple[dict[str, Any], EvaluationResult]] = {}
        self.initial_pool: list[dict[str, Any]] = copy.deepcopy(initial_pool or [])

    def run(self) -> tuple[list[ScoredIndividual], list[GenerationLog]]:
        population = self._build_initial_population()
        history: list[GenerationLog] = []
        archive: list[ScoredIndividual] = []

        executor: ProcessPoolExecutor | ThreadPoolExecutor | None = None
        if self.workers > 1:
            try:
                executor = ProcessPoolExecutor(max_workers=self.workers)
                self.parallel_backend = "process"
            except Exception:
                executor = ThreadPoolExecutor(max_workers=self.workers)
                self.parallel_backend = "thread"
        try:
            gen_iter: Any = range(self.settings.generations)
            if self.show_progress and tqdm is not None:
                gen_iter = tqdm(gen_iter, desc="GA generations", unit="gen", leave=True)

            for generation in gen_iter:
                scored, new_eval_count = self._evaluate_population(population, generation, executor=executor)
                archive.extend(scored)

                if not scored:
                    break

                scored.sort(key=lambda item: item.result.score, reverse=True)
                best = scored[0]
                history.append(
                    GenerationLog(
                        generation=generation,
                        evaluated=int(new_eval_count),
                        best_score=float(best.result.score),
                        best_objective=best.result.objective_value,
                        best_ok=best.result.ok,
                    )
                )

                if self.show_progress and tqdm is not None and hasattr(gen_iter, "set_postfix"):
                    gen_iter.set_postfix(
                        evals=self.evaluation_count,
                        best=f"{best.result.score:.4f}",
                    )

                if self._budget_reached():
                    break
                if generation == self.settings.generations - 1:
                    break

                population = self._next_population(scored)
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

        archive.sort(key=lambda item: item.result.score, reverse=True)
        return archive, history

    def _build_initial_population(self) -> list[dict[str, Any]]:
        target = int(self.settings.population_size)
        population: list[dict[str, Any]] = []
        seen: set[str] = set()

        if bool(self.settings.initial_pool_enabled) and self.initial_pool:
            max_items = int(self.settings.initial_pool_max_items)
            for raw in self.initial_pool:
                if len(population) >= target:
                    break
                if max_items > 0 and len(population) >= max_items:
                    break
                genome = self._materialize_pool_genome(raw)
                fp = self.search_space.fingerprint(genome)
                if fp in seen:
                    continue
                seen.add(fp)
                population.append(genome)

        while len(population) < target:
            genome = self.search_space.sample(self.rng)
            fp = self.search_space.fingerprint(genome)
            if fp in seen:
                continue
            seen.add(fp)
            population.append(genome)
        return population

    def _materialize_pool_genome(self, raw: dict[str, Any]) -> dict[str, Any]:
        genome = self.search_space.sample(self.rng)
        merged = copy.deepcopy(genome)
        for key, value in raw.items():
            if key == "extra" and isinstance(value, dict):
                extra = dict(merged.get("extra", {}) or {})
                extra.update(copy.deepcopy(value))
                merged["extra"] = extra
                continue
            merged[key] = copy.deepcopy(value)
        return merged

    def _evaluate_population(
        self,
        population: list[dict[str, Any]],
        generation: int,
        *,
        executor: ProcessPoolExecutor | ThreadPoolExecutor | None,
    ) -> tuple[list[ScoredIndividual], int]:
        considered: list[tuple[str, dict[str, Any]]] = []
        pending: dict[str, dict[str, Any]] = {}

        for genome in population:
            fp = self.search_space.fingerprint(genome)

            if fp in self._cache:
                considered.append((fp, copy.deepcopy(genome)))
                continue

            if fp not in pending and self._budget_reached(extra_new=len(pending) + 1):
                break

            considered.append((fp, copy.deepcopy(genome)))
            if fp not in pending:
                pending[fp] = copy.deepcopy(genome)

        new_eval_count = 0
        if pending:
            results = self._evaluate_pending(
                pending,
                generation=generation,
                executor=executor,
            )
            for fp, result in results.items():
                self._cache[fp] = result
            self.evaluation_count += len(results)
            new_eval_count = int(len(results))

        scored: list[ScoredIndividual] = []
        for fp, genome in considered:
            result = self._cache.get(fp)
            if result is None:
                result = _failed_evaluation(genome, "Evaluation missing from cache.")
            scored.append(ScoredIndividual(genome=genome, result=result, generation=generation))
        return scored, new_eval_count

    def _evaluate_pending(
        self,
        pending: dict[str, dict[str, Any]],
        *,
        generation: int,
        executor: ProcessPoolExecutor | ThreadPoolExecutor | None,
    ) -> dict[str, EvaluationResult]:
        out: dict[str, EvaluationResult] = {}
        items = list(pending.items())

        progress = None
        if self.show_progress and tqdm is not None:
            progress = tqdm(
                total=len(items),
                desc=f"Gen {generation + 1} eval",
                unit="ind",
                leave=False,
            )

        try:
            if executor is None:
                for fp, genome in items:
                    result = self.evaluator.evaluate(genome)
                    out[fp] = result
                    if progress is not None:
                        progress.update(1)
                return out

            futures = {}
            for fp, genome in items:
                payload = {
                    "fp": fp,
                    "genome": genome,
                    "fixed_params": self.evaluator.fixed_params,
                    "objective": self.evaluator.objective,
                    "fitness_split": self.evaluator.fitness_split,
                    "device": self.evaluator.device,
                    "eval_repeats": self.evaluator.eval_repeats,
                    "eval_seed_stride": self.evaluator.eval_seed_stride,
                }
                fut = executor.submit(_evaluate_genome_payload, payload)
                futures[fut] = (fp, genome)

            for fut in as_completed(futures):
                fp, genome = futures[fut]
                try:
                    got_fp, result = fut.result()
                    out[got_fp] = result
                except Exception as exc:
                    out[fp] = _failed_evaluation(genome, f"Worker failed: {exc}")
                finally:
                    if progress is not None:
                        progress.update(1)

            return out
        finally:
            if progress is not None:
                progress.close()

    def _next_population(self, scored: list[ScoredIndividual]) -> list[dict[str, Any]]:
        if self.memetic_enabled:
            return self._next_population_memetic(scored)
        return self._next_population_classic(scored)

    def _next_population_classic(self, scored: list[ScoredIndividual]) -> list[dict[str, Any]]:
        scored = sorted(scored, key=lambda item: item.result.score, reverse=True)
        next_pop: list[dict[str, Any]] = []

        elites = scored[: self.settings.elite_size]
        next_pop.extend(copy.deepcopy(item.genome) for item in elites)

        while len(next_pop) < self.settings.population_size:
            p1 = self._select_tournament(scored).genome
            p2 = self._select_tournament(scored).genome
            child = self.search_space.crossover(
                p1,
                p2,
                crossover_rate=self.settings.crossover_rate,
                rng=self.rng,
            )
            child = self.search_space.mutate(
                child,
                mutation_rate=self.settings.mutation_rate,
                rng=self.rng,
            )
            next_pop.append(child)
        return next_pop

    def _next_population_memetic(self, scored: list[ScoredIndividual]) -> list[dict[str, Any]]:
        scored = sorted(scored, key=lambda item: item.result.score, reverse=True)
        next_pop: list[dict[str, Any]] = []
        pop_size = int(self.settings.population_size)
        score_tol = float(self.settings.memetic_score_tol)

        elites = scored[: self.settings.elite_size]
        for elite in elites:
            pruned_genome, _ = self._pruned_parent(elite, score_tol=score_tol)
            next_pop.append(copy.deepcopy(pruned_genome))

        immigrant_rate = float(self.settings.memetic_immigrant_rate)
        immigrant_slots = int(round(pop_size * immigrant_rate))
        immigrant_slots = max(0, min(pop_size - len(next_pop), immigrant_slots))
        child_slots = max(0, pop_size - len(next_pop) - immigrant_slots)

        while len(next_pop) < (len(elites) + child_slots):
            p1 = self._select_tournament(scored)
            p2 = self._select_tournament(scored)
            p1_genome, p1_result = self._pruned_parent(p1, score_tol=score_tol)
            p2_genome, p2_result = self._pruned_parent(p2, score_tol=score_tol)

            child = self.search_space.crossover(
                p1_genome,
                p2_genome,
                crossover_rate=self.settings.crossover_rate,
                rng=self.rng,
            )
            child = self.search_space.mutate(
                child,
                mutation_rate=self.settings.mutation_rate,
                rng=self.rng,
            )
            child = self._canonical_prune_genome(child)
            child_result = self._evaluate_cached_genome(child)

            if child_result is None:
                next_pop.append(copy.deepcopy(child))
                continue

            parent_ref = max(float(p1_result.score), float(p2_result.score))
            if float(child_result.score) + score_tol < parent_ref:
                # Reject weak child and inject full-random genome.
                next_pop.append(self.search_space.sample(self.rng))
            else:
                next_pop.append(copy.deepcopy(child))

        for _ in range(immigrant_slots):
            next_pop.append(self.search_space.sample(self.rng))

        while len(next_pop) < pop_size:
            next_pop.append(self.search_space.sample(self.rng))
        return next_pop[:pop_size]

    def _pruned_parent(self, item: ScoredIndividual, *, score_tol: float) -> tuple[dict[str, Any], EvaluationResult]:
        base_fp = self.search_space.fingerprint(item.genome)
        cached = self._parent_prune_cache.get(base_fp)
        if cached is not None:
            g, r = cached
            return copy.deepcopy(g), r

        base_genome = copy.deepcopy(item.genome)
        base_result = item.result
        pruned_genome = self._canonical_prune_genome(base_genome)
        pruned_fp = self.search_space.fingerprint(pruned_genome)

        if pruned_fp == base_fp:
            chosen = (base_genome, base_result)
        else:
            pruned_result = self._evaluate_cached_genome(pruned_genome)
            if pruned_result is None:
                chosen = (base_genome, base_result)
            elif pruned_result.ok and (float(pruned_result.score) + score_tol >= float(base_result.score)):
                chosen = (pruned_genome, pruned_result)
            else:
                chosen = (base_genome, base_result)

        self._parent_prune_cache[base_fp] = (copy.deepcopy(chosen[0]), chosen[1])
        return copy.deepcopy(chosen[0]), chosen[1]

    def _canonical_prune_genome(self, genome: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(genome)
        expr = out.get("update_expr")
        if isinstance(expr, dict):
            out["update_expr"] = simplify_expr_tree(expr)
        if not bool(out.get("update_bias", True)):
            out["bias_expr"] = None
        return out

    def _evaluate_cached_genome(self, genome: dict[str, Any]) -> EvaluationResult | None:
        fp = self.search_space.fingerprint(genome)
        cached = self._cache.get(fp)
        if cached is not None:
            return cached
        if self._budget_reached(extra_new=1):
            return None
        result = self.evaluator.evaluate(genome)
        self._cache[fp] = result
        self.evaluation_count += 1
        return result

    def _select_tournament(self, scored: list[ScoredIndividual]) -> ScoredIndividual:
        k = max(1, min(self.settings.tournament_size, len(scored)))
        candidates = [self.rng.choice(scored) for _ in range(k)]
        candidates.sort(key=lambda item: item.result.score, reverse=True)
        return candidates[0]

    def _budget_reached(self, *, extra_new: int = 0) -> bool:
        if self.settings.max_evals is None:
            return False
        max_evals = int(self.settings.max_evals)
        if int(extra_new) <= 0:
            return self.evaluation_count >= max_evals
        return (self.evaluation_count + int(extra_new)) > max_evals

    @staticmethod
    def _resolve_workers(requested: int) -> int:
        if requested == 0:
            return max(1, int(os.cpu_count() or 1))
        return max(1, int(requested))


def _failed_evaluation(genome: dict[str, Any], error: str) -> EvaluationResult:
    return EvaluationResult(
        ok=False,
        score=float("-inf"),
        objective_value=None,
        fitness_split="val",
        metrics={},
        train_summary=None,
        merged_params=copy.deepcopy(genome),
        evolved_params=copy.deepcopy(genome),
        error=error,
        traceback=None,
    )
