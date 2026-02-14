from __future__ import annotations

import copy
import os
import random
from collections import Counter
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
        return self._next_population_classic(scored)

    def _next_population_classic(self, scored: list[ScoredIndividual]) -> list[dict[str, Any]]:
        scored = sorted(scored, key=lambda item: item.result.score, reverse=True)
        pop_size = int(self.settings.population_size)
        if pop_size <= 0:
            return []
        base = scored[:pop_size]
        if not base:
            return []

        next_pop = [self._canonicalize_genome(item.genome) for item in base]
        next_results: list[EvaluationResult] = [item.result for item in base]
        elite_size = max(0, min(int(self.settings.elite_size), len(next_pop)))
        occupied: set[str] = {self.search_space.fingerprint(g) for g in next_pop}

        immigrant_rate = float(self.settings.memetic_immigrant_rate)
        immigrant_slots = int(round(len(next_pop) * immigrant_rate))
        immigrant_slots = max(0, min(len(next_pop) - elite_size, immigrant_slots))
        self._inject_random_immigrants(
            population=next_pop,
            results=next_results,
            occupied=occupied,
            elite_size=elite_size,
            immigrant_slots=immigrant_slots,
        )
        child_slots = max(0, len(next_pop) - elite_size - immigrant_slots)

        for _ in range(child_slots):
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
            child = self._canonicalize_genome(child)
            child_result = self._evaluate_cached_genome(child)
            if child_result is None:
                break
            if not child_result.ok:
                continue

            replace_idx = self._pick_rtr_replacement_index(
                child=child,
                population=next_pop,
                elite_size=elite_size,
            )
            if replace_idx is None:
                break

            incumbent_result = next_results[replace_idx]
            if float(child_result.score) < float(incumbent_result.score):
                continue

            child_fp = self.search_space.fingerprint(child)
            incumbent_fp = self.search_space.fingerprint(next_pop[replace_idx])
            if child_fp != incumbent_fp and child_fp in occupied:
                continue

            occupied.discard(incumbent_fp)
            next_pop[replace_idx] = copy.deepcopy(child)
            next_results[replace_idx] = child_result
            occupied.add(child_fp)

        return [copy.deepcopy(g) for g in next_pop[:pop_size]]

    def _inject_random_immigrants(
        self,
        *,
        population: list[dict[str, Any]],
        results: list[EvaluationResult],
        occupied: set[str],
        elite_size: int,
        immigrant_slots: int,
    ) -> None:
        if immigrant_slots <= 0:
            return
        if elite_size >= len(population):
            return

        candidate_indices = list(range(elite_size, len(population)))
        self.rng.shuffle(candidate_indices)
        for idx in candidate_indices[:immigrant_slots]:
            immigrant = self._canonicalize_genome(self.search_space.sample(self.rng))
            imm_fp = self.search_space.fingerprint(immigrant)
            for _ in range(8):
                if imm_fp not in occupied:
                    break
                immigrant = self._canonicalize_genome(self.search_space.sample(self.rng))
                imm_fp = self.search_space.fingerprint(immigrant)

            immigrant_result = self._evaluate_cached_genome(immigrant)
            if immigrant_result is None:
                immigrant_result = _failed_evaluation(
                    immigrant,
                    "Budget reached before evaluating random immigrant.",
                )

            old_fp = self.search_space.fingerprint(population[idx])
            occupied.discard(old_fp)
            population[idx] = copy.deepcopy(immigrant)
            results[idx] = immigrant_result
            occupied.add(imm_fp)

    def _pick_rtr_replacement_index(
        self,
        *,
        child: dict[str, Any],
        population: list[dict[str, Any]],
        elite_size: int,
    ) -> int | None:
        if elite_size >= len(population):
            return None
        indices = list(range(elite_size, len(population)))
        if not indices:
            return None

        window = max(2, int(self.settings.tournament_size) * 2)
        window = min(window, len(indices))
        sampled = self.rng.sample(indices, k=window) if window < len(indices) else indices

        best_idx: int | None = None
        best_dist = float("inf")
        for idx in sampled:
            dist = self._genome_distance(child, population[idx])
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def _canonicalize_genome(self, genome: dict[str, Any]) -> dict[str, Any]:
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

    def _genome_distance(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        keys = sorted(set(left.keys()) | set(right.keys()))
        if not keys:
            return 0.0
        total = 0.0
        for key in keys:
            lv = left.get(key)
            rv = right.get(key)
            if key in {"update_expr", "bias_expr"}:
                total += self._expr_distance(lv, rv)
            else:
                total += self._value_distance(lv, rv)
        return float(total / len(keys))

    def _expr_distance(self, left: Any, right: Any) -> float:
        if not isinstance(left, dict) or not isinstance(right, dict):
            return self._value_distance(left, right)
        lc = self._expr_token_counts(left)
        rc = self._expr_token_counts(right)
        if not lc and not rc:
            return 0.0
        union = lc | rc
        inter = lc & rc
        denom = float(sum(union.values()))
        if denom <= 0.0:
            return 0.0
        return 1.0 - (float(sum(inter.values())) / denom)

    def _expr_token_counts(self, node: Any) -> Counter[str]:
        out: Counter[str] = Counter()
        if not isinstance(node, dict):
            out["none"] += 1
            return out

        t = str(node.get("t", "")).lower()
        if t == "term":
            out[f"term:{str(node.get('name', 'zeros')).lower()}"] += 1
            return out
        if t == "const":
            try:
                bucket = round(float(node.get("v", 0.0)), 2)
            except Exception:
                bucket = 0.0
            out[f"const:{bucket}"] += 1
            return out
        if t == "unary":
            out[f"u:{str(node.get('op', 'unknown')).lower()}"] += 1
            out += self._expr_token_counts(node.get("a"))
            return out
        if t == "binary":
            out[f"b:{str(node.get('op', 'unknown')).lower()}"] += 1
            out += self._expr_token_counts(node.get("a"))
            out += self._expr_token_counts(node.get("b"))
            return out

        out[f"unknown:{t}"] += 1
        return out

    @classmethod
    def _value_distance(cls, left: Any, right: Any) -> float:
        if left is None and right is None:
            return 0.0
        if isinstance(left, bool) and isinstance(right, bool):
            return 0.0 if left == right else 1.0
        if isinstance(left, (int, float)) and not isinstance(left, bool):
            if isinstance(right, (int, float)) and not isinstance(right, bool):
                denom = max(1.0, abs(float(left)), abs(float(right)))
                return min(1.0, abs(float(left) - float(right)) / denom)
        if isinstance(left, str) and isinstance(right, str):
            return 0.0 if left == right else 1.0
        if isinstance(left, dict) and isinstance(right, dict):
            keys = sorted(set(left.keys()) | set(right.keys()))
            if not keys:
                return 0.0
            total = 0.0
            for key in keys:
                total += cls._value_distance(left.get(key), right.get(key))
            return float(total / len(keys))
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            n = max(len(left), len(right))
            if n == 0:
                return 0.0
            total = 0.0
            for i in range(n):
                lv = left[i] if i < len(left) else None
                rv = right[i] if i < len(right) else None
                total += cls._value_distance(lv, rv)
            return float(total / n)
        return 0.0 if left == right else 1.0

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
