from __future__ import annotations

import hashlib
from typing import Any, Iterable

from .schema import Candidate, DecisionExample


DECIDER_REPO = "Mapika/decider"
DECIDER_COMMIT = "b08acf787d5d1f718a8c36c4677960f43772c7be"
DECIDER_MODEL = "Mapika/decider-2b"

# All are explicitly registered held-out/evaluation-only tasks in the pinned upstream.
TRANSFER_TASKS = (
    "massive_scenario",
    "bbc_news",
    "trec",
    "tweet_irony",
    "fin_sentiment",
    "cr_reviews",
    "ade",
    "cb",
    "truthfulqa",
    "hwu64",
    "trec_fine",
    "dbpedia_l2",
    "dbpedia_l3",
    "quality_full",
    "xstory_cloze",
    "hermes_tools",
    "reward_bench",
)

# In-task eval datasets used only as a regression guard during candidate selection.
REGRESSION_TASKS = ("banking77", "ag_news", "boolq", "helpsteer3_pref")

# A small, fixed replay population for fine-tuning. These are training partitions upstream.
REPLAY_TASKS = ("banking77", "ag_news", "boolq", "helpsteer3_pref")


def _digest(*parts: Any, seed: str) -> str:
    raw = "\x00".join(str(x) for x in parts)
    return hashlib.sha256(f"{seed}\x00{raw}".encode("utf-8")).hexdigest()


def _flatten_upstream_examples(
    task_name: str,
    upstream_examples: Iterable[Any],
    split: str,
    limit: int,
    seed: str,
) -> list[DecisionExample]:
    """Convert upstream decider Example/Q objects without depending on their class definitions."""
    rows = list(upstream_examples)
    # Outcome-blind order: context + question text + candidate strings, never the gold index.
    keyed: list[tuple[str, Any, Any, int]] = []
    for ex_idx, ex in enumerate(rows):
        for q_idx, q in enumerate(ex.qs):
            key = _digest(ex.context, q.text, *q.options, seed=seed)
            keyed.append((key, ex, q, q_idx))
    keyed.sort(key=lambda x: x[0])
    if limit and len(keyed) < limit:
        raise ValueError(f"task {task_name} provides only {len(keyed)} questions, requested {limit}")
    selected = keyed[:limit] if limit else keyed

    out: list[DecisionExample] = []
    for rank, (_, ex, q, q_idx) in enumerate(selected):
        n = len(q.options)
        if not 2 <= n <= 255:
            continue
        gold = int(q.gold)
        if not 0 <= gold < n:
            continue
        # Display-order permutation also depends only on input semantics, not the gold.
        perm = sorted(range(n), key=lambda i: _digest(task_name, ex.context, q.text, q.options[i], seed=seed + ":perm"))
        inv_gold = perm.index(gold)
        candidates = [Candidate(id=f"c{i:03d}", text=str(q.options[src])) for i, src in enumerate(perm)]
        target = [float(i == inv_gold) for i in range(n)]
        row_id = _digest(task_name, ex.context, q.text, q_idx, seed=seed + ":id")[:24]
        out.append(
            DecisionExample(
                id=f"decider:{task_name}:{row_id}",
                family_id=f"decider:{task_name}:{rank:04d}",
                split=split,
                domain=task_name,
                state=str(ex.context),
                question=str(q.text),
                candidates=candidates,
                target_probs=target,
                target_kind="deterministic_truth",
                metadata={
                    "source": DECIDER_REPO,
                    "source_commit": DECIDER_COMMIT,
                    "upstream_task": task_name,
                    "upstream_question_index": q_idx,
                    "candidate_order_randomized": True,
                },
            )
        )
    return out


def build_transfer_rows(load_task, calibration_per_task: int = 10, test_per_task: int = 40) -> list[DecisionExample]:
    out: list[DecisionExample] = []
    total = calibration_per_task + test_per_task
    for task_name in TRANSFER_TASKS:
        _, evaluation = load_task(task_name)
        rows = _flatten_upstream_examples(task_name, evaluation, "ood", total, seed="jev48-transfer-v1")
        # First calibration_per_task positions are selected by the same outcome-blind hash order.
        for idx, row in enumerate(rows):
            row.split = "calibration" if idx < calibration_per_task else "ood"
            row.metadata["benchmark_role"] = "transfer_calibration" if idx < calibration_per_task else "transfer_locked"
        out.extend(rows)
    return out


def build_regression_rows(load_task, per_task: int = 75) -> list[DecisionExample]:
    out: list[DecisionExample] = []
    for task_name in REGRESSION_TASKS:
        _, evaluation = load_task(task_name)
        out.extend(_flatten_upstream_examples(task_name, evaluation, "dev", per_task, seed="jev48-regression-v1"))
    return out


def build_replay_rows(load_task, per_task: int = 500) -> list[DecisionExample]:
    out: list[DecisionExample] = []
    for task_name in REPLAY_TASKS:
        train, _ = load_task(task_name)
        out.extend(_flatten_upstream_examples(task_name, train, "train", per_task, seed="jev48-replay-v1"))
    return out
