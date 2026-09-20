from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from .schema import Candidate, DecisionExample


DECIDER_REPO = "Mapika/decider"
DECIDER_COMMIT = "b08acf787d5d1f718a8c36c4677960f43772c7be"
DECIDER_MODEL = "Mapika/decider-2b"
DECIDER_MODEL_REVISION = "4a0e86782adfdb7393e04b8ec9f6b939dca09273"


def resolve_model_snapshot(model: str = DECIDER_MODEL, revision: str = DECIDER_MODEL_REVISION) -> str:
    """Resolve the public base to an immutable local Hugging Face snapshot."""
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo_id=model, revision=revision)).resolve()
    if path.name != revision:
        raise RuntimeError(f"resolved model snapshot mismatch: {path.name} != {revision}")
    return str(path)

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
    allow_conflicting_duplicates: bool = False,
) -> list[DecisionExample]:
    """Convert upstream decider Example/Q objects without depending on their class definitions."""
    rows = list(upstream_examples)
    # Collapse exact repeated source questions before applying the fixed quota.
    # BBC News contains repeated articles with identical labels; counting them
    # twice would overweight those inputs and also produce duplicate row IDs.
    semantic_groups: dict[tuple[str, str, tuple[str, ...], int], list[tuple[Any, Any]]] = {}
    for ex_idx, ex in enumerate(rows):
        for q_idx, q in enumerate(ex.qs):
            signature = (str(ex.context), str(q.text), tuple(str(x) for x in q.options), q_idx)
            semantic_groups.setdefault(signature, []).append((ex, q))

    # Outcome-blind order: context + question text + candidate strings, never the gold index.
    keyed: list[tuple[str, Any, Any, int, int, int | None]] = []
    for signature, occurrences in semantic_groups.items():
        labels = {int(q.gold) for _, q in occurrences}
        if len(labels) != 1:
            if not allow_conflicting_duplicates:
                raise ValueError(f"task {task_name} has conflicting labels for a duplicate question")
            # Replay is training-only. Preserve contradictory source signals as
            # separate rows, but give them outcome-blind occurrence keys/IDs.
            for occurrence_index, (ex, q) in enumerate(occurrences):
                q_idx = signature[3]
                key = _digest(ex.context, q.text, *q.options, occurrence_index, seed=seed)
                keyed.append((key, ex, q, q_idx, 1, occurrence_index))
            continue
        ex, q = occurrences[0]
        q_idx = signature[3]
        key = _digest(ex.context, q.text, *q.options, seed=seed)
        keyed.append((key, ex, q, q_idx, len(occurrences), None))
    keyed.sort(key=lambda x: x[0])
    if limit and len(keyed) < limit:
        raise ValueError(f"task {task_name} provides only {len(keyed)} questions, requested {limit}")
    selected = keyed[:limit] if limit else keyed

    out: list[DecisionExample] = []
    for rank, (_, ex, q, q_idx, duplicate_count, occurrence_index) in enumerate(selected):
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
        row_id = _digest(task_name, ex.context, q.text, q_idx, occurrence_index, seed=seed + ":id")[:24]
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
                    "upstream_duplicate_count": duplicate_count,
                    "upstream_conflicting_duplicate": occurrence_index is not None,
                    "upstream_source_occurrence_index": occurrence_index,
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
        out.extend(
            _flatten_upstream_examples(
                task_name,
                train,
                "train",
                per_task,
                seed="jev48-replay-v1",
                allow_conflicting_duplicates=True,
            )
        )
    return out
