from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Iterable, Any

from .schema import Candidate, DecisionExample


MTBENCH_REVISION = "ee34b9d273a7a35e4415c87678526c56c471098c"
SPLIT_SEED = "jev48-mtbench-v1"


def _stable_digest(value: str, seed: str = SPLIT_SEED) -> str:
    return hashlib.sha256(f"{seed}\x00{value}".encode("utf-8")).hexdigest()


def assign_question_splits(question_ids: Iterable[int | str]) -> dict[str, str]:
    """Outcome-blind 50/10/10/10 split over the 80 MT-Bench question IDs.

    The split depends only on question identity and a frozen seed, never votes.
    This keeps all model-pair judgments for the same underlying prompt family in
    exactly one split.
    """
    ids = sorted({str(x) for x in question_ids}, key=lambda x: _stable_digest(x))
    if len(ids) != 80:
        raise ValueError(f"expected exactly 80 MT-Bench question IDs, got {len(ids)}")
    boundaries = ((50, "train"), (60, "dev"), (70, "calibration"), (80, "test"))
    out: dict[str, str] = {}
    start = 0
    for end, split in boundaries:
        for qid in ids[start:end]:
            out[qid] = split
        start = end
    return out


def _conversation_text(conversation: Any) -> str:
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("conversation must be a nonempty list")
    lines: list[str] = []
    for turn in conversation:
        if not isinstance(turn, dict):
            raise ValueError("conversation turns must be objects")
        role = str(turn.get("role", "unknown")).strip().upper()
        content = turn.get("content")
        if not isinstance(content, str):
            raise ValueError("conversation turn content must be text")
        # The pinned dataset contains a genuine empty llama-13b answer for
        # question 127. Preserve that outcome explicitly instead of dropping
        # its human votes or allowing adjacent turns to collapse together.
        if not content.strip():
            lines.append(f"{role}: [EMPTY RESPONSE]")
            continue
        # Exactly one rendered line per structured chat message. Responses often
        # contain different numbers of internal newlines; preserving those as raw
        # line breaks would make A/B message alignment drift on multi-turn rows.
        one_line = " ⏎ ".join(part.strip() for part in content.strip().splitlines() if part.strip())
        lines.append(f"{role}: {one_line}")
    return "\n".join(lines)


def _paired_conversation(left_text: str, right_text: str) -> str:
    # _conversation_text is line based, so pair shared user/system context once and
    # keep alternative assistant turns side by side. This avoids duplicating long
    # prompts and keeps both alternatives inside the model context window.
    la = left_text.split("\n")
    lb = right_text.split("\n")
    out: list[str] = []
    for i in range(max(len(la), len(lb))):
        a = la[i] if i < len(la) else ""
        b = lb[i] if i < len(lb) else ""
        if a == b:
            out.append(a)
        else:
            # Roles are already included in each line. Use explicit anonymous sides.
            out.append("A> " + a)
            out.append("B> " + b)
    return "\n".join(x for x in out if x)


def _canonicalize_row(row: dict[str, Any]) -> tuple[tuple[str, int, str, str], str, str, str]:
    qid = str(row["question_id"])
    turn = int(row["turn"])
    a = str(row["model_a"])
    b = str(row["model_b"])
    winner = str(row["winner"])
    if winner not in {"model_a", "model_b", "tie"}:
        raise ValueError(f"unsupported MT-Bench winner {winner!r}")
    ca = _conversation_text(row["conversation_a"])
    cb = _conversation_text(row["conversation_b"])
    if a <= b:
        key = (qid, turn, a, b)
        return key, ca, cb, winner
    # Canonical pair order: swap both transcripts and the winner identity.
    key = (qid, turn, b, a)
    mapped = {"model_a": "model_b", "model_b": "model_a", "tie": "tie"}[winner]
    return key, cb, ca, mapped


def aggregate_human_votes(rows: Iterable[dict[str, Any]]) -> list[DecisionExample]:
    """Aggregate expert MT-Bench votes into empirical target distributions.

    Model names are retained only in metadata; the actual model sees anonymous
    Conversation A/B transcripts. Response order is deterministically shuffled
    per group to prevent position shortcuts.
    """
    rows = list(rows)
    split_map = assign_question_splits(row["question_id"] for row in rows)
    grouped: dict[tuple[str, int, str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for row in rows:
        key, ca, cb, winner = _canonicalize_row(row)
        grouped[key].append((ca, cb, winner))

    examples: list[DecisionExample] = []
    for key in sorted(grouped):
        qid, turn, model_left, model_right = key
        votes = grouped[key]
        # All annotations for a canonical pair/question/turn should judge the same transcripts.
        transcript_pairs = {(a, b) for a, b, _ in votes}
        if len(transcript_pairs) != 1:
            raise ValueError(f"non-identical transcripts inside vote group {key}")
        left, right = next(iter(transcript_pairs))
        counts = Counter(w for _, _, w in votes)
        n = sum(counts.values())
        probs = [counts["model_a"] / n, counts["model_b"] / n, counts["tie"] / n]

        gid = f"q{qid}:turn{turn}:{model_left}:{model_right}"
        # Outcome-blind shuffle based on group identity, not labels.
        swap = int(_stable_digest(gid, seed="jev48-mtbench-display-v1"), 16) % 2 == 1
        if swap:
            left, right = right, left
            probs[0], probs[1] = probs[1], probs[0]
            display_models = [model_right, model_left]
        else:
            display_models = [model_left, model_right]

        state = "Two anonymous candidate conversations:\n" + _paired_conversation(left, right)
        examples.append(
            DecisionExample(
                id="mtbench:" + hashlib.sha256(gid.encode("utf-8")).hexdigest()[:20],
                family_id=f"mtbench:q{qid}",
                split=split_map[qid],
                domain="mtbench_human_preference",
                state=state,
                question="Which conversation would expert human evaluators prefer overall?",
                candidates=[
                    Candidate(id="conversation_a", text="Conversation A is better"),
                    Candidate(id="conversation_b", text="Conversation B is better"),
                    Candidate(id="tie", text="The two conversations are about equally good"),
                ],
                target_probs=probs,
                target_kind="empirical_distribution",
                metadata={
                    "source": "lmsys/mt_bench_human_judgments",
                    "source_revision": MTBENCH_REVISION,
                    "question_id": int(qid),
                    "turn": turn,
                    "canonical_models": [model_left, model_right],
                    "display_models": display_models,
                    "vote_count": n,
                    "vote_counts": {
                        "conversation_a": round(probs[0] * n),
                        "conversation_b": round(probs[1] * n),
                        "tie": round(probs[2] * n),
                    },
                    "model_names_hidden_from_input": True,
                    "display_order_swapped": swap,
                },
            )
        )
    return examples


def manifest(examples: Iterable[DecisionExample]) -> dict[str, Any]:
    examples = list(examples)
    by_split = Counter(e.split for e in examples)
    votes = [int(e.metadata.get("vote_count", 0)) for e in examples]
    multi = sum(v >= 2 for v in votes)
    return {
        "examples": len(examples),
        "split_counts": dict(sorted(by_split.items())),
        "multi_vote_examples": multi,
        "max_votes": max(votes, default=0),
        "source_revision": MTBENCH_REVISION,
        "split_seed": SPLIT_SEED,
        "sha256": hashlib.sha256(
            "\n".join(e.canonical_json() for e in sorted(examples, key=lambda x: x.id)).encode("utf-8")
        ).hexdigest(),
    }
