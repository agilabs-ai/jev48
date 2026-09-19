#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import random
from pathlib import Path

import httpx

from openjev.io import dump_json
from openjev.schema import Candidate, DecisionExample


BANKING_LABELS = [
    "activate_my_card","age_limit","apple_pay_or_google_pay","atm_support","automatic_top_up",
    "balance_not_updated_after_bank_transfer","balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed","cancel_transfer","card_about_to_expire","card_acceptance","card_arrival",
    "card_delivery_estimate","card_linking","card_not_working","card_payment_fee_charged",
    "card_payment_not_recognised","card_payment_wrong_exchange_rate","card_swallowed","cash_withdrawal_charge",
    "cash_withdrawal_not_recognised","change_pin","compromised_card","contactless_not_working","country_support",
    "declined_card_payment","declined_cash_withdrawal","declined_transfer","direct_debit_payment_not_recognised",
    "disposable_card_limits","edit_personal_details","exchange_charge","exchange_rate","exchange_via_app",
    "extra_charge_on_statement","failed_transfer","fiat_currency_support","get_disposable_virtual_card",
    "get_physical_card","getting_spare_card","getting_virtual_card","lost_or_stolen_card","lost_or_stolen_phone",
    "order_physical_card","passcode_forgotten","pending_card_payment","pending_cash_withdrawal","pending_top_up",
    "pending_transfer","pin_blocked","receiving_money","Refund_not_showing_up","request_refund",
    "reverted_card_payment?","supported_cards_and_currencies","terminate_account","top_up_by_bank_transfer_charge",
    "top_up_by_card_charge","top_up_by_cash_or_cheque","top_up_failed","top_up_limits","top_up_reverted",
    "topping_up_by_card","transaction_charged_twice","transfer_fee_charged","transfer_into_account",
    "transfer_not_received_by_recipient","transfer_timing","unable_to_verify_identity","verify_my_identity",
    "verify_source_of_funds","verify_top_up","virtual_card_not_working","visa_or_mastercard","why_verify_identity",
    "wrong_amount_of_cash_received","wrong_exchange_rate_for_cash_withdrawal",
]


def pretty(x: str) -> str:
    return x.replace("_", " ").replace("?", "").strip().lower()


def one_hot(k: int, idx: int) -> list[float]:
    row = [0.0] * k
    row[idx] = 1.0
    return row


def fetch_text(url: str) -> str:
    r = httpx.get(url, timeout=60.0, follow_redirects=True)
    r.raise_for_status()
    return r.text


def banking_rows(split: str) -> list[tuple[str, int]]:
    url = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{split}.csv"
    reader = csv.DictReader(io.StringIO(fetch_text(url)))
    by_name = {name: i for i, name in enumerate(BANKING_LABELS)}
    out = []
    for row in reader:
        label_name = row.get("category") or row.get("label")
        text = row.get("text") or row.get("query")
        if label_name not in by_name or not text:
            raise ValueError(f"unexpected Banking77 row: {row}")
        out.append((text, by_name[label_name]))
    return out


def boolq_rows(split: str) -> list[dict]:
    url = {
        "train": "https://storage.googleapis.com/boolq/train.jsonl",
        "validation": "https://storage.googleapis.com/boolq/dev.jsonl",
    }[split]
    return [json.loads(line) for line in fetch_text(url).splitlines() if line.strip()]


def make_banking(text: str, label: int, *, split: str, idx: int) -> DecisionExample:
    candidates = [Candidate(id=f"intent_{i}", text=pretty(name)) for i, name in enumerate(BANKING_LABELS)]
    return DecisionExample(
        id=f"banking77-{split}-{idx:05d}", family_id=f"banking77-{split}-{idx:05d}", split=split,
        domain="banking77", state=text,
        question="Which banking support intent best matches this customer request?",
        candidates=candidates, target_probs=one_hot(len(candidates), label), target_kind="deterministic_truth",
        metadata={"source":"BANKING77","license":"CC-BY-4.0","source_split":split},
    )


def make_boolq(row: dict, *, split: str, idx: int) -> DecisionExample:
    answer = bool(row["answer"])
    candidates = [Candidate(id="no", text="No"), Candidate(id="yes", text="Yes")]
    return DecisionExample(
        id=f"boolq-{split}-{idx:05d}", family_id=f"boolq-{split}-{idx:05d}", split=split,
        domain="boolq", state=row["passage"], question=row["question"], candidates=candidates,
        target_probs=[0.0,1.0] if answer else [1.0,0.0], target_kind="deterministic_truth",
        metadata={"source":"BoolQ","license":"CC-BY-SA-3.0","source_split":split},
    )


def classlabel_examples(dataset_id: str, hf_split: str, *, out_split: str, domain: str, question: str,
                        n: int, seed: int, text_columns: list[str], max_chars: int | None = 1400) -> list[DecisionExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install public-data deps: pip install -e '.[public-data]'") from exc
    ds = load_dataset(dataset_id, split=hf_split)
    names = list(ds.features["label"].names)
    rng = random.Random(seed)
    indices = list(range(len(ds))); rng.shuffle(indices)
    candidates = [Candidate(id=f"label_{i}", text=pretty(name)) for i, name in enumerate(names)]
    rows=[]
    for source_idx in indices:
        row=ds[source_idx]
        text="\n\n".join(str(row[c]) for c in text_columns if c in row and row[c])
        if max_chars is not None and len(text) > max_chars:
            continue
        label=int(row["label"])
        j=len(rows)
        rows.append(DecisionExample(
            id=f"{domain}-{out_split}-{j:05d}", family_id=f"{domain}-{out_split}-{j:05d}", split=out_split,
            domain=domain, state=text, question=question, candidates=candidates,
            target_probs=one_hot(len(candidates),label), target_kind="deterministic_truth",
            metadata={"source":dataset_id,"source_split":hf_split,"source_index":source_idx},
        ))
        if len(rows) >= n:
            break
    if len(rows) < n:
        raise ValueError(f"{dataset_id}:{hf_split} yielded only {len(rows)} rows after filters; requested {n}")
    return rows


def sample(rows, n, rng, predicate=None):
    rows=list(rows)
    if predicate is not None:
        rows=[row for row in rows if predicate(row)]
    rng.shuffle(rows)
    return rows[:min(n,len(rows))]



def permute_candidates(ex: DecisionExample, rng: random.Random) -> DecisionExample:
    order=list(range(len(ex.candidates)))
    rng.shuffle(order)
    data=ex.model_dump()
    data["candidates"]=[data["candidates"][i] for i in order]
    data["target_probs"]=[data["target_probs"][i] for i in order]
    data["metadata"]={**data.get("metadata", {}), "benchmark_candidate_permuted": True}
    return DecisionExample.model_validate(data)

def write_jsonl(path: Path, rows: list[DecisionExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="data/public_decisions")
    p.add_argument("--benchmark-output",default="data/benchmark/open_decision_bench.jsonl")
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--banking-train",type=int,default=2500)
    p.add_argument("--boolq-train",type=int,default=2500)
    p.add_argument("--test-per-seen-domain",type=int,default=300)
    p.add_argument("--ood-per-domain",type=int,default=300)
    args=p.parse_args()
    rng=random.Random(args.seed)
    out=Path(args.output)

    bank_train=banking_rows("train")
    rng.shuffle(bank_train)
    train_bank=bank_train[:args.banking_train]
    dev_bank=bank_train[args.banking_train:args.banking_train+200]
    cal_bank=bank_train[args.banking_train+200:args.banking_train+400]
    bank_test=sample(banking_rows("test"),args.test_per_seen_domain,rng)

    # Keep the public comparison compatible with NanoJev's released max_length=512
    # checkpoint without truncating evidence after the fact. Character filtering is
    # deliberately conservative; the exact tokenizer is still validated at runtime.
    max_boolq_chars = 1400
    bq_train=[row for row in boolq_rows("train") if len(row.get("passage", "")) <= max_boolq_chars]
    rng.shuffle(bq_train)
    train_bq=bq_train[:args.boolq_train]
    dev_bq=bq_train[args.boolq_train:args.boolq_train+200]
    cal_bq=bq_train[args.boolq_train+200:args.boolq_train+400]
    bq_test=sample(boolq_rows("validation"),args.test_per_seen_domain,rng,
                   predicate=lambda row: len(row.get("passage", "")) <= max_boolq_chars)

    splits={
        "train":[make_banking(t,y,split="train",idx=i) for i,(t,y) in enumerate(train_bank)] +
                [make_boolq(r,split="train",idx=i) for i,r in enumerate(train_bq)],
        "dev":[make_banking(t,y,split="dev",idx=i) for i,(t,y) in enumerate(dev_bank)] +
              [make_boolq(r,split="dev",idx=i) for i,r in enumerate(dev_bq)],
        "calibration":[make_banking(t,y,split="calibration",idx=i) for i,(t,y) in enumerate(cal_bank)] +
                      [make_boolq(r,split="calibration",idx=i) for i,r in enumerate(cal_bq)],
        "test":[make_banking(t,y,split="test",idx=i) for i,(t,y) in enumerate(bank_test)] +
               [make_boolq(r,split="test",idx=i) for i,r in enumerate(bq_test)],
    }

    # Entirely unseen public domains. Nothing from these datasets enters training.
    ood=[]
    ood += classlabel_examples(
        "fancyzhx/dbpedia_14","test",out_split="ood",domain="dbpedia14",
        question="Which category best describes this entity or article?",n=args.ood_per_domain,seed=args.seed+101,
        text_columns=["title","content"],
    )
    ood += classlabel_examples(
        "szhuggingface/ag_news","test",out_split="ood",domain="ag_news",
        question="Which news category best describes this article?",n=args.ood_per_domain,seed=args.seed+202,
        text_columns=["text"],
    )
    splits["ood"]=ood

    # Freeze semantic candidate order as a deterministic permutation so models cannot
    # benefit from canonical class positions. The target vector is permuted identically.
    perm_rng=random.Random(args.seed + 99991)
    splits["test"]=[permute_candidates(ex,perm_rng) for ex in splits["test"]]
    splits["ood"]=[permute_candidates(ex,perm_rng) for ex in splits["ood"]]

    for split,rows in splits.items():
        if split in {"train","dev","calibration"}:
            rng.shuffle(rows)
        write_jsonl(out/f"{split}.jsonl",rows)

    benchmark=splits["test"]+splits["ood"]
    benchmark_path=Path(args.benchmark_output); write_jsonl(benchmark_path,benchmark)
    manifest={
        "seed":args.seed,"counts":{k:len(v) for k,v in splits.items()},"benchmark_rows":len(benchmark),
        "seen_domains":["banking77","boolq"],"held_out_domains":["dbpedia14","ag_news"],
        "licenses":{"banking77":"CC-BY-4.0","boolq":"CC-BY-SA-3.0","dbpedia14":"CC-BY-SA-3.0","ag_news":"Apache-2.0 (szhuggingface/ag_news benchmark copy)"},
        "policy":"Official/held-out test data never enters training; DBpedia14 and AG News are domain-OOD; all benchmark candidate orders are deterministically permuted.",
    }
    dump_json(out/"manifest.json",manifest)
    print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
