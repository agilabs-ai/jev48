from __future__ import annotations

import math
import random
from typing import Callable

from .schema import Candidate, DecisionExample


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    ys = [math.exp(x - m) for x in xs]
    s = sum(ys)
    return [y / s for y in ys]


def _split_for(i: int, n: int) -> str:
    r = i / max(n, 1)
    if r < 0.65:
        return "train"
    if r < 0.75:
        return "dev"
    if r < 0.85:
        return "calibration"
    return "test"


def fraud_case(rng: random.Random, i: int, n: int) -> DecisionExample:
    age = rng.randint(0, 3650)
    failed = rng.randint(0, 20)
    mismatch = rng.random() < 0.22
    bad_ip = rng.random() < 0.18
    unusual_amount = rng.random() < 0.20
    known_device = rng.random() < 0.62
    logit = -2.3 + 1.4 * (age < 14) + 0.12 * failed + 1.1 * mismatch + 1.25 * bad_ip + 0.75 * unusual_amount - 0.9 * known_device
    p = sigmoid(logit)
    state = (
        f"Account age: {age} days. Failed logins in the last day: {failed}. "
        f"Billing address {'does not match' if mismatch else 'matches'} the payment profile. "
        f"IP reputation is {'poor' if bad_ip else 'normal'}. "
        f"Transaction amount is {'unusually high' if unusual_amount else 'typical'}. "
        f"The device is {'previously known' if known_device else 'new to this account'}."
    )
    return DecisionExample(
        id=f"fraud-{i:05d}", family_id=f"fraud-world-{i:05d}", split=_split_for(i, n), domain="fraud",
        state=state, question="Is this transaction fraudulent?",
        candidates=[
            Candidate(id="legit", text="The transaction is legitimate."),
            Candidate(id="fraud", text="The transaction is fraudulent."),
        ],
        target_probs=[1 - p, p], target_kind="known_distribution", metadata={"simulator": "fraud_v1"},
    )


def churn_case(rng: random.Random, i: int, n: int) -> DecisionExample:
    days_since = rng.randint(0, 90)
    tickets = rng.randint(0, 8)
    usage_change = rng.uniform(-0.9, 0.5)
    contract = rng.choice(["monthly", "annual"])
    nps = rng.randint(0, 10)
    logit = -1.7 + 0.032 * days_since + 0.25 * tickets - 1.4 * usage_change + 0.8 * (contract == "monthly") - 0.16 * nps
    p = sigmoid(logit)
    state = (
        f"The customer last used the product {days_since} days ago, opened {tickets} support tickets this quarter, "
        f"and usage changed by {usage_change * 100:+.0f}% month over month. They are on a {contract} contract and their latest NPS score is {nps}/10."
    )
    return DecisionExample(
        id=f"churn-{i:05d}", family_id=f"churn-world-{i:05d}", split=_split_for(i, n), domain="churn",
        state=state, question="Will this customer churn in the next 60 days?",
        candidates=[
            Candidate(id="stay", text="The customer remains active."),
            Candidate(id="churn", text="The customer churns within 60 days."),
        ],
        target_probs=[1 - p, p], target_kind="known_distribution", metadata={"simulator": "churn_v1"},
    )


def delivery_case(rng: random.Random, i: int, n: int) -> DecisionExample:
    distance = rng.randint(2, 1200)
    weather = rng.choice(["clear", "rain", "storm"])
    backlog = rng.randint(0, 100)
    premium = rng.random() < 0.25
    handoffs = rng.randint(0, 4)
    logit = -2.2 + 0.0015 * distance + 0.018 * backlog + 0.55 * handoffs + {"clear": 0, "rain": 0.6, "storm": 1.5}[weather] - 0.7 * premium
    p_late = sigmoid(logit)
    state = (
        f"Shipment distance is {distance} km. Weather is {weather}. The carrier backlog index is {backlog}/100. "
        f"The package uses {'premium' if premium else 'standard'} service and requires {handoffs} network handoffs."
    )
    return DecisionExample(
        id=f"delivery-{i:05d}", family_id=f"delivery-world-{i:05d}", split=_split_for(i, n), domain="delivery",
        state=state, question="Will this shipment arrive late?",
        candidates=[
            Candidate(id="on_time", text="The shipment arrives on time."),
            Candidate(id="late", text="The shipment arrives late."),
        ],
        target_probs=[1 - p_late, p_late], target_kind="known_distribution", metadata={"simulator": "delivery_v1"},
    )


def qualification_case(rng: random.Random, i: int, n: int) -> DecisionExample:
    employees = int(10 ** rng.uniform(0.7, 4.2))
    budget = int(10 ** rng.uniform(2.0, 6.0))
    urgency = rng.randint(1, 5)
    integrations = rng.randint(0, 10)
    authority = rng.randint(1, 5)
    enterprise = 0.00028 * employees + 0.000004 * budget + 0.35 * integrations + 0.4 * authority + 0.2 * urgency - 3.4
    selfserve = -0.00012 * employees + 0.000002 * budget - 0.12 * integrations + 0.25 * urgency + 0.2 * authority + 1.2
    notfit = 1.8 - 0.0000015 * budget - 0.22 * urgency - 0.18 * authority
    probs = softmax([enterprise, selfserve, notfit])
    state = (
        f"Prospect has {employees} employees, an indicated annual budget of ${budget:,}, urgency {urgency}/5, "
        f"needs {integrations} integrations, and the contact's buying authority is {authority}/5."
    )
    return DecisionExample(
        id=f"qualification-{i:05d}", family_id=f"qualification-world-{i:05d}", split=_split_for(i, n), domain="qualification",
        state=state, question="Which sales motion is most appropriate?",
        candidates=[
            Candidate(id="enterprise", text="Route to an enterprise account executive."),
            Candidate(id="self_serve", text="Route to self-serve or product-led onboarding."),
            Candidate(id="not_fit", text="Do not prioritize this lead right now."),
        ],
        target_probs=probs, target_kind="known_distribution", metadata={"simulator": "qualification_v1"},
    )


def equipment_ood_case(rng: random.Random, i: int, n: int) -> DecisionExample:
    vibration = rng.uniform(0.1, 5.0)
    temp = rng.uniform(35, 130)
    hours = rng.randint(0, 20000)
    oil = rng.uniform(0.1, 1.0)
    alarm = rng.random() < 0.15
    logit = -5.0 + 0.8 * vibration + 0.035 * (temp - 60) + 0.00012 * hours + 1.8 * (oil < 0.35) + 1.2 * alarm
    p = sigmoid(logit)
    state = (
        f"Machine vibration is {vibration:.1f} mm/s, bearing temperature is {temp:.0f} C, service age is {hours} hours, "
        f"oil level is {oil:.2f} of nominal, and an anomaly alarm is {'active' if alarm else 'not active'}."
    )
    return DecisionExample(
        id=f"equipment-{i:05d}", family_id=f"equipment-world-{i:05d}", split="ood", domain="equipment",
        state=state, question="Will the machine fail within the next operating shift?",
        candidates=[
            Candidate(id="no_failure", text="The machine completes the next shift without failure."),
            Candidate(id="failure", text="The machine fails during the next shift."),
        ],
        target_probs=[1 - p, p], target_kind="known_distribution", metadata={"simulator": "equipment_ood_v1"},
    )


SIMULATORS: dict[str, Callable[[random.Random, int, int], DecisionExample]] = {
    "fraud": fraud_case,
    "churn": churn_case,
    "delivery": delivery_case,
    "qualification": qualification_case,
}


def generate_dataset(seed: int = 17, seen_per_domain: int = 500, ood_count: int = 500) -> list[DecisionExample]:
    rng = random.Random(seed)
    examples: list[DecisionExample] = []
    for fn in SIMULATORS.values():
        domain_rng = random.Random(rng.randint(0, 2**31 - 1))
        examples.extend(fn(domain_rng, i, seen_per_domain) for i in range(seen_per_domain))
    ood_rng = random.Random(rng.randint(0, 2**31 - 1))
    examples.extend(equipment_ood_case(ood_rng, i, ood_count) for i in range(ood_count))
    return examples
