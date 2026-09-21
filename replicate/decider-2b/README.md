# Decider-2B on Replicate

[Run the model](https://replicate.com/getedge/decider-2b) ·
[Upstream model](https://huggingface.co/Mapika/decider-2b) ·
[Upstream code](https://github.com/Mapika/decider)

Public, scale-to-zero inference for Mapika's Apache-2.0 `decider-2b`: a
Qwen3.5-2B-Base model that answers typed decision questions with probability
distributions in one forward pass. It is useful when an application needs a
small structured decision rather than generated prose.

This is an independent community deployment maintained by Edge Labs. It is not
affiliated with TypeSafe or Jev, and it hosts the untouched Mapika checkpoint—not
the Jev48 experimental derivative.

## Inputs

| Input | Type | Description |
|---|---|---|
| `state` | string | The text or JSON-like context to evaluate. |
| `questions_json` | JSON string | Questions keyed by ID. Supports `choice`, `noul`, and `score`. |
| `independent` | boolean | Evaluate questions without cross-question influence. |

## Quick start

```python
import json
import replicate

questions = {
    "action": {
        "type": "choice",
        "instructions": "What should happen next?",
        "criteria": {
            "approve": "Safe to proceed automatically",
            "review": "Needs a human decision",
            "reject": "Should not proceed",
        },
    },
    "urgent": {
        "type": "noul",
        "instructions": "Does this need attention today?",
        "criteria": {
            "true": "Delay would materially worsen the outcome",
            "false": "It can safely wait",
        },
    },
}

output = replicate.run(
    "getedge/decider-2b",
    input={
        "state": "A customer reports a duplicate card charge.",
        "questions_json": json.dumps(questions),
        "independent": True,
    },
)
print(output)
```

Community models should be pinned to a version in production. Replicate exposes
the current version identifier in the model's API tab.

## Example questions

### Route a support case

```json
{
  "route": {
    "type": "choice",
    "instructions": "Which team should own this case?",
    "criteria": {
      "billing": "Payments, invoices, refunds, or duplicate charges",
      "security": "Account compromise, fraud, or suspicious access",
      "product": "Feature behavior or technical problems"
    }
  }
}
```

### Gate an automated action

```json
{
  "safe_to_execute": {
    "type": "noul",
    "instructions": "Can this action run without human review?",
    "criteria": {
      "true": "The action is reversible, authorized, and low risk",
      "false": "The action is irreversible, ambiguous, or high impact"
    }
  }
}
```

### Score risk

```json
{
  "risk": {
    "type": "score",
    "instructions": "How risky is this transaction?",
    "criteria": ["low", "medium", "high", "critical"]
  }
}
```

## Output

The model returns the upstream `system_one` response: an `answers` object keyed
by question ID, including probability distributions over the supplied choices.
Treat probabilities as model estimates, not guarantees.

## Runtime and billing

- Hardware: Nvidia T4 (`$0.000225` per active second at publication time).
- The caller pays Replicate for active processing time.
- Setup and idle time are not billed for public models; cold starts add latency.
- Edge Labs does not receive an automatic margin or revenue share from this
  community listing. A commercial margin requires a separately priced API or a
  direct marketplace/commercial agreement.

Check the live model page before relying on price or availability.

## Intended use and limitations

- Suitable for routing, ranking, moderation assistance, policy gates, and other
  typed decisions where probabilistic output is acceptable.
- Validate on your own distribution and choose thresholds on held-out data.
- Keep a human in the loop for consequential medical, legal, financial, safety,
  employment, or access-control decisions.
- Do not use this as the sole control against fraud, prompt injection, or other
  adversarial inputs.
- Community models can cold-start and do not carry an uptime SLA.

## Reproducibility

The container pins:

- Weights: `Mapika/decider-2b` revision
  `b37f7e1ba3fbc9238004cf531fabbee2619973fd`
- Inference code: `Mapika/decider` commit
  `c4daaac28af9fea95d627015cffa2dd5a5926ee6`
- Runtime interface: [`predict.py`](predict.py)
- Environment: [`cog.yaml`](cog.yaml) and [`requirements.txt`](requirements.txt)

The weights are downloaded during the image build and embedded in the published
container, so predictions do not download them from Hugging Face at request time.

## License and attribution

The upstream code and weights are published under Apache License 2.0, which
permits commercial use, hosting, modification, and redistribution subject to its
conditions. Preserve the license, copyright, patent, and attribution notices;
state significant modifications; do not imply endorsement or trademark rights.

- Model and inference code: [Mapika/decider](https://github.com/Mapika/decider)
- Weights: [Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b)
- License: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Replicate packaging: [Edge Labs / Jev48](https://github.com/edgelabs-ai/jev48)

This summary is not legal advice; the upstream repository and model card remain
the authoritative sources for the licensed artifacts.
