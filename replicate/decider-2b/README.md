# Decider-2B on Replicate

Public, scale-to-zero packaging for [Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b), an Apache-2.0 one-pass typed decision model based on Qwen3.5-2B-Base.

**Live model:** [replicate.com/getedge/decider-2b](https://replicate.com/getedge/decider-2b)

The model accepts a state plus Jev-shaped `choice`, `noul`, and `score` questions and returns calibrated probabilities without generating prose. The Replicate playground and API handle access, metering, and billing; no Edge Labs proxy is required.

## Source and attribution

- Model and inference code: [Mapika/decider](https://github.com/Mapika/decider)
- Weights: [Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b)
- License: Apache-2.0
- Replicate packaging: [Edge Labs / Jev48](https://github.com/edgelabs-ai/jev48)

This is an independent community deployment of Mapika's model. It is not affiliated with TypeSafe or Jev.

## API example

```python
import replicate

output = replicate.run(
    "getedge/decider-2b",
    input={
        "state": "A customer reports a duplicate card charge.",
        "questions_json": '{"action":{"type":"choice","instructions":"What should happen next?","criteria":{"approve":"Refund automatically","review":"Send to a human","reject":"Reject the request"}}}',
        "independent": True,
    },
)
```

Replicate handles the playground, API authentication, metering, per-user billing,
and scale-to-zero runtime. Callers need their own Replicate account and credit.
