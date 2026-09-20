# Jev48 launch facts

Machine-generated from the locked result receipts. Do not edit numbers by hand.

- Selected reproduction: **soft-lr3e-6**.
- Public starting point: **Mapika/decider-2b** at pinned commit `b08acf787d5d1f718a8c36c4677960f43772c7be`.
- Locked human-preference test: Jev48 **62.1% accuracy**, **0.4110 Brier**.
- Locked transfer OOD: Jev48 **79.3% accuracy**, **0.3039 Brier**.
- Versus public starting point on human preference: **+3.9 pp accuracy**, **-0.0598 Brier** (negative Brier delta is better).
- Versus public starting point on transfer: **-0.7 pp accuracy**, **+0.0043 Brier**.
- Jev outputs used for training/model selection: **0**.
- External GPU experiment elapsed time: **2273.2702112197876 seconds**.
- Estimated NVIDIA H200 list cost: **$2.87** (list-price estimate only; see run manifest for exclusions).
- Live Jev comparison: **not run yet**.

## Claim guardrails

- Say: `I gave ChatGPT 48 hours/a weekend to recreate Jev using anything publicly available.`
- Do not say: `from scratch`.
- Do not imply the upstream architecture or weights were created by Jev48.
- If a metric is not in this file or the locked report, do not invent it.
