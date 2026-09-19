# Results — local pre-GPU validation

These numbers validate the pipeline only. They are **not evidence that the model matches Jev**.

## CPU feature baseline

Source: `results/cpu_smoke.json`

| Split | Argmax agreement | Brier | NLL | TV |
|---|---:|---:|---:|---:|
| Seen-domain test | 0.760 | 0.1101 | 0.7111 | 0.2029 |
| Unseen equipment OOD | 0.542 | 0.1960 | 0.6931 | 0.2802 |

On OOD, this baseline becomes effectively uniform. That is a feature of the evaluation design: unseen-domain failure is visible.

## Tiny dynamic decision model

Source: `results/tiny_cpu_train.json`

150 training steps with a tiny local mean-embedding backbone and the same dynamic decision head used by the real training path.

| Split | Argmax agreement | Brier | NLL | TV |
|---|---:|---:|---:|---:|
| Seen-domain test | 0.760 | 0.1012 | 0.6967 | 0.1906 |
| Unseen equipment OOD | 0.458 | 0.1982 | 0.6954 | 0.2817 |

Fitted calibration temperature: **1.1363**.

The important result here is not quality. It is that the exact variable-candidate head, soft target losses, calibration stage, and evaluation suite execute end-to-end before any GPU spend.
