# Jev48 locked comparison

Selected on dev only: `soft-lr3e-6`. Locked test/OOD were not used for model selection.

| Model | Human preference acc ↑ | Human preference Brier ↓ | Human preference ECE ↓ | Transfer acc ↑ | Transfer Brier ↓ | Transfer ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| decider-2b (public starting point) | 0.5819 | 0.4708 | 0.1032 | 0.8000 | 0.2995 | 0.0397 |
| Jev48 / ChatGPT build | 0.6207 | 0.4110 | 0.0587 | 0.7926 | 0.3039 | 0.0457 |

Human preference = locked MT-Bench expert-vote groups over real model outputs. Transfer = pinned tasks held out by the public starting model's training registry.
The open base and Jev48 each receive one scalar temperature fitted on the same calibration rows. Jev's primary row uses the probabilities returned natively by the provider; any extra Jev temperature fit is retained only as a diagnostic.
Jev outputs are never used for training or candidate selection.
