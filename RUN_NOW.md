# Run now

The repository is locally complete up to the external-compute boundary.

## Round 1 — independent model vs NanoJev

No paid model APIs are required.

```bash
python -m pip install 'modal>=1.1,<2'
modal setup
modal run modal_app.py
```

This automatically:

1. builds real public train/test/OOD data
2. validates leakage
3. freezes and hashes the benchmark
4. trains the Qwen3-0.6B head-only model on A10
5. benchmarks the independent model
6. benchmarks untuned Qwen option likelihood
7. downloads and benchmarks the public NanoJev checkpoint on the exact same frozen rows
8. applies the predeclared launch gates
9. **only if head-only loses**, runs the LoRA fallback and checks the gates again
10. persists results/checkpoints into a Modal Volume

No OpenAI, Anthropic, or Jev key is needed for this round.

## Round 2 — only if independent model still loses

Run the explicitly derivative NanoJev++ branch:

```bash
modal run modal_app.py::nanojev_plus_gpu
```

This validates our data against NanoJev's own schema and then fine-tunes the public NanoJev checkpoint on the broader dataset on an A100.

This branch must be disclosed as derived from NanoJev.

## After round 1

Download the artifacts using the command printed by the Modal run. The key files are:

```text
open_decision_comparison.md
launch_gate.json
launch_gate_lora.json          # only if LoRA ran
ours_open_decision.summary.json
ours_lora_open_decision.summary.json  # only if LoRA ran
nanojev_open_decision.summary.json
```

Bring those files back into this chat and the next iteration can be driven from the actual failure slices rather than guesses.
