# Run OpenJev now

OpenJev now uses **NanoJev as its starting checkpoint**. There is no independent-model gate before the main experiment.

## 1. Authenticate Modal

```bash
python -m pip install 'modal>=1.1,<2'
modal setup
```

## 2. Run the full experiment

```bash
modal run modal_app.py
```

The run automatically:

1. builds public semantic data;
2. combines it with known-probability simulator data;
3. validates split/family/content leakage;
4. freezes and hashes `OpenDecisionBench`;
5. clones the pinned NanoJev source revision;
6. downloads the public NanoJev checkpoint;
7. benchmarks base NanoJev on the frozen benchmark;
8. validates OpenJev training data with NanoJev's own validator;
9. fine-tunes the NanoJev checkpoint into OpenJev;
10. benchmarks OpenJev on the identical rows;
11. applies the predeclared launch gate;
12. persists raw predictions, summaries, checkpoint, and comparison report.

No OpenAI, Anthropic, or Jev API key is required for this run.

## 3. Download results

The Modal run prints a command of this form:

```bash
modal volume get openjev-artifacts <run-name> ./modal-results
```

Bring these back into ChatGPT:

```text
openjev_vs_nanojev.md
openjev_gate.json
nanojev_open_decision.summary.json
openjev_open_decision.summary.json
```

Those files are sufficient to drive the next iteration from measured failure slices.

## Optional ablation

The original independent-from-scratch approach is preserved under `experiments/` and in the supporting scripts, but it is no longer the default launch path.
