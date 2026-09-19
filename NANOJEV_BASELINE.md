# NanoJev baseline

NanoJev is a required baseline, not an optional reference.

Current public NanoJev facts used by this project:

- source: `TianyuCodings/NanoJev`
- public model: `C-Tianyu/NanoJev`
- backbone: `Qwen/Qwen3-0.6B`
- dynamic choice: 2–255 candidates
- zero output-token decoding
- root checkpoint is the earlier navigation release
- public code reports complete candidate paths batched in one backbone call and **no prefix sharing**

## Why we did not fork it

Open System One was built independently around:

- a simpler reusable library layout
- semantic decision benchmarks
- simulator-grounded probability targets
- explicit OOD testing
- calibration utilities
- hard-case teacher distillation hooks

NanoJev's implementation and published methodology were inspected as a research/reference baseline. We should credit it clearly.

If later experiments show the fastest route to a better model is to initialize from NanoJev's public stage-1 or final checkpoint, that becomes a **new experimental branch** and must be labeled as such. Do not blur it with the independent v0.

## Run NanoJev on our benchmark

Clone the pinned source revision:

```bash
git clone https://github.com/TianyuCodings/NanoJev.git vendor/NanoJev
cd vendor/NanoJev
git checkout 71a513bb0163b5634467842b523ee0c0ed6fb1c7
cd ../..
```

Download its public root checkpoint:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="C-Tianyu/NanoJev",
    local_dir="checkpoints/nanojev",
    allow_patterns=[
        "best.safetensors",
        "config.json",
        "tokenizer/*",
        "backbone_config/*",
    ],
)
```

Then:

```bash
PYTHONPATH=. python scripts/benchmark_nanojev.py \
  --data data/benchmark/semantic_smoke.jsonl \
  --nanojev-repo vendor/NanoJev \
  --checkpoint-dir checkpoints/nanojev \
  --output results/nanojev_semantic_smoke.jsonl
```

The CLI mode includes model-load time in its wall clock, so use it for quality first. For fair latency, start NanoJev's persistent server and run:

```bash
PYTHONPATH=. python scripts/benchmark_nanojev.py \
  --data data/benchmark/semantic_smoke.jsonl \
  --service-url http://127.0.0.1:8765 \
  --output results/nanojev_semantic_smoke.jsonl
```
