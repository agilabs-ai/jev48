# Prior art / references

This is an independent implementation. It is not affiliated with TypeSafe and does not claim to reproduce TypeSafe's proprietary Jev architecture or training method.

Public projects that informed the design and benchmark plan:

- **NanoJev** — TianyuCodings/NanoJev (MIT). Demonstrates a Qwen3-0.6B backbone with dynamic decision heads, complete probability distributions, and zero output-token decoding.
- **open-jev** — daseinlabs/open-jev. Demonstrates one-pass / cached option-likelihood scoring with Gemma on Apple Silicon and is used as conceptual baseline prior art.

The code in this repository is written independently around the project goals in `PROJECT_SPEC.md`.
