# NanoJev baseline

NanoJev is both the **upstream initialization** and the mandatory baseline for OpenJev.

Pinned upstream:

```text
source: TianyuCodings/NanoJev
commit: 71a513bb0163b5634467842b523ee0c0ed6fb1c7
model:  C-Tianyu/NanoJev
```

OpenJev v0 intentionally keeps NanoJev's architecture and modifies primarily the training distribution.

## Why

NanoJev already provides the hard plumbing:

- Qwen3-0.6B backbone
- dynamic 2–255 candidate choices
- decision heads
- complete distributions
- zero output-token decoding
- proper-scoring experiments

Reimplementing those components again does not improve the weekend marketing or research result.

The interesting experiment is whether a **small amount of additional semantic + calibrated training** broadens the model.

## Fair comparison

Both base NanoJev and OpenJev are evaluated on the exact same frozen benchmark rows and candidate order.

The benchmark is hashed before either model is evaluated.
