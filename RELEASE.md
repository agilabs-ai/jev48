# Jev48 release state

The Edge Labs AI source repository and verified model checkpoint are public.

## Destinations

- Source and launch page: `edgelabs-ai/jev48`
- Model checkpoint: https://github.com/edgelabs-ai/jev48/releases/tag/v1.0.0
- Release version: `v1.0.0`

## Frozen artifacts

- Modal run: `jev48-1789867911`
- Modal volume: `jev48-artifacts`
- Selected trial: `soft-lr3e-6`
- Model size: 3,763,692,048 bytes
- Model SHA-256: `20948bb0163f7230d3922e292e919a62cf6c0e0c7600718ab0d152b693227aec`
- Complete private manifest: `jev48-1789867911/release/MANIFEST.json`

## Public verification path

The release carries a configuration archive, two weight parts, and `SHA256SUMS`.
Follow [`MODEL_CARD.md`](MODEL_CARD.md) to verify the parts, reassemble the exact
3,763,692,048-byte `model.safetensors`, and verify its frozen SHA-256 before loading.

Source and checkpoint are public at https://github.com/edgelabs-ai/jev48.
