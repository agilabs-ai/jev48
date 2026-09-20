# Jev48 release state

The AGI Labs source repository is public. The model checkpoint remains staged
privately because the AGI Labs Hugging Face namespace is not authenticated here.

## Destinations

- Source and launch page: `agilabs-ai/jev48`
- Model checkpoint: `agilabs-ai/jev48-2b`
- Release version: `v1.0.0`

## Frozen artifacts

- Modal run: `jev48-1789867911`
- Modal volume: `jev48-artifacts`
- Selected trial: `soft-lr3e-6`
- Model size: 3,763,692,048 bytes
- Model SHA-256: `20948bb0163f7230d3922e292e919a62cf6c0e0c7600718ab0d152b693227aec`
- Complete private manifest: `jev48-1789867911/release/MANIFEST.json`

## Remaining model publication hold

Before making the model checkpoint public:

1. authenticate the AGI Labs Hugging Face namespace;
2. upload privately, re-download by immutable commit, verify every hash, and run inference;
3. make the model public and verify the final URL;
4. create `v1.0.0` only after all intended public artifacts resolve.

The source repository is already public at https://github.com/agilabs-ai/jev48.
