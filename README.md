# Evo 2 SAE Embeddings Exploration

## Setup Notes

- Due to the use of the `Transformer Engine` FP8 library for some layers, getting the embeddings/forward pass to work requires an H100/A100 GPU with the latest drivers (GPU with compute capabitility >= 8.9).
- To use Evo 2 40B, you need multiple GPUs.

## Usage

```bash
```

## Issues

- Retrieving the embeddings locally requires an H100.
- The response times from the `/forward` API from NVIDIA are highly variable, and often too slow.