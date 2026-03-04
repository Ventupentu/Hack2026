# Hugging Face Hub Integration

This project can automatically sync training artifacts to a Hugging Face model repository.

## What is uploaded

When `hf.push_to_hub=true`, the training code uploads artifacts on:

- New best checkpoint (`best.pt`)
- Final checkpoint (`last.pt`)

Each upload includes:

- Checkpoint file (`best.pt` or `last.pt`)
- Metrics file:
  - OpenCLIP: `metrics.jsonl`
  - GR-Lite: `train_metrics.json`
- Hydra runtime config as `hydra_config.yaml`

Repo paths are namespaced by backend:

- `openclip/...`
- `grlite/...`

## Configuration

The root config contains:

```yaml
hf:
  push_to_hub: false
  hf_repo_id: ""
  hf_token: ""
```

Recommended usage is passing secrets at runtime:

```bash
export HF_TOKEN=hf_xxx
python -m src.train \
  hf.push_to_hub=true \
  hf.hf_repo_id=your-org-or-user/your-model-repo \
  hf.hf_token=$HF_TOKEN
```

## Strict error handling

If `hf.push_to_hub=true`, training fails fast at startup when:

- `hf.hf_token` is missing
- `hf.hf_repo_id` is missing
- Token is invalid (`whoami` check fails)
- Repo cannot be accessed or created
- Hydra runtime config file is missing

Upload failures are not ignored. Background upload exceptions are raised before training exits.

## Performance behavior

Uploads run in a background worker thread to avoid blocking the main training loop.

## Security behavior

- `hf_token` is redacted in uploaded `hydra_config.yaml`.
- `hf_token` is also redacted from checkpoint `args` payloads.

## Dependency

Make sure `huggingface_hub` is installed (included in `requirements.txt`).
