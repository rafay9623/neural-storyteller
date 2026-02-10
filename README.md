# neural-storyteller

## Testing

From the project root:

```bash
python tests/test_app.py
```

Runs 7 checks: vocab/model files exist, vocab load, full resource load (model + ResNet + transform), caption generation (beam search), ResNet feature shape, and encoder shape. First run may download ResNet50 weights (~98 MB).