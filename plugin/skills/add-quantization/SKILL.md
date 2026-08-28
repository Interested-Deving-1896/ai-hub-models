---
name: add-quantization
description: Add a quantized precision to a float recipe that already passes `validate-on-device`. Tries w8a8, falls back to w8a16, or gives up. Hard-requires a dataset + evaluator (authors them under supervision if missing). Use after `validate-on-device`.
---

# Add Quantization

Shallow first pass. Try w8a8; if unreasonable, try w8a16; if both fail, revert to `[float]` and stop. No mixed precision / sensitivity search / per-op tuning — that's follow-up work.

## Prerequisites (stop and report if missing)

- `validate-on-device` passed on the float recipe.
- `model.py` implements `get_evaluator()`, `get_eval_dataset_classes()`, `get_calibration_dataset_cls()`, and the returned dataset class exists (either under `qai_hub_models/datasets/` listed in `manifest.yaml`'s `datasets:`, or as a local `dataset.py` in the recipe folder — both are fine).

If the dataset/evaluator is missing, prompt the user to wire an existing one from `qai_hub_models/datasets/` + `templates/…` or author new ones (see `.claude/docs/onboarding/datasets-and-evaluators.md`). Once wired, rerun `validate-on-device` at float, then come back.

## Command

```
qai-hub-models evaluate <path> --target-runtime qnn_dlc --precision w8a8 --num-samples 100
```

No separate "torch pre-check" — without `--compute-quant-cpu-accuracy`, torch runs *float* (already checked in `validate-on-device`). Go straight to the on-device quantized run and compare to the float number `validate-on-device` produced.

## Flow

1. **Try w8a8.** Add to `supported_precisions:`, run the command. If drop from float is within the threshold below, **done**.
2. **Fall back to w8a16.** Remove `w8a8`, add `w8a16`, rerun with `--precision w8a16`. Same threshold check.
3. **Give up.** If both fail, revert `supported_precisions:` to `[float]` and report the numbers plus what follow-up work would look like (mixed precision, per-channel schemes, different calibration data). Do NOT try those yourself.

Iterate at most **once** on a failure, and only if the fix is an obvious wiring problem: wrong `get_calibration_dataset_cls()` return, calibration dataset returning wrong dtype/shape/range, evaluator needing a per-precision hyperparameter. Anything else — stop.

## Thresholds (max drop from float)

| Task | Threshold |
|------|-----------|
| Classification (Top-1) | 3 pp |
| Detection (mAP) | 5 pp |
| Segmentation (mIoU) | 3 pp |
| Super-resolution (PSNR) | 1 dB |
| Speech-to-text (WER) | 2 pp |
| LLM | see closest sibling's `numerics.yaml` |

For anything else, check a sibling's `numerics.yaml` — any drop smaller than what it accepted is fine.

## Reporting

Success:
```
Quantization: PASS — chose <w8a8|w8a16>
- float:       <metric>  (from validate-on-device)
- <precision>: <metric>  (drop: <delta>)
```

Failure:
```
Quantization: FAILED — reverted to [float]
- w8a8:  <metric>  (drop: <delta>) — over threshold
- w8a16: <metric>  (drop: <delta>) — over threshold

Follow-up: mixed precision (w8a8_mixed_int16), per-channel schemes, different calibration data.
```

Hand off to the user for the PR either way.
