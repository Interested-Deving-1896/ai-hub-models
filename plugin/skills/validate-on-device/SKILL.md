---
name: validate-on-device
description: Compile, profile, and (if evaluator wired) evaluate a float recipe on a Snapdragon device via AI Hub. Iterates on failures using `.claude/docs/on-device-debugging.md`. Use after `onboard` produces a `validate`-green float recipe, and before `add-quantization`.
---

# Validate a Float Recipe on Device

`onboard` proves the recipe is *authored* correctly. This skill proves it **compiles, profiles, and (if an evaluator is wired) evaluates** on a real device at `float`. No quantization here — `supported_precisions:` stays `[float]`.

## Prerequisites

- `qai-hub-models validate <path>` prints `0 failed`. If not, go back to `onboard` — do NOT paper over validate failures to reach this skill.
- User's AI Hub token is configured (`qai-hub configure --api_token ...`). If not, stop and tell them.
- `supported_precisions:` in `manifest.yaml` includes `float`. If not, this skill has nothing to check.

## Pass criteria (all must hold — do not declare success on a subset)

1. **Compile** — Hub job for `--target-runtime qnn_dlc --precision float` on the manifest's `default_device` finishes `state == SUCCESS`.
2. **Profile** — completes with non-empty `layer_details`, `estimated_inference_time_ms`, `peak_memory_usage`. No `dspservice just died`, no unresolved rank / memory / segfault errors in the device log.
3. **Torch accuracy** *(only if evaluator wired)* — within ~5 pp of the reference number (model card / paper / sibling `numerics.yaml`) on a 100-sample subset, or matches the reference exactly if the model is deterministic on that many samples.
4. **On-device accuracy** *(only if evaluator wired)* — within ~1 pp of torch at float. Larger gap → preprocessing / postprocessing / dtype / layout issue. Quantization drift is a separate concern that belongs to `add-quantization`, not here.

No evaluator (no `get_evaluator()`, no `get_eval_dataset_classes()`)? Steps 1–2 only. Report explicitly: "float compile + profile pass; no evaluator wired, accuracy not checked."

## Commands

**Internal (in-tree) models:** prefix every `qai-hub-models` invocation with `QAIHM_DEV_MODE=1`. The model isn't published yet, so without it the CLI blocks on the unpublished-model prompt and rejects any `--precision` / `--target-runtime` not already listed in `supported_precisions:`. Standalone/external recipes don't need it.

Compile + profile:
```
qai-hub-models export <path> --target-runtime qnn_dlc --precision float \
    --skip-inferencing --skip-downloading --skip-summary
```

By default `export` submits both compile and profile. Read job URLs from stdout and wait for both. For jobs >5 min, write the polling loop to `${TMPDIR:-/tmp}/claude/poll_<job_id>.sh` and run via the Agent tool so the main conversation stays responsive.

Torch-only accuracy pre-check (fast, ~seconds):
```
qai-hub-models evaluate <path> --precision float --num-samples 100 --skip-device-accuracy
```

Torch + on-device accuracy (few minutes):
```
qai-hub-models evaluate <path> --target-runtime qnn_dlc --precision float --num-samples 100
```

Always run the torch-only pre-check first. If torch is already broken, on-device will be too, and torch failures are much cheaper to diagnose.

Use these forms verbatim, adjusting `<path>` and `--device` if the user asked for a specific device. Do not invent flags.

## Iterate-until-green

Same shape as `onboard`'s validate loop:

1. Run the next check in the sequence (compile → profile → torch eval → device eval).
2. Pass → move to the next check.
3. Fail → read the actual error, apply a targeted fix, rerun the **same** check. Do not skip ahead.
4. Repeat until all applicable checks pass, or the anti-runaway rule fires.

**Anti-runaway rule:** two consecutive attempts on the same check producing the same failure signature — same error class, same offending op, same delta from torch — stop iterating and report to the user. Never bypass a check by removing it, mocking it, or adding a `# noqa`-style suppression to the recipe.

## Failure taxonomy

See `.claude/docs/on-device-debugging.md` for the canonical list. Highlights:

**Compile:**
- `ImportError` / `ModuleNotFoundError` → missing `external_repos:` entry, self-referential import in a standalone recipe (see `onboard` rule), or missing `requirements.txt` line. Fix and rerun.
- Rank > 5 (`incorrect Rank 6`) → a reshape/permute produces a 6D intermediate. Restructure to stay ≤5D, verify torch output is numerically identical before resubmitting. See `.claude/docs/on-device-debugging.md` § Rank errors.
- Unsupported op → compile log names the specific op. Options: replace with an equivalent supported op, monkeypatch upstream to swap the impl, or slice the graph so the op runs on CPU (last resort).

**Profile:**
- Memory exceeded (`unable to tile`, `std::bad_alloc`) → walk resolution down; when the model has a resolution knob, walk it until profile succeeds and set that as `default_resolution` in the manifest. See `.claude/docs/on-device-debugging.md` § Resolution Search.
- `dspservice just died` → nearly always memory pressure; same fix path.
- Timeout → graph too large. Iterative structure (autoregressive decoding, refinement loops) → likely needs to be split into a `CollectionModel` (see `.claude/docs/collection-models.md`).

**Torch accuracy:**
- Way below reference → preprocessing is almost always the culprit. Verify resolution, normalization mean/std, channel order (RGB vs BGR), interpolation mode (bicubic vs bilinear) all match upstream. For timm-based models, `timm.data.resolve_data_config()` is ground truth.
- Slightly below → statistical noise on a 100-sample subset. Bump `--num-samples` to 500–1000 for a tighter check.

**On-device accuracy at float:**
- Torch fine, device off → nearly always dtype / layout. QNN typically expects `NHWC` inputs when the model was authored `NCHW`. Check `get_input_spec()` and `forward()` for dtype coercion.
- Postprocessing diverging on device → some ops (softmax, argmax, NMS) run differently on device. Heavy postprocessing → move it out of the compiled graph into the App layer on CPU.

## Reporting

Pass:
```
Float on-device validation: PASS
- Compile: <job_url>  (SUCCESS, <s>s)
- Profile: <job_url>  (SUCCESS, <ms>ms, <MB> peak)
- Torch:   <metric> on <n> samples (ref: <ref>)
- Device:  <metric> on <n> samples (Δ torch: <delta>)
```

Fail (anti-runaway hit): same shape, `PASS → FAIL`, paste the persistent error and what you tried. Let the user decide.

## Non-goals

- **Not adding quantization.** `supported_precisions:` stays `[float]`.
- **Not authoring the recipe.** If the recipe isn't `validate`-green, hand off to `onboard`.
- **Not tuning perf.** Once compile + profile succeed with reasonable numbers, ship it. Perf work is a separate project.
- **Not scorecarding.** In-tree recipes get scorecarded automatically by weekly CI (see `onboard-internal`); do not attempt to trigger it here.
