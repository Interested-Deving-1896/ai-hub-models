# Model Onboarding Agent

Use this agent when adding a new model to qai_hub_models.

## Shell Command Best Practices

- **Never write inline Python with comments in bash commands.** A multiline Python string containing `#` comment lines triggers Claude Code's tool-use approval prompt, blocking execution. Instead, write a `.py` file to `/tmp/claude/` and run it with `python /tmp/claude/script.py`. Always use `/tmp/claude/` (not `/tmp/` directly) to avoid affecting other files.
- **Never chain shell commands with `&&`, `||`, `;`, pipes, or redirects.** Claude Code's tool-use approval matches on the first command in the string and can be confused by chaining or redirects. Run each command as a separate Bash call.
- **For long-running jobs** (AI Hub compile/profile, which take 5-15 minutes), write the job submission and polling logic to a script file and run it as a background task. Use the Agent tool to run it in a subagent so the main conversation stays responsive.
- **Prefer script files for anything over ~5 lines of Python.** It's faster, avoids escaping issues, and the script can be re-run easily.

## Design Goals

1. **Modify PyTorch code to work around compilation failures.** The goal is to get models compiling and running on device, even if it requires changes to the original model architecture.

2. **PyTorch code cannot depend on GPU.** All models must run on CPU. Look at existing models for tricks to remove GPU dependencies if you get stuck.

3. **Prefer monkeypatching over SourceAsRoot.** When modifying external model code, use monkeypatching techniques first. `SourceAsRoot` (copying source into the repo) should always be the last resort.

4. **Prefer pip install from GitHub over SourceAsRoot.** If a package isn't on PyPI, install directly from GitHub (e.g., `git+https://github.com/...`). After installing, verify the model code is actually importable — some packages install successfully but exclude the model architecture code. Use `SourceAsRoot` only if direct GitHub install is not possible or the installed package is incomplete.

5. **Merge pre/postprocessing into the model.** Include as much preprocessing and postprocessing as possible in the model itself to simplify on-device implementation.

6. **Follow existing I/O conventions.** Models with existing examples (e.g., object detectors, segmentation) should follow the input/output format of similar models in the repo for consistency.

7. **Use shared code.** Check `qai_hub_models/utils/` for existing utilities before implementing your own pre/postprocessing. Look at similar models' `app.py` files and consolidate shared code when possible.

### Using SourceAsRoot

When pip install doesn't work or the installed package is incomplete, use `SourceAsRoot` from `qai_hub_models.utils.asset_loaders`. This clones a GitHub repo at a pinned commit and temporarily adds it to `sys.path`. Look at existing examples for patterns:

- **Simple**: `gkt/model.py`, `fomm/model.py` — clone repo, import model code, load weights
- **With patches**: use `find_replace_in_repo()` or monkeypatching to fix incompatibilities
- **With subdirectories**: add nested paths via `sys.path.insert()`

Key steps:
1. Find the repo URL and a stable commit hash
2. In `from_pretrained()`, use `with SourceAsRoot(url, commit, model_id, version):` to clone and import
3. Find or download pretrained weights (use `CachedWebModelAsset` for URLs)
4. Apply any patches needed for tracing compatibility

## Terminology

- **model_id** - The folder name (e.g., `yolov7`, `ddrnet23_slim`)
- **model_name** - The published display name in info.yaml (e.g., "YOLOv7", "DDRNet23-Slim")

## Model Directory Structure

Each model lives in `qai_hub_models/models/<model_id>/` and requires:

### Required Files

0. **__init__.py** - Package initialization. Must export `App`, `Model`, and `MODEL_ID`. Look at any existing model's `__init__.py` for the pattern.

1. **model.py** - PyTorch model inheriting from `BaseModel`

   **Required methods:**
   - `from_pretrained(cls)` - classmethod to load pretrained weights; all args must have defaults
   - `get_input_spec()` - staticmethod returning `InputSpec` dict of `{input_name: (shape, dtype)}`
   - `get_output_spec()` - returns list of outputs

   **Optional overrides (have default implementations):**
   - `_sample_inputs_impl()` - provide real sample inputs instead of random data
   - `get_hub_compile_options()` / `get_hub_profile_options()` - custom AI Hub flags
   - `get_unsupported_reason()` - marks specific device attributes that can't be supported by this model (eg hexagon version)
   - `eval_datasets()` - list of dataset names for evaluation
   - `get_evaluator()` - return evaluator class for accuracy measurement
   - `calibration_dataset_name()` - dataset for quantization calibration
   - `get_hub_litemp_percentage(precision)` - returns percentage (0-100) of sensitive layers to keep in higher precision for mixed precision quantization (e.g., `w8a8_mixed_int16`)

2. **app.py** - End-to-end application with pre/post-processing
   - `App` class taking a `Callable` (works with PyTorch or on-device inference)
   - `predict()` method for inference
   - Use `app_to_net_image_inputs()` from `image_processing.py` for standard image input conversion
   - Use utilities from `draw.py` and `bounding_box_processing.py` for postprocessing overlays

3. **demo.py** - CLI demo running the app on sample data
   - Parse args, init model, run app, display/save results
   - Use `model_from_cli_args()` from `args.py` for argument parsing and model loading
   - Use `load_image()` from `asset_loaders.py` for loading sample inputs
   - Use `display_or_save_image()` from `display.py` for output

4. **test.py** - Unit tests
   - `test_task`: PyTorch model accuracy
   - `test_trace`: TorchScript accuracy (mark with `@pytest.mark.trace`)
   - `test_demo`: Demo runs without error

5. **info.yaml** - Model metadata for public website
   - Auto-fill size/params: `python qai_hub_models/scripts/autofill_info_yaml.py -m <model_id>`
   - Fields like `use_case`, `tags`, `domain`, and `license_type` are validated enums. Check `qai_hub_models/configs/_info_yaml_enums.py` for valid values. If your use case doesn't exist, you'll need to add it to the enum.

### Optional Files

- **code-gen.yaml** - Custom options for export.py generation
- **requirements.txt** - Model-specific dependencies (pinned versions required)

### Auto-generated Files

- `README.md`, `export.py`, `test_generated.py` via `python qai_hub_models/scripts/run_codegen.py -m <model_id>`
- `evaluate.py` - Only generated if model defines `eval_datasets()` and `get_evaluator()`
- `perf.yaml` - Generated weekly by CI

## Architecture Patterns

### Base Classes (`qai_hub_models/utils/base_model.py`)

- **BaseModel** - Standard single-model, inherits `torch.nn.Module`
- **CollectionModel** - Multi-component models (e.g., encoder-decoder where components are compiled separately)
- **PrecompiledWorkbenchModel** - Pre-compiled assets only (no PyTorch source available)

### Shared Components

- `models/_shared/` - Reusable model components (LLM tokenizers, pose estimation, etc.)
- `extern/` - Safe imports for optional deps (numba, xtcocotools, git)

### Key Utilities (`qai_hub_models/utils/`)

Before writing preprocessing, postprocessing, or CLI code from scratch, check these utility modules — they contain standard implementations that most models use.

**Image processing** (`image_processing.py`):
- `app_to_net_image_inputs()` — standard way to convert PIL/numpy images to model input tensors in app.py. Handles batching, channel ordering, and normalization to [0,1].
- `preprocess_PIL_image()` — converts a PIL image to a float32 torch tensor in [0,1]
- `normalize_image_torchvision()` — applies ImageNet mean/std normalization
- `pil_resize_pad()` / `pil_undo_resize_pad()` — resize maintaining aspect ratio with padding (and inverse)
- `torch_tensor_to_PIL_image()` — convert model output back to PIL

**CLI argument parsing** (`args.py`):
- `get_model_cli_parser()` / `model_from_cli_args()` — standard demo argument parsing and model instantiation
- `get_on_device_demo_parser()` / `validate_on_device_demo_args()` — on-device demo args
- `add_output_dir_arg()` — adds `--output-dir` to any parser

**Asset loading** (`asset_loaders.py`):
- `CachedWebModelAsset` — download and cache model weights from URLs
- `load_image()` — load a PIL image from URL or local path (used in demos for sample inputs)
- `SourceAsRoot` — clone a GitHub repo and temporarily add to sys.path

**Display and drawing** (`display.py`, `draw.py`):
- `display_or_save_image()` — show image in GUI/notebook or save to disk (used in every demo.py)
- `draw_box_from_xyxy()`, `draw_points()`, `create_color_map()` — drawing overlays for detection/pose/segmentation

**Comparison** (`compare.py`):
- `compute_psnr()` — PSNR between two arrays (used in denoising/super-res evaluators)

**Bounding boxes** (`bounding_box_processing.py`):
- `batched_nms()`, `get_iou()`, `box_xywh_to_xyxy()` — standard detection postprocessing

Always look at a similar existing model's app.py and demo.py to see which utilities they import — this is the fastest way to find the right functions for your task.

## Onboarding Workflow

See also `CONTRIBUTING.md` at the repo root for the official contribution guide.

1. Create folder `qai_hub_models/models/<model_id>/`
2. Implement required files: `__init__.py`, `model.py`, `app.py`, `demo.py`, `test.py`, `info.yaml` (and `code-gen.yaml` if non-default options are needed)
3. Add `requirements.txt` if model needs additional dependencies
4. Run codegen: `python qai_hub_models/scripts/run_codegen.py -m <model_id>`
5. Auto-fill info.yaml: `python qai_hub_models/scripts/autofill_info_yaml.py -m <model_id>`
6. Run verification steps (see acceptance criteria below)

## Acceptance Criteria

A model is fully onboarded when all of the following pass:

### Local checks
- [ ] `python -m pytest qai_hub_models/models/<model_id>/test.py -v` — all tests pass (test_task, test_trace, test_demo)
- [ ] `pre-commit run --files qai_hub_models/models/<model_id>/*` — no failures
- [ ] `python qai_hub_models/scripts/run_codegen.py -m <model_id>` — generates export.py, README, etc. (note: pre-commit hook "failures" that auto-fix files are expected; re-run to confirm clean)
- [ ] Model output matches the reference implementation numerically on the same input

### On-device checks
- [ ] `python -m qai_hub_models.models.<model_id>.export` — compilation succeeds
- [ ] Profiling succeeds on device (no memory exceeded, no DSP crashes)
- [ ] Majority of ops run on NPU (check profile data for CPU fallback ops)

### Metadata
- [ ] `info.yaml` has all required fields filled in (name, id, status, headline, description, use_case, domain, license_type, research_paper, source_repo)
- [ ] `info.yaml` technical_details includes parameter count, model size, and input resolution. Use `python qai_hub_models/scripts/autofill_info_yaml.py -m <model_id>` to generate these values rather than computing them manually — this ensures they match the compiled model.
- [ ] Model license is verified. **Non-commercial licenses (CC-BY-NC, etc.) cannot be added — stop immediately and inform the user.** GPL/copyleft licenses are acceptable only if the model itself is released under that license. Permissive licenses (Apache 2.0, MIT, BSD) are preferred.
- [ ] Dataset license is also verified — same rules apply. Check the dataset source for license restrictions before using it.

### Demo quality
- [ ] Demo runs successfully with default sample inputs
- [ ] Demo output is qualitatively reasonable (e.g., classifier picks the correct class, detector boxes align with objects, segmentation mask matches the scene, restored image is visibly cleaner). Compare against the reference implementation's output on the same input — don't just check that it "produces something."

### Assets (if applicable)
- [ ] Test images and expected outputs uploaded to S3

## Preprocessing Verification

**Always verify the model's expected input range before writing any code.** This is the single most common source of silently wrong results. The model will trace, export, and even produce plausible-looking output with the wrong normalization — but the output will be garbage.

- Read the `weights.transforms()` source or the official tutorial/demo code
- Common ranges: `[0, 1]`, `[-1, 1]` (normalize with mean=0.5, std=0.5), `[0, 255]`, ImageNet mean/std
- **Also verify input resolution.** Check the model's default config or documentation for the expected input size.
- **Align inputs to repo standards.** Image based inputs are expected to be in the range [0, 1]. Any conversion to the model's native format should happen inside `forward()`.
- After implementing, compare your model's output against the reference implementation's output on the same input. Don't just check that the output "looks reasonable" — do a numerical comparison.

## On-Device Deployment and Debugging

### First Export

Run the export script at the model's native resolution. If compilation or profiling fails, read `.claude/docs/on-device-debugging.md` for guidance on diagnosing rank errors, memory failures, DSP crashes, and resolution search.

### Iterative/Recurrent Models

If a model has loops (refinement iterations, autoregressive decoding) and profiling fails with memory errors, the model likely needs to be split into a `CollectionModel`. Read `.claude/docs/collection-models.md` for the pattern, pitfalls, and examples.

## Adding Quantization Support

To enable quantized precision options (e.g., w8a8, w8a16) for a model:

### 1. Add a Dataset
Create or reuse a dataset in `qai_hub_models/datasets/`:
- Inherit from appropriate base class (e.g., `BaseDataset`)
- Implement data loading and preprocessing
- Register the dataset name

### 2. Add an Evaluator
Create or reuse an evaluator in `qai_hub_models/evaluators/`:
- Inherit from base evaluator class
- Implement accuracy metrics for your task (e.g., mAP for detection, IoU for segmentation)

### 3. Update the Model
In `model.py`, implement these methods:
```python
@staticmethod
def eval_datasets() -> list[str]:
    return ["<dataset_name>"]

@staticmethod
def calibration_dataset_name() -> str:
    return "<dataset_name>"

@classmethod
def get_evaluator(cls) -> type[BaseEvaluator]:
    return <YourEvaluator>
```

### 4. Update code-gen.yaml
Add supported precisions:
```yaml
supported_precisions:
  - float
  - w8a8
  - w8a16
```

### 5. Re-run Codegen
```bash
python qai_hub_models/scripts/run_codegen.py -m <model_id>
```

This will generate/update `evaluate.py` and add quantization options to `export.py`.

### 6. Test Quantized Accuracy
Run evaluate.py to verify quantization accuracy:
```bash
python -m qai_hub_models.models.<model_id>.evaluate --precision w8a8
```

Ensure accuracy drop from float is reasonable (10 points or less). If accuracy drop is too large, consider using mixed precision (e.g., `w8a8_mixed_int16`).

## S3 Assets

For model checkpoints or test data not available via public URLs:
- Run `python scripts/build_and_test.py validate_aws_credentials` first (prompts for password)
- Use AWS profile `qaihm` (e.g., `aws s3 cp --profile qaihm ...`)
- Upload to `qaihub-public-assets` S3 bucket under `qai-hub-models/models/<model_id>/v1/`
- Use versioned folders (v1, v2, ...) - assets cannot be deleted
- Set `MODEL_ASSET_VERSION = 1` in model.py
- Grant public-read access when uploading

## Requirements

- All packages in `requirements.txt` must be pinned to exact versions (e.g., `torch==2.0.1`)
- Check `global_requirements.txt` before adding new deps
- If a different version than global is required, set `global_requirements_incompatible: true` in `code-gen.yaml`
