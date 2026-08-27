# Writing Datasets and Evaluators

How to add new datasets and evaluators for model evaluation and quantization calibration.

## Datasets

Check `qai_hub_models/datasets/` for existing datasets before writing a new one. Browse the directory to find one that matches your task.

### License

**Always verify the dataset license before using it.** Non-commercial datasets cannot be used. Check the dataset source page for license terms. If the license is unclear, ask the user before proceeding.

### Writing a New Dataset

Inherit from `BaseDataset` in `qai_hub_models/datasets/common.py`. Look at existing datasets in the same directory for the pattern — they all implement the same interface.

Key requirements:
- `__getitem__()` returns a `(model_input, ground_truth)` tensor pair. Include a docstring describing both tensors' shapes, dtypes, and value ranges.
- `_download_data()` fetches data using `CachedWebDatasetAsset`
- `_validate_data()` checks that data was downloaded correctly
- Register the new dataset in `qai_hub_models/datasets/__init__.py`

**Fixed-size inputs**: All returned tensors must match the model's `input_spec` dimensions exactly. The calibration pipeline requires fixed-size inputs. Accept `input_spec` in `__init__()` and resize or center-crop each sample. Prefer resizing over cropping when applicable so the evaluator sees the full image content.

**Use standard image utilities**: For image-based datasets, use the helpers in `qai_hub_models/utils/image_processing.py` rather than writing manual preprocessing:
- `pil_resize_pad()` — resize to target dimensions while preserving aspect ratio (preferred over manual crop/resize)
- `preprocess_PIL_image()` — convert PIL image to float32 tensor in [0, 1]
- `normalize_image_torchvision()` — apply ImageNet normalization if the model expects it

## Evaluators

Check for existing evaluators before writing a new one. Evaluators live either in a shared folder under `qai_hub_models/models/_shared/<name>/` (when reused by more than one model) or in the model folder itself.

### Writing a New Evaluator

Inherit from `BaseEvaluator` in `qai_hub_models/utils/base_evaluator.py`. Place the new evaluator next to the model that uses it (or in a shared folder if it will be reused). Look at existing evaluators for the pattern.

Key requirements:
- `add_batch()` accumulates metrics comparing model output to ground truth
- `get_accuracy_score()` returns a single scalar (higher = better by default)
- `formatted_accuracy()` returns a human-readable string
- `get_metric_metadata()` returns a `MetricMetadata` describing the metric name, unit, acceptable thresholds, and direction

If the score is an error/distance metric where a smaller value is a better model (WER, NME, perplexity, RMSE, ...), the metric must set `higher_is_better=False`. Prefer reusing a registered metric from `qai_hub_models/utils/metrics.py`, which already declares direction; that flag flows into `numerics.yaml` and the published proto, so consumers don't have to guess from the metric name.

## Docstrings

All dataset `__getitem__` methods and evaluator `add_batch` methods must have docstrings describing the expected tensor shapes, dtypes, and value ranges. The model `forward()` method must also have a detailed docstring documenting inputs and outputs. Look at existing models for the expected format.
