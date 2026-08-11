# Guide: Export a HuggingFace Model to ONNX and Profile on Qualcomm AI Hub

This guide contains everything needed to take a HuggingFace model, export it to ONNX,
compile it for Snapdragon, and profile it on-device.

## Scope

This guide is **self-contained**. Do not reference or depend on the broader
`ai-hub-models-internal` repo (e.g. `qai_hub_models/`, `.claude/agents/`,
`.claude/docs/onboarding/`). Those follow a different workflow for adding models
to the QAIHM package. This guide is solely for the lightweight
HuggingFace → ONNX → AI Hub compile/profile pipeline.

## Environment Setup

```bash
python3 -m venv ${TMPDIR:-/tmp}/claude/hf_venv
PIP=${TMPDIR:-/tmp}/claude/hf_venv/bin/pip
$PIP install torch --index-url https://download.pytorch.org/whl/cpu
$PIP install --force-reinstall torchvision --index-url https://download.pytorch.org/whl/cpu
$PIP install optimum-onnx onnxruntime onnx qai_hub huggingface_hub
$PIP install timm sentence-transformers onnxscript
```

The PyPI package is `optimum-onnx`. In code, import from `optimum`:
`from optimum.exporters.onnx import main_export`.

After setup, run all scripts with `${TMPDIR:-/tmp}/claude/hf_venv/bin/python`.

## Architecture: One script per model

Generate a **separate Python script for each model** in `${TMPDIR:-/tmp}/claude/hf_scripts/`.
Also write `helpers.py` to that directory — its **complete source is included at the
end of this guide** (see the "helpers.py" section). No additional files or external
scripts are needed beyond what is defined in this guide.

Each script has **3 CLI steps**:

```bash
python model_script.py                  # run full pipeline
python model_script.py submit-compile   # export + upload + compile only
python model_script.py submit-profile   # wait for compiles, submit profiles
python model_script.py collect-results  # wait for profiles, print results
```

State persists to `<safe_name>_jobs.json` between steps. This lets you `submit-compile`
for many models in a loop, then `submit-profile` and `collect-results` later.

**helpers.py** provides:
- `export_optimum(model_id, task)` — Approach A export
- `export_optimum_no_token_type(model_id, task, model_class, config_class)` — Approach B export
- `export_torch(model_id, wrapper, sample_inputs, ...)` — Approach C export
- `read_onnx_input_spec(onnx_path)` — read input names/shapes/dtypes, resolve dynamic dims
- `extract_profile_results(profile_job)` — extract latency (ms) and NPU%
- `make_no_token_type_config(base_cls)` — OnnxConfig subclass factory
- `OutputWrapper` — wraps HF models to return single tensor

Hub API calls live **in each script** (not in helpers) so the flow is visible and editable.

## Choosing the export approach

**First, check if the model is BERT-family:**

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_id)
model_type = getattr(config, "model_type", "")

BERT_FAMILY = {"bert", "albert", "electra", "mobilebert", "convbert", "camembert",
               "roberta", "xlm-roberta", "squeezebert", "mpnet"}
```

- **BERT-family** → use **Approach B** (skip A — it silently produces a model that crashes on the NPU)
- **DistilBERT** → use **Approach A** (no token_type_ids issue)
- **CausalLM (GPT2, OPT)** → use **Approach C** (backbone extraction, no KV cache)
- **Everything else** → try **A**, fall back to **C**, then **D**

### Approach A: optimum main_export

Works for ~60% of non-BERT models. Use `export_optimum(model_id, task)` from helpers.

If `task="auto"` fails, map the model's `pipeline_tag` to an explicit task:

| pipeline_tag | task |
|---|---|
| image-classification | `image-classification` |
| object-detection | `object-detection` |
| image-segmentation | `semantic-segmentation` |
| depth-estimation | `depth-estimation` |
| feature-extraction, sentence-similarity | `feature-extraction` |
| fill-mask | `fill-mask` |
| text-classification, text-ranking | `text-classification` |
| token-classification | `token-classification` |
| text-generation | `text-generation` |
| translation | `seq2seq-lm` |
| automatic-speech-recognition | `automatic-speech-recognition` |

For `timm` models: `task="image-classification"`. For `sentence-transformers`: `task="feature-extraction"`.

### Approach B: optimum without token_type_ids (BERT-family)

Use `export_optimum_no_token_type(model_id, task, model_class, onnx_config_class)` from helpers.

You must pick the right model class and config class:

**Model class** (by pipeline_tag):
- fill-mask → `AutoModelForMaskedLM`
- text-classification → `AutoModelForSequenceClassification`
- token-classification → `AutoModelForTokenClassification`
- feature-extraction, sentence-similarity → `AutoModel`
- question-answering → `AutoModelForQuestionAnswering`

**OnnxConfig class** (by model_type):
- bert, electra, mpnet, squeezebert → `BertOnnxConfig`
- albert → `AlbertOnnxConfig`
- roberta → `RobertaOnnxConfig`
- camembert → `CamembertOnnxConfig`
- mobilebert → `MobileBertOnnxConfig`
- convbert → `ConvBertOnnxConfig`
- xlm-roberta → `XLMRobertaOnnxConfig`

All from `optimum.exporters.onnx.model_configs`.

**Why this fix is needed:** Optimum exports BERT with `token_type_ids` as a third input.
The model compiles fine but crashes on the Snapdragon NPU at runtime (QNN HTP error 1100,
`dspservice just died!`). This is a QNN runtime bug. Removing the input produces a
functionally identical model that runs correctly — `token_type_ids` is just zeros for
single-sentence use cases.

### Approach C: torch.onnx.export

Use `export_torch(model_id, wrapper, sample_inputs, input_names, ...)` from helpers.
Wrap the model with `OutputWrapper` from helpers to convert dict outputs to single tensors.

Always use `dynamo=False` — PyTorch 2.11+ defaults to the dynamo exporter which fails
on many models. The `export_torch` helper handles this.

**CausalLM backbone extraction (GPT2):**
```python
from transformers import AutoModel  # NOT AutoModelForCausalLM

class GPT2Encoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.wte, self.wpe, self.drop, self.h, self.ln_f = m.wte, m.wpe, m.drop, m.h, m.ln_f
    def forward(self, input_ids, attention_mask):
        pos_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.drop(self.wte(input_ids) + self.wpe(pos_ids))
        mask = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype)) * torch.finfo(hidden.dtype).min
        for block in self.h:
            hidden = block(hidden, attention_mask=mask)[0]
        return self.ln_f(hidden)
```

Use `AutoModel` (not `AutoModelForCausalLM`) to get the raw transformer without the
language model head. The `GPT2Encoder` strips the KV cache.

**Sample inputs by modality:**

| Type | sample_inputs | input_names |
|---|---|---|
| Text | `(torch.randint(0, 30000, (1, 128)), torch.ones(1, 128, dtype=torch.long))` | `["input_ids", "attention_mask"]` |
| Vision | `(torch.randn(1, 3, 224, 224),)` | `["pixel_values"]` |
| Audio (waveform) | `(torch.randn(1, 16000),)` | `["input_values"]` |

### Approach D: Direct ONNX download

For repos with pre-existing `.onnx` files. Use `export_direct_onnx(model_id, filename)`
from helpers. Watch out for `.onnx_data` external weight files (tiny `.onnx` = incomplete).

## Compile and profile details

These are handled by the template code below, but here are the key rules:

**Compile options:**
- `--truncate_64bit_io --truncate_64bit_tensors` for QNN DLC and TFLite when inputs are int64
- Do NOT pass truncation flags for ONNX runtime
- Upload the model ONCE and reuse for all 3 runtimes
- Hub does NOT support float16 inputs

**Input spec format:** `{name: (shape_tuple, dtype_string)}`
- e.g. `{"input_ids": ((1, 128), "int64")}`, `{"pixel_values": ((1, 3, 224, 224), "float32")}`
- Dynamic dim defaults: batch→1, sequence→128, height/width→224, channel→3

**Profile results:**
- `download_profile()` returns a **dict** with `"execution_summary"` and `"execution_detail"`
- Latency: `perf["execution_summary"]["estimated_inference_time"]` — in **MICROSECONDS**, divide by 1000 for ms
- NPU%: sum `execution_time` by `compute_unit`. QNN DLC reports 0 for all times — fall back to counting ops
- `extract_profile_results(profile_job)` in helpers handles all of this

**Job status:** `status = job.get_status()` — attributes: `.success`, `.running`, `.pending`, `.finished`, `.code`, `.message`. To resume: `job = hub.get_job(job_id)`.

## Per-model script template

Copy this template and adapt the `submit_compile` function for your model's approach.

```python
"""Export, compile, and profile: {model_id}"""
import json
import os
import sys

import qai_hub as hub

# Add helpers.py location to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import export_optimum, read_onnx_input_spec, extract_profile_results

MODEL_ID = "{model_id}"
SAFE_NAME = MODEL_ID.replace("/", "__")
DEVICE_NAME = "Samsung Galaxy S25 (Family)"
JOBS_FILE = os.path.join(os.path.dirname(__file__), f"{SAFE_NAME}_jobs.json")


def submit_compile():
    # ── EDIT THIS for model-specific export logic ──
    # Approach A: export_optimum(MODEL_ID, task="auto")
    # Approach B: export_optimum_no_token_type(MODEL_ID, task, model_class, config_class)
    # Approach C: export_torch(MODEL_ID, wrapper, sample_inputs, input_names, ...)
    print(f"Exporting {MODEL_ID}...")
    onnx_path = export_optimum(MODEL_ID, task="auto")

    input_spec, has_int64 = read_onnx_input_spec(onnx_path)
    print(f"  Input spec: {input_spec}")

    device = hub.Device(DEVICE_NAME)
    hub_model = hub.upload_model(onnx_path)
    print(f"  Uploaded: {hub_model.model_id}")

    jobs = {"model_id": MODEL_ID, "runtimes": {}}
    for runtime in ["qnn_dlc", "tflite", "onnx"]:
        opts = f"--target_runtime {runtime}"
        if has_int64 and runtime != "onnx":
            opts += " --truncate_64bit_io --truncate_64bit_tensors"
        try:
            cj = hub.submit_compile_job(
                model=hub_model, input_specs=input_spec, device=device,
                name=f"{SAFE_NAME}_{runtime}", options=opts,
            )
            jobs["runtimes"][runtime] = {"compile_job_id": cj.job_id}
            print(f"  {runtime}: compile submitted ({cj.job_id})")
        except Exception as e:
            jobs["runtimes"][runtime] = {"error": str(e)}
            print(f"  {runtime}: FAILED - {e}")

    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2)
    print(f"  Jobs saved to {JOBS_FILE}")


def submit_profile():
    with open(JOBS_FILE) as f:
        jobs = json.load(f)
    print("Waiting for compiles and submitting profiles...")
    device = hub.Device(DEVICE_NAME)

    for runtime in ["qnn_dlc", "tflite", "onnx"]:
        rt_info = jobs["runtimes"].get(runtime, {})
        cjid = rt_info.get("compile_job_id")
        if not cjid:
            continue
        compile_job = hub.get_job(cjid)
        compile_job.wait()
        status = compile_job.get_status()
        if not status.success:
            rt_info["result"] = f"compile FAILED - {status.message}"
            print(f"  {runtime}: compile FAILED - {status.message}")
            continue
        target_model = compile_job.get_target_model()
        profile_job = hub.submit_profile_job(model=target_model, device=device)
        rt_info["profile_job_id"] = profile_job.job_id
        print(f"  {runtime}: profile submitted ({profile_job.job_id})")

    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2)


def collect_results():
    with open(JOBS_FILE) as f:
        jobs = json.load(f)
    print("Collecting results...")

    for runtime in ["qnn_dlc", "tflite", "onnx"]:
        rt_info = jobs["runtimes"].get(runtime, {})
        pjid = rt_info.get("profile_job_id")
        if not pjid:
            continue
        profile_job = hub.get_job(pjid)
        profile_job.wait()
        latency_ms, npu_pct = extract_profile_results(profile_job)
        if latency_ms is not None:
            rt_info["latency_ms"] = latency_ms
            rt_info["npu_pct"] = npu_pct
            print(f"  {runtime}: {latency_ms}ms @ {npu_pct}% NPU")
        else:
            pstatus = profile_job.get_status()
            rt_info["result"] = f"profile FAILED - {pstatus.message}"
            print(f"  {runtime}: profile FAILED - {pstatus.message}")

    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2)

    print(f"\nResults: {MODEL_ID}")
    for rt in ["qnn_dlc", "tflite", "onnx"]:
        info = jobs["runtimes"].get(rt, {})
        if "latency_ms" in info:
            print(f"  {rt}: {info['latency_ms']}ms @ {info['npu_pct']}% NPU")
        else:
            print(f"  {rt}: {info.get('result', info.get('error', 'not attempted'))}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step == "submit-compile":
        submit_compile()
    elif step == "submit-profile":
        submit_profile()
    elif step == "collect-results":
        collect_results()
    elif step == "all":
        submit_compile()
        submit_profile()
        collect_results()
    else:
        print(f"Unknown step: {step}")
        print("Usage: python <script>.py [submit-compile|submit-profile|collect-results|all]")
        sys.exit(1)
```

**For BERT-family**, replace the export line in `submit_compile`:
```python
from helpers import export_optimum_no_token_type
from optimum.exporters.onnx.model_configs import BertOnnxConfig
from transformers import AutoModelForMaskedLM
onnx_path = export_optimum_no_token_type(MODEL_ID, task="fill-mask",
    model_class=AutoModelForMaskedLM, onnx_config_class=BertOnnxConfig)
```

**For CausalLM**, replace with backbone extraction:
```python
from helpers import export_torch
# Define GPT2Encoder class (see Approach C above), then:
raw = AutoModel.from_pretrained(MODEL_ID); raw.eval()
onnx_path = export_torch(MODEL_ID, GPT2Encoder(raw),
    sample_inputs=(torch.randint(0, 50000, (1, 128)), torch.ones(1, 128, dtype=torch.long)),
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                  "last_hidden_state": {0: "batch"}})
```

## Known failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Compile OK, profile crashes with `QNN_COMMON_ERROR_GENERAL` | `token_type_ids` in ONNX | Re-export with Approach B |
| `unordered_map::at` during export | Rotary embeddings (Llama, Pythia) | No fix |
| `number of output names exceeded` | Wrong model class | Use task-specific class |
| `Provided input_shapes don't match` | Wrong dynamic dim resolution | Fix shapes in input_spec |
| `QAIRT converter failed` | Audio 1D convs on QNN DLC | Only works on TFLite |
| `Internal compiler error` on TFLite | DeBERTa-v3, SwinV2 | Only works on QNN DLC + ONNX |
| `aten::fft_fftn not supported` | FFT ops not in ONNX | No fix |
| `dspservice just died!` | NPU crash — token_type_ids or KV cache | Remove input or extract backbone |
| Float16 input rejected | Hub doesn't support fp16 | No fix |
| `missing external weights` | ONNX with `.onnx_data` | Skip or bundle weights |

## helpers.py

This is the **complete, self-contained source** for `helpers.py`. Write it verbatim
to the same directory as your per-model scripts. No other files or dependencies beyond
this guide are needed — everything required to run the pipeline is defined here and in
the per-model script template above.

```python
"""Shared utilities for per-model export scripts."""

import os
import shutil
from collections import OrderedDict
from pathlib import Path

import onnx
import torch

DTYPE_MAP = {
    1: "float32",
    2: "uint8",
    3: "int8",
    5: "int16",
    6: "int32",
    7: "int64",
    10: "float16",
}

EXPORT_DIR = os.path.join(os.environ.get("TMPDIR", "/tmp"), "claude", "onnx_export")


def safe_name(model_id):
    return model_id.replace("/", "__")


# ── Export helpers ────────────────────────────────────────


def export_optimum(model_id, task="auto"):
    """Export using optimum main_export (Approach A).

    Returns path to the exported ONNX file.
    """
    from optimum.exporters.onnx import main_export

    output_dir = os.path.join(EXPORT_DIR, safe_name(model_id))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    main_export(model_id, output=output_dir, task=task, no_post_process=True)

    onnx_path = os.path.join(output_dir, "model.onnx")
    print(f"  Exported: {onnx_path} ({os.path.getsize(onnx_path)/(1024*1024):.1f}MB)")
    return onnx_path


def export_optimum_no_token_type(model_id, task, model_class, onnx_config_class):
    """Export using optimum WITHOUT token_type_ids (Approach B).

    Args:
        model_id: HuggingFace model ID
        task: optimum task string (e.g. "fill-mask", "text-classification")
        model_class: transformers model class (e.g. AutoModelForMaskedLM)
        onnx_config_class: optimum config class (e.g. BertOnnxConfig)

    Returns path to the exported ONNX file.
    """
    from optimum.exporters.onnx.convert import export_pytorch
    from transformers import AutoConfig

    NoTTConfig = make_no_token_type_config(onnx_config_class)

    config = AutoConfig.from_pretrained(model_id)
    custom_config = NoTTConfig(config, task=task)

    model = model_class.from_pretrained(model_id)
    model.eval()

    output_dir = os.path.join(EXPORT_DIR, safe_name(model_id))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    onnx_path = os.path.join(output_dir, "model.onnx")
    export_pytorch(
        model=model, config=custom_config, opset=17, output=Path(onnx_path)
    )

    print(f"  Exported: {onnx_path} ({os.path.getsize(onnx_path)/(1024*1024):.1f}MB)")
    return onnx_path


def export_torch(
    model_id,
    wrapper,
    sample_inputs,
    input_names,
    output_names=None,
    dynamic_axes=None,
    opset=17,
):
    """Export using torch.onnx.export with legacy TorchScript exporter (Approach C).

    Returns path to the exported ONNX file.
    """
    output_dir = os.path.join(EXPORT_DIR, safe_name(model_id))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    onnx_path = os.path.join(output_dir, "model.onnx")

    if output_names is None:
        output_names = ["output"]
    if dynamic_axes is None:
        dynamic_axes = {n: {0: "batch"} for n in input_names + output_names}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            sample_inputs,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    print(f"  Exported: {onnx_path} ({os.path.getsize(onnx_path)/(1024*1024):.1f}MB)")
    return onnx_path


def export_direct_onnx(model_id, source_file):
    """Download a pre-existing ONNX file from the HF repo (Approach D).

    Returns path to the saved ONNX file.
    """
    from huggingface_hub import hf_hub_download

    output_dir = os.path.join(EXPORT_DIR, safe_name(model_id))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    onnx_path = os.path.join(output_dir, "model.onnx")
    local = hf_hub_download(repo_id=model_id, filename=source_file)
    shutil.copy2(local, onnx_path)

    print(f"  Downloaded: {onnx_path} ({os.path.getsize(onnx_path)/(1024*1024):.1f}MB)")
    return onnx_path


# ── ONNX input spec ──────────────────────────────────────


def read_onnx_input_spec(onnx_path):
    """Read input spec from an ONNX file, resolving dynamic dims to defaults.

    Returns (input_spec, has_int64) where input_spec is
    {name: (shape_tuple, dtype_string)}, the format qai_hub expects.

    Dynamic dimension defaults: batch->1, sequence->128, height/width->224, channel->3.
    """
    model = onnx.load(onnx_path)
    input_spec = {}
    has_int64 = False
    for inp in model.graph.input:
        dtype = DTYPE_MAP.get(inp.type.tensor_type.elem_type, "float32")
        if dtype == "int64":
            has_int64 = True
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                p = dim.dim_param.lower()
                if "batch" in p:
                    shape.append(1)
                elif "seq" in p or "length" in p:
                    shape.append(128)
                elif "height" in p or "width" in p:
                    shape.append(224)
                elif "channel" in p:
                    shape.append(3)
                else:
                    shape.append(1)
            else:
                shape.append(1)
        input_spec[inp.name] = (tuple(shape), dtype)
    return input_spec, has_int64


# ── Profile results ───────────────────────────────────────


def extract_profile_results(profile_job):
    """Extract latency (ms) and NPU% from a completed profile job.

    The AI Hub API returns latency in MICROSECONDS. This function converts to ms.
    For QNN DLC, execution_time is 0 for all ops, so we fall back to counting
    ops by compute_unit.

    Returns (latency_ms, npu_pct) or (None, None) if profile failed.
    """
    status = profile_job.get_status()
    if not status.success:
        return None, None

    perf = profile_job.download_profile()
    latency_us = perf["execution_summary"]["estimated_inference_time"]
    latency_ms = latency_us / 1000.0

    details = perf["execution_detail"]
    time_by_unit = {}
    for op in details:
        unit = op.get("compute_unit", "Unknown")
        time_by_unit[unit] = time_by_unit.get(unit, 0) + op.get("execution_time", 0)
    total = sum(time_by_unit.values())
    if total > 0:
        npu_pct = time_by_unit.get("NPU", 0) / total * 100
    else:
        npu_ops = sum(1 for op in details if op.get("compute_unit") == "NPU")
        npu_pct = npu_ops / len(details) * 100 if details else 0

    return round(latency_ms, 1), round(npu_pct, 1)


# ── Token type IDs fix ────────────────────────────────────


def make_no_token_type_config(base_onnx_config_cls):
    """Create an OnnxConfig subclass that excludes token_type_ids.

    BERT-family models exported with token_type_ids compile fine but crash on
    the Snapdragon NPU at runtime (QNN HTP error 1100). Removing this optional
    input produces a functionally identical model that runs correctly.

    Usage:
        from optimum.exporters.onnx.model_configs import BertOnnxConfig
        NoTTConfig = make_no_token_type_config(BertOnnxConfig)
        config = NoTTConfig(model_config, task="fill-mask")
    """

    class NoTokenTypeConfig(base_onnx_config_cls):
        @property
        def inputs(self):
            parent = super().inputs
            return OrderedDict(
                (k, v) for k, v in parent.items() if k != "token_type_ids"
            )

        def generate_dummy_inputs(self, framework="pt", **kwargs):
            dummy = super().generate_dummy_inputs(framework=framework, **kwargs)
            dummy.pop("token_type_ids", None)
            return dummy

    return NoTokenTypeConfig


# ── Output wrapper ────────────────────────────────────────


class OutputWrapper(torch.nn.Module):
    """Wraps a HuggingFace model to return a single tensor.

    Most HF models return ModelOutput dicts, which torch.onnx.export can't handle.
    This wrapper extracts the primary output tensor (logits or last_hidden_state).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        if isinstance(out, dict):
            for key in ["logits", "last_hidden_state", "prediction_logits"]:
                if key in out:
                    return out[key]
            return list(out.values())[0]
        if hasattr(out, "logits"):
            return out.logits
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        if isinstance(out, (tuple, list)):
            return out[0]
        return out
```
