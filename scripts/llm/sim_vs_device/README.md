<!--
Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
SPDX-License-Identifier: BSD-3-Clause
-->

# sim-vs-device: localizing quantized-LLM accuracy divergence on device

A debugging harness for the case where a quantized split LLM **matches its
AIMET-ONNX QuantSim ("sim") reference but diverges once compiled and run on
device** (Snapdragon HTP). It pinpoints *which part, which layer, and which
op* is responsible, without any training or full-model re-quantization — using
only inference jobs against the already-exported ONNX and encodings.

It is built for the shared PreSplit/Part LLM export scheme in
`models/templates/llm` (Qwen3, Llama-family, etc.). The tap discovery is
**pure-topology** and works for any standard pre-norm transformer; the
model-specific pieces are small, documented contracts (see
[Extending](#extending-to-a-new-model)).

---

## Why this exists

Quantized LLMs are split into several ONNX graphs ("Parts") for on-device
deployment, and each Part is compiled separately. When on-device accuracy drops
but the QuantSim reference looks fine, the gap is almost always a **structural
sim/target divergence** localized to a few ops — an op the HTP computes
differently than the sim models it (fusion, a dtype/accumulator effect, a
calibration artifact that only bites at real inference). A handful of such ops
in one of ~30 layers can wreck generation while every aggregate metric looks
plausible.

The harness turns "accuracy is off somewhere" into a single number per layer per
op, by comparing the **sim (golden)** value of a tensor against the **device**
value of the *same tensor*, with golden inputs injected at every boundary so
each measurement is isolated from upstream error.

---

## The two phases

The driver runs two stages of increasing resolution. (Historically these were
"Phase A / Phase B"; the driver still prints those labels, but the modules are
named for what they do.)

### Phase A — `tap_per_part.py` (coarse localization, no surgery)

For each transformer Part, feed it the **golden QuantSim inputs** captured at
its boundary, run one on-device inference, and compare the Part's **boundary
outputs** — the residual/hidden-state stream, plus the per-layer
`past_key_*`/`past_value_*` KV tensors that are *already* graph outputs (so no
ONNX surgery is needed). Metrics: SQNR (dB), cosine, max-abs-error.

This produces a **verdict**:

| verdict | meaning |
|---|---|
| `localized`   | one Part's hidden output is an outlier — a bug in that ~N-layer band |
| `shared_op`   | all Parts uniformly depressed — an op repeated every decoder block |
| `kv_divergence` | the KV cache diverges while the residual survives — prime suspect for *generation* garbage (bad KV is re-read every decode step) |
| `clean`       | all Parts track device — the divergence is elsewhere (decode loop, sampling, or the eval harness) |

The KV check is evaluated **first**, because a KV divergence corrupts generation
even when the single-pass prefill residual looks acceptable on SQNR.

### Phase B — `tap_per_layer.py` (fine localization, per-layer sub-graphs)

Given the Part that Phase A flagged (or every transformer Part with
`--phase-b all`), split that Part into **one sub-graph per decoder layer** using
the production split utility (cut at residual-add boundaries). Each sub-graph is
fed the **golden previous-layer residual** + shared attention mask / position
ids + this layer's KV (fully isolated inputs), compiled and run independently,
and SQNR is computed at each of that layer's tap points.

Report columns, in block execution order:

```
 layer      o_proj     gate_up        down    residual
```

- `o_proj`   — attention output (attention residual add's branch input)
- `gate_up`  — the gated-MLP product (SwiGLU `Mul`), i.e. the MLP interior
- `down`     — MLP output (MLP residual add's branch input)
- `residual` — the layer output (= previous residual + attn + down)

A crater sweeping in from the right (e.g. `down` + `residual` drop while
`o_proj`/`gate_up` stay healthy) reads as "the divergence enters at the MLP
output and propagates into the layer output." Cells ≥12 dB below the grid median
are flagged with `*`, and a one-line verdict names the offending column + onset
layer.

Because each job outputs only one layer's few small tensors, this avoids the
2 GB flatbuffer output limit that tapping a whole Part at once would hit, and a
compile/runtime failure localizes to a single layer.

---

## How the golden / injection / tap machinery works

1. **Golden capture** (`golden.py`, `harness.py`): the QuantSim split model is
   driven through the shared `HubCompatibleGenerator` on real dataset windows.
   Each Part's `forward` is wrapped to record its inputs and outputs. The
   generator emits inputs in on-device layout (`position_ids_cos/sin`,
   Hub-transposed KV), so captured tensors are drop-in inputs for a device job.
   Only the **last prefill slice** of each window is kept — earlier slices read a
   zero-filled KV cache (warmup transients) that don't reflect real inference.

2. **Injection**: for a device job we feed the *golden* inputs for that boundary
   rather than chaining the device's own (degrading) outputs. Each part/layer is
   thus scored on its own transfer function, immune to upstream accumulation —
   that is what makes localization sharp.

3. **Tap discovery** (`tap_per_layer.discover_taps`): **pure topology**, no
   tensor/module-name matching. It replicates the production splitter's
   residual-add walk (`is_residual_add`/`can_visit`) to find each layer's two
   residual adds, then:
   - `o_proj` / `down` = each add's *branch* (non-skip) input,
   - `gate_up` = back-trace from `down` through shape ops to the gated `Mul`,
   - `residual` = the add output.
   Only tensors that already carry an **INT activation encoding** are tapped
   (the compile uses `--quantize_io`, so an un-encoded or floated output would
   break compilation). AIMET-ONNX keeps quantization in the encodings, not as
   QDQ nodes in the graph, so the op's own output tensor is the encoded one.

4. **Surgery** (`tap_per_layer.build_tapped_bundle`): promotes tap tensors to
   graph outputs (copying weights + encodings, subsetting encodings to the
   surviving tensors, and removing value_info/IO collisions). The LM-head-bearing
   final part is pruned to just its taps so its huge logits output doesn't OOM
   the HTP graph-prepare.

5. **Metrics** (`metrics.py`): SQNR in dB (via `compute_psnr`), cosine
   similarity, max-abs diff. SQNR ≈ 40 dB → ~1% error, 20 dB → ~10%.

---

## CLI usage

The tool lives at `scripts/llm/sim_vs_device/` and uses flat sibling imports
(matching `scripts/compiler_nightly/`), so run it from **inside that directory**.
`qai_hub_models` must be importable (installed / on `PYTHONPATH`).

```bash
cd scripts/llm/sim_vs_device
python run.py \
    --model-id qwen3_1_7b \
    --checkpoint DEFAULT_W4A16 \
    --device "Snapdragon 8 Elite QRD" \
    --num-windows 3
```

Runs Phase A, then auto-runs Phase B on the Part its verdict flags.

### Flags

| flag | default | meaning |
|---|---|---|
| `--model-id` | `qwen3_1_7b` | model package under `qai_hub_models.models` |
| `--checkpoint` | *(required)* | calibrated AIMET checkpoint dir, or `DEFAULT_W4A16` |
| `--device` | *(required)* | AI Hub device name |
| `--precision` | `w4a16` | quantization precision |
| `--num-windows` | `3` | dataset windows to capture (more = tighter SQNR, larger jobs) |
| `--sequence-length` | `2048` | prefill AR bucket |
| `--context-length` | `4096` | KV context length |
| `--dataset` | `wikitext` | input source (see `dataset.py`) |
| `--phase-b` | `auto` | `auto` (flagged Part) · `off` (Phase A only) · `force` (selected Part) · `all` (every transformer Part) |
| `--phase-b-part` | — | override the Part Phase B instruments, e.g. `part3_of_4` |
| `--skip-phase-a` | off | go straight to Phase B (requires `--phase-b all` or `--phase-b-part`) |
| `--model-cache-mode` | `disable` | AI Hub model cache: `disable` = always compile fresh (safe while iterating); `enable` = let the service reuse a compiled model of the same name |
| `--report-json` | `${TMPDIR:-/tmp}/claude/...json` | machine-readable metrics dump (`''` to skip) |

### Common recipes

(All commands below assume you are in `scripts/llm/sim_vs_device/`.)

Full run on a small model — grid for every layer of every Part:

```bash
python run.py \
    --model-id qwen3_0_6b --checkpoint DEFAULT_W4A16 \
    --device "Snapdragon 8 Elite QRD" --num-windows 1 --phase-b all
```

Investigate one Part directly (skip Phase A):

```bash
python run.py \
    --model-id qwen3_1_7b --checkpoint /path/to/ckpt \
    --device "Snapdragon 8 Elite QRD" --num-windows 1 \
    --skip-phase-a --phase-b force --phase-b-part part2_of_4
```

Validate tap discovery **locally, no device jobs** (recommended before spending
compiles, and the fastest way to sanity-check a new model):

```bash
python inspect_taps.py \
    --model-id qwen3_1_7b --checkpoint DEFAULT_W4A16 --part part2_of_4
# per-layer sub-graph split preview:
python inspect_taps.py \
    --model-id qwen3_1_7b --checkpoint DEFAULT_W4A16 --part part2_of_4 --split-layers
```

> **Job size:** the down-proj input (SwiGLU product) and KV tensors dominate
> payload size; `--num-windows` scales it. If an inference job hits the 2 GB
> flatbuffer limit, reduce `--num-windows`. Each window at seq 2048/ctx 4096
> yields one real (last-slice) sample.

---

## Module map

| module | role |
|---|---|
| `run.py`           | CLI driver: build context → Phase A → dispatch Phase B |
| `harness.py`       | shared scaffolding: model resolution, per-part layer distribution, `build_context` (build split model + capture golden) |
| `golden.py`        | per-Part golden I/O capture from the QuantSim split model |
| `dataset.py`       | `DatasetSource` interface + `WikiTextSource` |
| `device.py`        | all AI Hub I/O: compile submit/collect, inference submit/collect, byte-budget chunking, single-pass profiling |
| `metrics.py`       | SQNR/cosine/max-abs, per-part metrics, and the Phase A verdict logic |
| `tap_per_part.py`  | Phase A: per-Part boundary comparison + verdict + report |
| `tap_per_layer.py` | Phase B: pure-topology tap discovery, tap surgery, per-layer split execution + report |
| `layer_split.py`   | split a Part into per-layer sub-graphs and build their golden feeds |
| `inspect_taps.py`  | local (no-jobs) diagnostics: tap coverage, per-layer split preview, raw layer dumps |

Data flow: `harness.build_context` → `golden` capture → `tap_per_part.run`
(Phase A) → verdict → `tap_per_layer.run`/`run_all` (Phase B, via `layer_split`
+ `device`) → reports.

---

## Extending to a new model

The tool targets the shared PreSplit/Part export scheme. A new model works if it
satisfies these **contracts** (all already met by the Qwen3 family):

- The model package (`qai_hub_models.models.<model_id>`) exports a
  **`QuantizedSplitModelWrapper`** class (a `SplitForwardMixin` over the
  quantizable PreSplit) and module-level **`NUM_LAYERS`** / **`NUM_SPLITS`**
  constants.
- The Parts follow the convention: **Part 1 = embedding**, middle parts hold the
  transformer layers, the **last part carries the LM head**; `part_id` is
  1-indexed.
- It is a **standard pre-norm transformer**: two residual adds per decoder block
  (attention add, MLP add). Tap discovery is otherwise name-agnostic.
- Exported via the AIMET-ONNX path (`--quantize_io`, encodings-in-session, no QDQ
  nodes in the graph).

To add a **new input source**, subclass `DatasetSource` in `dataset.py` and
register it in `build_source`.

Per-model tuning: the SQNR/crater thresholds in `metrics.py` and
`tap_per_layer.py` (`good_sqnr_db`, the −12 dB crater margin, the Heisenberg
margin) were calibrated on Qwen3 runs; they are reasonable defaults but may want
adjustment on very different architectures.

---

## Limitations & notes

- **Gated MLP assumption for `gate_up`.** The `gate_up` tap back-traces to a
  SwiGLU/GEGLU `Mul`. On a non-gated MLP (e.g. GPT-style `fc1→act→fc2`) there is
  no such `Mul`; the `gate_up` column is simply omitted (down + residual still
  bracket the MLP). No crash.
- **Two-residual-adds-per-layer assumption.** Inherited from the production
  splitter. Architectures with parallel attention+MLP residuals (GPT-J/NeoX
  style) would mis-count layers.
- **AIMET-ONNX / `--quantize_io` specific.** Tap selection relies on the op's own
  output being the encoded tensor (no QDQ nodes). A different quant backend would
  need the discovery to hop to the Q node.
- **Prefill only.** The harness compares single-pass prefill. A `clean` Phase A
  verdict points you at the decode loop / sampling / eval harness, which this
  tool does not exercise directly.
- **`TODO(unify-with-production)`** in `tap_per_layer.py`: the residual-add walk
  is re-implemented here from `split_onnx_utils.utils.get_split_tensors`'s
  private closures. A future PR should factor that walk into a shared exported
  helper consumed by both the production splitter and this tool, so the tap
  points can never drift from how the graph is actually cut.
