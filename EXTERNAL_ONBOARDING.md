# Onboarding an External Recipe

An *external recipe* is a self-contained model folder — `model.py`, `app.py`, `demo.py`,
`test.py`, `manifest.yaml` — that lives anywhere on disk. It is not merged into this repo
and does not appear in the Qualcomm catalog. The `qai-hub-models` CLI drives it directly,
and you publish it under your own Hugging Face namespace.

```
clone  →  install skills  →  /onboard  →  /validate-on-device  →  /add-quantization  →  upload-to-hf
                                                                                            ↓
                                                                        others: qai-hub-models register
```

> **LLMs aren't supported yet** — contact the AI Hub Models team for those. Other
> generative models (diffusion, TTS) can work, but expect more iteration than vision.

## Prerequisites

- Python 3.10–3.13 (x86-64 only on Windows)
- `pip install qai_hub_models`
- A token from [AI Hub Workbench](https://workbench.aihub.qualcomm.com/account/), then
  `qai-hub configure --api_token <TOKEN>`
- [Claude Code](https://claude.com/claude-code)

Workbench access is needed from Step 3 on; Steps 1–2 are local.

## 1. Clone

```shell
git clone https://github.com/qualcomm/ai-hub-models.git
cd ai-hub-models
```

The clone carries the authoring skills (`plugin/`) and the ~200 existing recipes under
`src/qai_hub_models/models/` that the skills copy patterns from. Your recipe does *not*
go inside the clone — it is authored in whatever directory you launch Claude Code from.

## 2. Install the skills

```shell
claude plugin marketplace add $(pwd)          # from the repo root
claude plugin install ai-hub-models@ai-hub-models
```

Or interactively with `/plugin`. This adds `/ai-hub-models:onboard`,
`:validate-on-device`, and `:add-quantization`. (`:onboard-internal` is for Qualcomm
engineers adding catalog models — ignore it.) After pulling new commits, run
`claude plugin update ai-hub-models`.

Then start a session where you want the recipe to live:

```shell
mkdir -p ~/recipes
cd ~/recipes
claude
```

## 3. `/ai-hub-models:onboard`

> For Qualcomm engineers trying to contribute directly to this repo, use `/ai-hub-models:onboard-internal` instead.
> The usage is the same as `/ai-hub-models:onboard` but performs a few additions needed to make it follow repo rules.

Give it **the model** (HF repo, GitHub URL, PyPI package, or a name) and **the dataset +
metric** up front:

```
/ai-hub-models:onboard https://huggingface.co/Intel/dpt-hybrid-midas, eval on NYUv2 with RMSE
```

It will ask for either if you leave it out, so supplying both saves a round trip. License,
task type, input spec, and which sibling recipe to mirror are derived and reported back —
don't bother specifying those.

Existing datasets live in `src/qai_hub_models/datasets/` and shared evaluators in
`src/qai_hub_models/models/templates/<task>/` — worth a look before picking a metric.

Produces `<model_id>/` in your CWD: `model.py`, `app.py`, `demo.py`, `test.py`,
`manifest.yaml`, `requirements.txt`, and `external_repos/` if upstream source is needed.

The gate is `qai-hub-models validate <model_id>` printing `N passed, 0 failed` (`WARN`
rows are informational). All local — no device jobs. Ships `supported_precisions:
[float]`.

## 4. `/ai-hub-models:validate-on-device`

Pass the recipe folder the previous step produced:

```
/ai-hub-models:validate-on-device dpt_hybrid_midas
```

First step that hits AI Hub Workbench. Compiles and profiles on a hosted Snapdragon
device across **both** `tflite` and `qnn_dlc` in parallel, then checks numerics — dataset
accuracy if an evaluator is wired, PSNR against torch otherwise. Failing
`(precision, runtime)` pairs land in `disabled_paths` in `manifest.yaml`.

It reports failures rather than chasing them; ask if you want one debugged.

By hand, this is:

```shell
qai-hub-models export <model_id> --target-runtime tflite --precision float \
    --device "Samsung Galaxy S25 (Family)"
```

### 4b. `/ai-hub-models:add-quantization` (optional)

```
/ai-hub-models:add-quantization dpt_hybrid_midas
```

Tries `w8a8`, falls back to `w8a16`, or concludes cleanly that the model won't quantize —
checking both runtimes and disabling the pairs that miss. **Requires** a dataset +
evaluator. Step 4 first is recommended (per-runtime float baseline) but not required.

## 5. Publish: `upload-to-hf`

No skill — one command. First get a **write** token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and either
`hf auth login` (cached) or `export HF_TOKEN=hf_xxx`.

```shell
qai-hub-models upload-to-hf <model_id> --dry-run   # preview, needs no token
qai-hub-models upload-to-hf <model_id>
```

- Publishes to **your own namespace**, `<your-hf-username>/<folder-name>`; `--repo-id`
  overrides. Hugging Face enforces the namespace, so ownership is real and no org
  membership is needed.
- Publishes **source plus a generated model card**, so anyone can register, install, and
  re-export it. Build output and the clones under `external_repos/` are excluded — the
  recipe re-fetches those from the manifest.
- **Public by default**; `--private` lets you review the card first.
- Each upload is one commit tagged `v1`, `v2`, … and makes the repo an exact copy of your
  folder, deleting files you removed locally. `--no-tag` skips tagging.
- Updating a repo you did not create is refused.

Published recipes are tagged `qai-hub-models` and listed at
<https://huggingface.co/models?other=qai-hub-models>.

## 6. Pull one down: `register`

```shell
qai-hub-models register ashwmurt/dpt_hybrid_midas
```

Downloads into `~/.qaihm/cli/recipes/<name>/` and registers a short name usable anywhere
a built-in model id works. The default name is the repo name lowercased with `-`/`.`
folded to `_` (`dpt-hybrid-midas` → `dpt_hybrid_midas`).

```shell
qai-hub-models install  dpt_hybrid_midas
qai-hub-models demo     dpt_hybrid_midas
qai-hub-models export   dpt_hybrid_midas --target-runtime qnn_dlc --precision float
qai-hub-models evaluate dpt_hybrid_midas

qai-hub-models register ashwmurt/dpt_hybrid_midas --version v2   # pin a tag/branch/commit
qai-hub-models register ashwmurt/dpt_hybrid_midas --version v1 --alias dpt_hybrid_midas_v1
qai-hub-models register ~/recipes/my_model                       # local, no network
qai-hub-models list-registered
qai-hub-models unregister dpt_hybrid_midas
```

- **Local folders win** — only an `owner/name` that is *not* a directory is treated as an
  HF repo id. Names colliding with built-in model ids are rejected; use `--alias`.
- The repo must have `manifest.yaml` at its root, or nothing is downloaded.
- Registering an HF repo needs the full `qai_hub_models` package; the lean
  `qai_hub_models_cli` handles local folders only. Private/gated repos work with a token.

## Command reference

| Command | Purpose |
| --- | --- |
| `validate <target>` | Local pass/fail report card. The authoring gate. |
| `install <target>` | Walk the dependency graph, pip-install each node once. |
| `generate-files <target>` | Regenerate `README.md` / `external_repos/__init__.py` after editing the manifest. |
| `export <target>` | Compile + profile + inference on a hosted device; download the asset. |
| `demo <target>` | End-to-end demo (`--eval-mode fp` or `on-device`). |
| `evaluate <target>` | Dataset accuracy evaluation. |
| `upload-to-hf <target>` | Publish to your HF namespace. |
| `register` / `list-registered` / `unregister` | Manage recipe names. |

All prefixed `qai-hub-models`. `<target>` is a recipe folder, a registered name, or a
built-in model id — except `upload-to-hf`, where it is always read as a folder. Add
`--help` to any of them.
