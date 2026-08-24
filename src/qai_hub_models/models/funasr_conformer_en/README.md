# [FunASR-Conformer-EN: English speech recognition with Conformer encoder and CTC decoder](https://aihub.qualcomm.com/models/funasr_conformer_en)

FunASR Conformer-EN is a large-scale English ASR model from Alibaba DAMO Academy, trained on ~50,000 hours of English speech. It uses a 32-block Conformer encoder(512 hidden dim, 16 attention heads) with a CTC head for decoding. The model accepts raw 16kHz audio and outputs transcribed text via CTC greedy decoding.

This is based on the implementation of FunASR-Conformer-EN found [here](https://huggingface.co/funasr/conformer-en).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/funasr_conformer_en).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download FunASR-Conformer-EN:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info FunASR-Conformer-EN

# Print performance and accuracy metrics
qai-hub-models perf FunASR-Conformer-EN
qai-hub-models numerics FunASR-Conformer-EN

# Download a ready-to-deploy asset
qai-hub-models fetch FunASR-Conformer-EN --runtime tflite --precision float
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Setup
### 1. Install System-Level Dependencies
#### Linux
```bash
sudo apt install ffmpeg
```

 #### Windows
```
winget install ffmpeg
```

### 2. Install the package
Install the base package, then use the `qai-hub-models` CLI to install this
recipe's dependencies:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install qai-hub-models
qai-hub-models install funasr_conformer_en
```

### 3. Configure Qualcomm® AI Hub Workbench
Sign-in to [Qualcomm® AI Hub Workbench](https://workbench.aihub.qualcomm.com/) with your
Qualcomm® ID. Once signed in navigate to `Account -> Settings -> API Token`.

With this API token, you can configure your client to run models on the cloud
hosted devices.
```bash
qai-hub configure --api_token API_TOKEN
```
Navigate to [docs](https://workbench.aihub.qualcomm.com/docs/) for more information.

## Run CLI Demo
Run the following simple CLI demo to verify the model is working end to end:

```bash
python -m qai_hub_models.models.funasr_conformer_en.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

By default, the demo will run locally in PyTorch. Pass `--eval-mode on-device` to the demo script to run the model on a cloud-hosted target device.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct.
Use the following command to export the model:
```bash
qai-hub-models export funasr_conformer_en --target-runtime tflite --precision float
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of FunASR-Conformer-EN can be found
  [here](https://github.com/modelscope/FunASR?tab=MIT-1-ov-file).

## References
* [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)
* [Source Model Implementation](https://huggingface.co/funasr/conformer-en)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
