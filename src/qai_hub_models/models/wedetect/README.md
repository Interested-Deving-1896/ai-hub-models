# [WeDetect: Real-time text-conditioned object detection optimized for mobile and edge](https://aihub.qualcomm.com/models/wedetect)

WeDetect is a text-conditioned object detection model that uses a prompt-then-detect strategy. Text class names are reparameterized into the model weights before export, producing a fixed-vocabulary detector that runs efficiently on-device. The model uses LTRB (Left-Top-Right-Bottom) regression with multi-scale feature maps at strides 8, 16, and 32.

This is based on the implementation of WeDetect found [here](https://github.com/WeChatCV/WeDetect).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/wedetect).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect WeDetect:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model's metadata
qai-hub-models info WeDetect

# Print performance and accuracy metrics
qai-hub-models perf WeDetect
qai-hub-models numerics WeDetect

# Pre-exported assets are not available to download for this model due to
# licensing restrictions. Continue to the next section to export it yourself.
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Setup
### 1. Install the package
Install the base package, then use the `qai-hub-models` CLI to install this
recipe's dependencies:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install qai-hub-models
qai-hub-models install wedetect
```

### 2. Configure Qualcomm® AI Hub Workbench
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
python -m qai_hub_models.models.wedetect.demo { --quantize mixed_with_float }
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
qai-hub-models export wedetect --target-runtime tflite --precision float
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of WeDetect can be found
  [here](https://github.com/WeChatCV/WeDetect/blob/main/README.md).

## References
* [WeDetect: Fast Open-Vocabulary Object Detection as Retrieval](https://arxiv.org/abs/2512.12309)
* [Source Model Implementation](https://github.com/WeChatCV/WeDetect)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
