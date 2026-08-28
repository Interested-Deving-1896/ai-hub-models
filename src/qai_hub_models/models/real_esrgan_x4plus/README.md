# [Real-ESRGAN-x4plus: Upscale images and remove image noise](https://aihub.qualcomm.com/models/real_esrgan_x4plus)

Real-ESRGAN is a machine learning model that upscales an image with minimal loss in quality. The implementation is a derivative of the Real-ESRGAN-x4plus architecture, a larger and more powerful version compared to the Real-ESRGAN-general-x4v3 architecture.

This is based on the implementation of Real-ESRGAN-x4plus found [here](https://github.com/xinntao/Real-ESRGAN).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/real_esrgan_x4plus).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download Real-ESRGAN-x4plus:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info Real-ESRGAN-x4plus

# Print performance and accuracy metrics
qai-hub-models perf Real-ESRGAN-x4plus
qai-hub-models numerics Real-ESRGAN-x4plus

# Download a ready-to-deploy asset
qai-hub-models fetch Real-ESRGAN-x4plus --runtime tflite --precision float
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
qai-hub-models install real_esrgan_x4plus
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
qai-hub-models demo real_esrgan_x4plus
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

By default, the demo will run locally in PyTorch. Pass `--eval-mode on-device` to run the model on a cloud-hosted target device.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct.
Use the following command to export the model:
```bash
qai-hub-models export real_esrgan_x4plus --target-runtime tflite --precision float
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of Real-ESRGAN-x4plus can be found
  [here](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE).

## References
* [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)
* [Source Model Implementation](https://github.com/xinntao/Real-ESRGAN)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
