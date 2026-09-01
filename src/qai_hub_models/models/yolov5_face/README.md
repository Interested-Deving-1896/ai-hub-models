> [!WARNING]
> This model is not published. Use with caution; it may not meet performance/accuracy standards and may not support some runtimes or chipsets/devices. We do not provide support for unpublished models. If this model was previously published, use earlier releases.

# [YoloV5-Face: Real-time face detection with 5-point facial landmark estimation on mobile and edge devices](https://aihub.qualcomm.com/models/yolov5_face)

YoloV5-Face-Nano is a lightweight single-class face detector based on the YoloV5 backbone (StemBlock + ShuffleV2Block). Each detected face is accompanied by 5 facial landmark predictions (left eye, right eye, nose, left mouth corner, right mouth corner). Trained on the WiderFace dataset.

This is based on the implementation of YoloV5-Face found [here](https://github.com/deepcam-cn/yolov5-face).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/yolov5_face).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect YoloV5-Face:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model's metadata
qai-hub-models info YoloV5-Face

# Print performance and accuracy metrics
qai-hub-models perf YoloV5-Face
qai-hub-models numerics YoloV5-Face

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
qai-hub-models install yolov5_face
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
qai-hub-models demo yolov5_face
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
qai-hub-models export yolov5_face
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of YoloV5-Face can be found
  [here](https://github.com/deepcam-cn/yolov5-face/blob/master/LICENSE).

## References
* [YOLO5Face: Why Reinventing a Face Detector](https://arxiv.org/abs/2105.12931)
* [Source Model Implementation](https://github.com/deepcam-cn/yolov5-face)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
