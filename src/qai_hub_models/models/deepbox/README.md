# [3D-Deep-BOX: Real-time 3D object detection](https://aihub.qualcomm.com/models/deepbox)

3D Deep Box is a machine learning model that predicts 3D bounding boxes and classes of objects in an image.

This is based on the implementation of 3D-Deep-BOX found [here](https://github.com/skhadem/3D-BoundingBox/).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/deepbox).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download 3D-Deep-BOX:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info 3D-Deep-BOX

# Print performance and accuracy metrics
qai-hub-models perf 3D-Deep-BOX
qai-hub-models numerics 3D-Deep-BOX

# Download a ready-to-deploy asset
qai-hub-models fetch 3D-Deep-BOX --runtime tflite --precision float
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
qai-hub-models install deepbox
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
qai-hub-models demo deepbox
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct.
Use the following command to export the model:
```bash
qai-hub-models export deepbox
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of 3D-Deep-BOX can be found
  [here](https://github.com/skhadem/3D-BoundingBox/blob/master/LICENSE).

## References
* [3D Bounding Box Estimation Using Deep Learning and Geometry](https://arxiv.org/abs/1612.00496)
* [Source Model Implementation](https://github.com/skhadem/3D-BoundingBox/)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
