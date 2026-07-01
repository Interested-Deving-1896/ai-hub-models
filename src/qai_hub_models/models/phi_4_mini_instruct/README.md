# [Phi-4-Mini-Instruct: State-of-the-art large language model useful on a variety of language understanding and generation tasks](https://aihub.qualcomm.com/models/phi_4_mini_instruct)

Phi-4-mini-instruct is a lightweight open model built upon synthetic data and filtered publicly available websites - with a focus on high-quality, reasoning dense data.

This is based on the implementation of Phi-4-Mini-Instruct found [here](https://huggingface.co/microsoft/Phi-4-mini-instruct).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/phi_4_mini_instruct).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download Phi-4-Mini-Instruct:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info Phi-4-Mini-Instruct

# Print performance and accuracy metrics
qai-hub-models perf Phi-4-Mini-Instruct
qai-hub-models numerics Phi-4-Mini-Instruct

# Download a ready-to-deploy asset
qai-hub-models fetch Phi-4-Mini-Instruct --runtime geniex_llamacpp --precision q4_0
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Deploying Phi-4-Mini-Instruct on-device

Follow the [GenieX quickstart](https://geniex.aihub.qualcomm.com/en/get-started/quickstart) to install GenieX and deploy the model on a target device.


## License
* The license for the original implementation of Phi-4-Mini-Instruct can be found
  [here](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/LICENSE).

## References
* [Phi-4 Technical Report](https://arxiv.org/abs/2412.08905)
* [Source Model Implementation](https://huggingface.co/microsoft/Phi-4-mini-instruct)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).

## Usage and Limitations

This model may not be used for or in connection with any of the following applications:

- Accessing essential private and public services and benefits;
- Administration of justice and democratic processes;
- Assessing or recognizing the emotional state of a person;
- Biometric and biometrics-based systems, including categorization of persons based on sensitive characteristics;
- Education and vocational training;
- Employment and workers management;
- Exploitation of the vulnerabilities of persons resulting in harmful behavior;
- General purpose social scoring;
- Law enforcement;
- Management and operation of critical infrastructure;
- Migration, asylum and border control management;
- Predictive policing;
- Real-time remote biometric identification in public spaces;
- Recommender systems of social media platforms;
- Scraping of facial images (from the internet or otherwise); and/or
- Subliminal manipulation
