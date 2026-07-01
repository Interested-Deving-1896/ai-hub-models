# [Qwen3-0.6B: State-of-the-art large language model useful on a variety of language understanding and generation tasks](https://aihub.qualcomm.com/models/qwen3_0_6b)

The Qwen3-0.6B is a state-of-the-art multilingual language model with 0.6 billion parameters, excelling in language understanding, generation, coding, and mathematics.

This is based on the implementation of Qwen3-0.6B found [here](https://github.com/QwenLM/Qwen3).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/qwen3_0_6b).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download Qwen3-0.6B:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info Qwen3-0.6B

# Print performance and accuracy metrics
qai-hub-models perf Qwen3-0.6B
qai-hub-models numerics Qwen3-0.6B

# Download a ready-to-deploy asset
qai-hub-models fetch Qwen3-0.6B --runtime geniex_llamacpp --precision q4_0
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## Deploying Qwen3-0.6B on-device

Follow the [GenieX quickstart](https://geniex.aihub.qualcomm.com/en/get-started/quickstart) to install GenieX and deploy the model on a target device.


## License
* The license for the original implementation of Qwen3-0.6B can be found
  [here](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/LICENSE).

## References
* [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
* [Source Model Implementation](https://github.com/QwenLM/Qwen3)

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
