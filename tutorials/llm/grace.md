# Grace

Grace (Grading Response Accuracy Evaluation) is a benchmark for evaluating
LLMs. It consists of evaluating responses to prompts that are graded by a larger
LLM grader. The AI Hub Models project uses this benchmark to measure the impact
of quantization on LLMs.


## Motivation

Before this evaluation score, we primarily used PPL (Perplexity) on English
WikiText, as well as a subset of [MMLU](https://en.wikipedia.org/wiki/MMLU)
(multiple-choice questions). However, these have a serious blindspot:

* PPL only looks at the probability of the ground truth next token (1 token)
* MMLU only looks at valid responses (~4 tokens)

That means you can shuffle and re-weight all other logits without affecting
the benchmark score. This is a weakness that we discovered is particularly
harmful when measuring the effects of quantization. Quantization when done
unsuccessfully can sometimes boost tokens that should stay low probability.
Here's an example of a failure mode:

```
Question: What is gravity?
Response: Gravity is a fundamental force of nature that pulls 物体 ...
```

Here, after "pulls" it switches language. The first Chinese token should not
have had enough probability to be sampled (yet it did even with nucleus
sampling top-p 0.9).

PPL and MMLU both failed to catch this and other similar undesirable outcomes,
since they are blind to what is happening in the tail of the distribution.

We also explored suggestions from [Accuracy is Not All You
Need](https://arxiv.org/abs/2407.09141) to use KL divergence and MMLU "flips".
KL divergence does look at the entire probability distribution, which is
promising. However, experimentally we found that these metrics did not help us
catch these issues either.

What we really need is a human that rates responses based on their severity. We
use an LLM grader as a proxy for this. A comparable benchmark is
[MT-Bench](https://arxiv.org/abs/2306.05685). We chose to maintain our own
metric to allow the grading rubric and prompts to better measure
quantization-related degradations.

## Key requirements

* **Small**: Running this on device across many models weekly is expensive, so
  the dataset has to be representative, but modest in size.
* **Absolute**: We would like to be able to score any single LLM (quantized
  or not) with an absolute score, as opposed to a distance between two LLMs.
* **Quantization**: Even though it can be used as a general score, some
  emphasis in the grading rubric and prompt selection is toward giving signal
  for quantization-related decay.

## Grace2

The current version is Grace2:

* Language: English
* Samples: [100 prompts](../../src/qai_hub_models/models/_shared/llm/grader/grace2.jsonl) (across 10 categories)
* Grader: Qwen3.6-35B-A3B

Please find the grading rubric in
[grader.py](../../src/qai_hub_models/models/_shared/llm/grader/grader.py). We
publish results to [AI Hub Models](https://aihub.qualcomm.com/models). These
numbers can also be found in `numerics.yaml` files in the
[ai-hub-models](https://github.com/qualcomm/ai-hub-models) repository.
