---
language:
- en
license: apache-2.0
base_model: mssfj/qwen25-0.5b-finemath-4plus
datasets:
- mssfj/qwen25-openmathinstruct2-sft-50k
pipeline_tag: text-generation
library_name: transformers
tags:
- qwen2
- math
- sft
- litgpt
---

# Qwen2.5 0.5B FineMath OpenMath SFT

This is a full-parameter supervised fine-tune of
[mssfj/qwen25-0.5b-finemath-4plus](https://huggingface.co/mssfj/qwen25-0.5b-finemath-4plus).
It provides a mathematical instruction-following baseline for the TDT comparison
in [lowbit-math-reasoning](https://github.com/mssfj/lowbit-math-reasoning).
Training uses the repository's local `mylitgpt/litgpt/finetune/full.py` implementation.

The base model was initialized through FineWeb-Edu pretraining and continued
pretraining on FineMath 4+. It uses the Qwen2.5-0.5B architecture and tokenizer,
with separate input embeddings and output-head weights (630,167,424 parameters).
The export preserves `tie_word_embeddings: false`.

## Training data and format

The dataset is [mssfj/qwen25-openmathinstruct2-sft-50k](https://huggingface.co/datasets/mssfj/qwen25-openmathinstruct2-sft-50k),
pinned to revision `95cdc178b2dda77628f98a801b99f4d549db403a`.
It contains 49,000 training examples and 1,000 separate validation examples,
selected from `nvidia/OpenMathInstruct-2` with a 70% MATH / 30% GSM8K source mix.
The dataset card documents exact, template, lexical, and semantic filtering
against the GSM8K test and MATH500 test problems. These checks do not guarantee
that all semantic overlap or earlier exposure in the base model has been removed.

For SFT, the original solutions are retained and an explicit final line
`\boxed{ANSWER}` is added when necessary, using `expected_answer` as the answer.
Prompts match the normal prompts in `eval/gsm8k-eval.py` and `eval/math500-eval.py`:

```text
System: You are a careful mathematical problem solver.
User: Solve the following math problem step by step.
The last line of your response should be in the format: \boxed{ANSWER}
Problem: {question}
```

Qwen ChatML is used. Only assistant response tokens contribute to the loss,
including the `<|im_end|>` assistant-turn terminator. Every example fits within
2,048 tokens without truncation. Both benchmark answer extractors successfully
extract the final boxed answer from every formatted training/validation solution.

## Training configuration

| Setting | Value |
| --- | --- |
| Method | Full SFT, all 630,167,424 parameters |
| Epochs | 1 |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1 |
| Peak learning rate | 0.00002 |
| Schedule | 50 optimizer steps of warmup, then cosine decay |
| Global / micro batch | 32 / 1 |
| Precision | BF16 mixed precision, FP32 master weights |
| Maximum sequence length | 2,048 |
| Seed | 42 |
| Memory optimizations | Transformer activation checkpointing and chunked output-head loss |
| Hardware | One NVIDIA GeForce RTX 5060 Ti, 16 GB |

Native LitGPT checkpoints retain the trained master weights. The Transformers
export stores BF16 safetensors. `generation_config.json` stops on `<|im_end|>`
or `<|endoftext|>`. Training details, the final validation loss, and small generation
checks are recorded in `training_report.json`.
Validation loss is computed on the SFT validation split, not on either benchmark.
Benchmark results and evaluation conditions are reported below.

## Evaluation

Use the existing repository evaluation scripts with the local Transformers export:

```bash
python eval/gsm8k-eval.py \
  --model-name out/finemath-openmath-sft/huggingface \
  --max-samples 0 --max-tokens 2048 --batch-size 8 --wandb-mode disabled \
  --output-path eval/outputs/finemath-openmath-sft-gsm8k.jsonl

python eval/math500-eval.py \
  --model-name out/finemath-openmath-sft/huggingface \
  --max-samples 0 --max-tokens 4096 --batch-size 8 --quantization none --load-format none \
  --wandb-mode disabled \
  --output-path eval/outputs/finemath-openmath-sft-math500.jsonl
```

The scripts require their vLLM evaluation environment. Use the same generation
and retry settings across models for the TDT comparison. SFT targets the normal
step-by-step prompt; MATH500's optional final-answer-only retry is an evaluation
procedure rather than a separate SFT training target.

## Benchmark results

Evaluated on October 6, 2026 with the repository's existing
`eval/gsm8k-eval.py` and `eval/math500-eval.py` scripts and answer verifiers.
The evaluated checkpoint is the BF16 Transformers export at revision
`44cb74b31f3093321e45c471b3b40cb53e703d4c` of
`mssfj/qwen25-0.5b-finemath-4plus-openmath-sft`.
These results describe that export; the FP32 native LitGPT checkpoint was not
independently benchmarked.

| Benchmark | Test questions | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| GSM8K (`openai/gsm8k`, `main`, test) | 1,319 | 161 | 12.21% |
| MATH500 (`HuggingFaceH4/MATH-500`, test), including the evaluator's retry | 500 | 24 | 4.80% |

Conditions: zero-shot, step-by-step boxed-answer prompts, greedy decoding
(temperature 0, top-p 1), BF16, no quantization or LoRA, and batch size 8.
GSM8K allows 2,048 generated tokens with a 4,096-token context limit;
MATH500 allows 4,096 generated tokens with an 8,192-token context limit.
GSM8K uses eager execution; MATH500 uses the script's default graph execution.
The environment was Python 3.11.17, vLLM 0.30.0, Transformers 5.18.0,
PyTorch 2.13.0+cu130, datasets 5.1.0, and SymPy 1.14.0 on one
NVIDIA GeForce RTX 5060 Ti (16 GB).

MATH500 performs one final-answer-only retry when the first response has no
extractable final answer. There were **105 retried questions**; the table reports
the evaluator's final score after that retry. It is not a single-generation
pass@1 measurement. GSM8K does not use that retry. Both scores use this
repository's exact/numeric/symbolic answer verification rules.

### GSM8K answer-extraction audit

All 1,319 questions and gold answers were checked against the test split.
For all **1,231 responses with a complete boxed answer**, the last complete box
in the full response matched the saved extracted answer. An independent numeric
comparison, including thousands separators and simple fractions, found the same
**161 correct answers**, with no additional correct boxed answers recovered.
Of the **88 responses without a boxed answer**, **86 reached the 2,048-token
output limit**, commonly with repeated reasoning. A loose trailing-number
fallback found only two numerical matches in unfinished responses; these are
not counted as validated final answers.

For example, the first GSM8K problem has gold answer `18`, while the model
explicitly ends with `\boxed{42}`; the evaluator correctly extracts `42`.
Low SFT validation loss therefore does not imply high freely generated
mathematical accuracy.

Detailed artifacts:
[GSM8K predictions](https://huggingface.co/mssfj/qwen25-0.5b-finemath-4plus-openmath-sft/blob/main/benchmark_results/gsm8k.jsonl),
[GSM8K summary](https://huggingface.co/mssfj/qwen25-0.5b-finemath-4plus-openmath-sft/blob/main/benchmark_results/gsm8k.summary.json),
[MATH500 predictions](https://huggingface.co/mssfj/qwen25-0.5b-finemath-4plus-openmath-sft/blob/main/benchmark_results/math500.jsonl),
[MATH500 summary](https://huggingface.co/mssfj/qwen25-0.5b-finemath-4plus-openmath-sft/blob/main/benchmark_results/math500.summary.json),
[environment and model hash](https://huggingface.co/mssfj/qwen25-0.5b-finemath-4plus-openmath-sft/blob/main/benchmark_results/metadata.json), and
[GSM8K extraction audit](https://huggingface.co/mssfj/qwen25-0.5b-finemath-4plus-openmath-sft/blob/main/benchmark_results/gsm8k-extraction-audit.json).

## Limitations and licensing

A boxed answer is a trained output convention, not a guarantee of mathematical
correctness or formatting at inference. This model is intended for controlled
mathematical experiments, rather than general-purpose assistant use.

The base model is distributed under Apache-2.0. The OpenMathInstruct-2-derived
training dataset retains its CC BY 4.0 license and attribution requirements;
consult its dataset card for source attribution and filtering provenance.
