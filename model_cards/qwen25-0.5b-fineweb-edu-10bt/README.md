---
library_name: transformers
pipeline_tag: text-generation
license: apache-2.0
datasets:
- HuggingFaceFW/fineweb-edu
tags:
- litgpt
- qwen2
- pretraining
- tdt
- research
---

# qwen25-0.5b-fineweb-edu-10bt

A baseline model for TDT comparison experiments, pretrained from scratch with LitGPT on FineWeb-Edu sample-10BT.

## Purpose

These models were created as baselines for comparison experiments with TDT. Their weights come from conventional gradient-based autoregressive language model training and provide a reference for comparing TDT training or adaptation methods and low-bit variants. The released checkpoints themselves were not trained with TDT.

The name `qwen25-0.5b` identifies the architecture. These models use the Qwen2.5-0.5B architecture and tokenizer, but their initial training did not start from official Qwen pretrained weights. The FineWeb-Edu stage starts from random initialization; the FineMath variants continue training from that FineWeb-Edu checkpoint.

## Architecture

| Property | Value |
|---|---|
| Architecture | Qwen2.5-0.5B-compatible decoder-only Transformer |
| Training framework | LitGPT included in the code repository, package version 0.5.12 |
| Layers / hidden dimension / FFN dimension | 24 / 896 / 4,864 |
| Attention heads / KV heads | 14 / 2 (GQA) |
| Padded vocabulary size | 151,936 |
| Parameters | 630,167,424 with separate input embeddings and output weights |
| Training sequence length | 2,048 tokens |
| Configured maximum positions / RoPE base | 32,768 / 1,000,000 |
| Weight tying | `train.tie_embeddings: false` |

The configured maximum of 32,768 positions is an architectural setting. Training used sequences of 2,048 tokens; long-context performance has not been established.

## Data preparation and training objective

The same [prepare_fineweb_edu.py](https://github.com/mssfj/lowbit-math-reasoning/blob/main/mylitgpt/litgpt/data/prepare_fineweb_edu.py) script is used for FineWeb-Edu and FineMath.

- Read the `text` column from Parquet files and skip empty text.
- Tokenize with the `Qwen/Qwen2.5-0.5B` tokenizer, without BOS and with EOS at the end of each document.
- Create streaming chunks with `litdata.optimize` and `TokensLoader`.
- Load blocks of 2,049 tokens: use the first 2,048 as inputs and the shifted 2,048 as targets, minimizing next-token cross-entropy.
- Shuffle training data, leave validation data unshuffled, and use seed 42.

## Use in comparison experiments and limitations

These are base models for text completion. The recorded pipeline does not include conversational supervised fine-tuning, instruction tuning, or RLHF. A chat template in the tokenizer does not establish that the model was trained to converse.

The training configurations use `split_names: [train, train]`, so validation reads from the same split as training. Reported validation losses are not independent held-out results. For TDT comparisons, control the initial weights, tokenizer, data splits, token budget, and evaluation conditions, and use a separate evaluation dataset. This card does not report TDT comparison results or mathematics benchmark scores.

Data sources are [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and, for the FineMath variants, [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath). Both source datasets list `odc-by` in their public metadata. Refer to their dataset cards for terms and attribution details.

## Related models

| Repository | Training stage | Format |
|---|---|---|
| [mssfj/qwen25-0.5b-fineweb-edu-10bt](https://huggingface.co/mssfj/qwen25-0.5b-fineweb-edu-10bt) | FineWeb-Edu pretraining | Transformers / safetensors |
| [mssfj/qwen25-0.5b-finemath-4plus](https://huggingface.co/mssfj/qwen25-0.5b-finemath-4plus) | FineWeb-Edu → FineMath-4plus | Transformers / safetensors |
| [mssfj/qwen25-0.5b-finemath-4plus-lit](https://huggingface.co/mssfj/qwen25-0.5b-finemath-4plus-lit) | FineWeb-Edu → FineMath-4plus | LitGPT / model state dict |

Code: [mssfj/lowbit-math-reasoning](https://github.com/mssfj/lowbit-math-reasoning)

## Training data and configuration

Initial weights are randomly initialized. `pretrain.sh` downloads only the tokenizer from the official Qwen model, and `initial_checkpoint_dir` is unset. With `resume: auto`, an existing checkpoint in the output directory is used to resume training.

- Source data: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample-10BT` (`sample/10BT/*.parquet`).
- Pretokenized data: [mssfj/qwen25-0.5b-fineweb-edu-10bt](https://huggingface.co/datasets/mssfj/qwen25-0.5b-fineweb-edu-10bt). This is a dataset repository with the same ID as the model repository.
- Training token budget: 10,000,000,000. The source sample name `10BT` should be distinguished from the actual corpus token count under the Qwen tokenizer.

The table below describes the [current YAML configuration](https://github.com/mssfj/lowbit-math-reasoning/blob/main/mylitgpt/config_hub/pretrain/qwen25-0.5b-fineweb-edu-10bt.yaml). It has not been fully reconciled with the execution logs for the published weights.

| Property | Configuration |
|---|---|
| Precision | bf16-mixed |
| Global batch / micro batch | 512 / 8 |
| Optimizer | AdamW |
| Learning rate / minimum LR | 4e-4 / 4e-5, cosine decay after warmup |
| Warmup | 1,000 optimizer steps |
| Weight decay / betas | 0.1 / (0.9, 0.95) |
| Gradient clipping | Maximum norm 1.0 |
| Checkpoint / validation interval | 2,000 / 1,000 optimizer steps |
| Validation | 100 batches; end-of-training validation disabled |

The script exports the end-of-training `final` checkpoint. This is not an automatically selected minimum-validation-loss checkpoint. Losses and benchmark results for this FineWeb-Edu model have not been verified and are not reported here.

## Reproduction code

[pretrain.sh](https://github.com/mssfj/lowbit-math-reasoning/blob/main/mylitgpt/pretrain.sh) contains data download, training, conversion, and upload steps. Some preprocessing steps are commented out. The training command is:

```bash
cd mylitgpt
litgpt pretrain \
  --config config_hub/pretrain/qwen25-0.5b-fineweb-edu-10bt.yaml \
  --out_dir out/pretrain/qwen25-0.5b-fineweb-edu-10bt
```

## Checkpoint format and usage

This repository uses Hugging Face Transformers format with `model.safetensors`. The export pipeline removes optimizer state from the training `final` checkpoint, converts LitGPT weight names to HF names, and saves the result.

The example explicitly disables weight tying to handle a published configuration that may differ from the training setup.

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

repo = "mssfj/qwen25-0.5b-fineweb-edu-10bt"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
) if device == "cuda" else torch.float32
config = AutoConfig.from_pretrained(repo)
config.tie_word_embeddings = False
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(
    repo, config=config, torch_dtype=dtype
).to(device).eval()
inputs = tokenizer(
    "The answer to 2 + 3 is",
    return_tensors="pt", add_special_tokens=False
).to(device)
with torch.inference_mode():
    output = model.generate(
        **inputs, max_new_tokens=64, do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
print(tokenizer.decode(
    output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
))
```

