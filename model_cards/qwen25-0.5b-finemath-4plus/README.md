---
library_name: transformers
pipeline_tag: text-generation
datasets:
- HuggingFaceFW/fineweb-edu
- HuggingFaceTB/finemath
base_model: mssfj/qwen25-0.5b-fineweb-edu-10bt
tags:
- litgpt
- qwen2
- pretraining
- tdt
- research
---

# qwen25-0.5b-finemath-4plus

A baseline model for TDT comparison experiments, pretrained on FineWeb-Edu and then continued-pretrained on FineMath-4plus. This repository contains the final weights in Transformers format.

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

Continued pretraining starts from the FineWeb-Edu pretrained weights and uses next-token prediction on mathematics documents.

- Initial checkpoint: [mssfj/qwen25-0.5b-fineweb-edu-10bt-litgpt](https://huggingface.co/mssfj/qwen25-0.5b-fineweb-edu-10bt-litgpt), the default in `continued_pretrain.sh`.
- Source data: [HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath), `finemath-4plus` configuration, `train` split.
- Pretokenized data: [mssfj/finemath-4plus-qwen25](https://huggingface.co/datasets/mssfj/finemath-4plus-qwen25).
- The preprocessing script divides the data into 128 Parquet shards and defaults to 32 tokenization workers.
- FineMath-stage training token budget: 9,600,000,000, separate from the FineWeb-Edu-stage budget.

The recorded settings for [W&B run `84pczgsb`](https://wandb.ai/mssfj-1/qwen25-finemath-pretrain/runs/84pczgsb) are:

| Property | Recorded value |
|---|---|
| Precision / seed | bf16-mixed / 42 |
| Global batch / micro batch | 512 / 8 |
| Optimizer | AdamW |
| Learning rate / minimum LR | 4e-4 / 4e-5, cosine decay after warmup |
| Warmup | 1,000 optimizer steps |
| Weight decay / betas | 0.1 / (0.9, 0.95) |
| Gradient clipping | Maximum norm 1.0 |
| Checkpoint / validation interval | 1,000 / 1,000 optimizer steps |
| Final optimizer step | 9,155 |
| Recorded tokens processed | 9,599,713,280 |
| Validation | Micro batch size 8, 100 batches |
| End-of-training validation | Disabled |

The [current YAML](https://github.com/mssfj/lowbit-math-reasoning/blob/main/mylitgpt/config_hub/pretrain/qwen25-0.5b-finemath-4plus.yaml) uses micro batch size 16 and a checkpoint interval of 2,000, whereas the recorded run used 8 and 1,000. Use the W&B records when reproducing the execution conditions of that run.

## Loss measurements

| Checkpoint | Validation loss | Source |
|---|---:|---|
| Step 9,000 | 1.580837 | Last validation record in W&B |
| Final | 1.579783 | Reevaluation of the published weights with LitGPT |

The W&B summary `val_loss` belongs to step 9,000, not the final checkpoint. The final checkpoint was reevaluated using pretokenized dataset revision `c0e59cf67bdca55c2b298a06a298f229dfecd447`, sequence length 2,048, original batch size 8, 100 batches, 8 data workers, seed 42, and bf16-mixed precision. To fit GPU memory, each batch was evaluated one sequence at a time and the losses were averaged. Reevaluation used PyTorch `2.12.0a0+0291f960b6.nv26.04.48445190`, a different environment from the original run. A strict comparison of this small loss difference requires reevaluating both checkpoints in the same environment.

This evaluation uses the training split and is not an independent measure of mathematics performance.

## Reproduction code

[continued_pretrain.sh](https://github.com/mssfj/lowbit-math-reasoning/blob/main/mylitgpt/continued_pretrain.sh) contains steps for obtaining the FineWeb-Edu checkpoint, preprocessing FineMath, continued pretraining, and conversion.

```bash
cd mylitgpt
litgpt pretrain \
  --config config_hub/pretrain/qwen25-0.5b-finemath-4plus.yaml \
  --initial_checkpoint_dir out/pretrain/qwen25-0.5b-fineweb-edu-10bt-lit \
  --out_dir out/pretrain/qwen25-0.5b-finemath-4plus
```

Prepare the tokenizer, pretokenized data, and initial checkpoint before training. This command uses the current YAML configuration.

## Checkpoint format and usage

This repository uses Hugging Face Transformers format with `model.safetensors`. The export pipeline removes optimizer state from the training `final` checkpoint, converts LitGPT weight names to HF names, and saves the result.

The HF configuration uses `tie_word_embeddings: false` to match training. The earlier `true` setting produced outputs different from LitGPT and was corrected. The example below also disables weight tying explicitly.

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

repo = "mssfj/qwen25-0.5b-finemath-4plus"
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

With the same inputs, FP32 weights, BF16 autocast, and greedy decoding, the LitGPT and HF versions produced identical 24-token sequences for both text completion and chat-formatted prompts. This is a short-input compatibility check, not an evaluation of conversational ability.

