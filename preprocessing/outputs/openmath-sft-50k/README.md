---
license: cc-by-4.0
language:
- en
task_categories:
- text-generation
size_categories:
- 10K<n<100K
tags:
- math
- sft
- qwen2
- openmathinstruct
- tdt
- research
pretty_name: Qwen2.5 OpenMathInstruct-2 SFT 50K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: validation
    path: data/validation-*.parquet
---

# Qwen2.5 OpenMathInstruct-2 SFT 50K

A 50,000-example, short-context mathematics SFT dataset extracted from
[NVIDIA OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2).
It was prepared for instruction tuning
[mssfj/qwen25-0.5b-finemath-4plus](https://huggingface.co/mssfj/qwen25-0.5b-finemath-4plus)
and subsequent GSM8K and MATH-500 benchmarking, including controlled comparisons with TDT.
This is an extracted and filtered dataset, not a newly generated set of solutions.

## Composition

MATH-related examples make up 70% of each split; GSM8K-related examples make up 30%.
These categories describe the upstream `problem_source`, not a newly assigned difficulty label.
Each selected numeric problem template occurs only once across the entire dataset.

| Split | Examples | MATH-related | GSM8K-related | Total formatted tokens | Maximum tokens |
|---|---:|---:|---:|---:|---:|
| train | 49,000 | 34,300 | 14,700 | 21,212,482 | 1854 |
| validation | 1,000 | 700 | 300 | 427,265 | 1868 |

The token limit is **2,048 per complete conversation**, including the system prompt,
user problem, full assistant solution, and Qwen chat-template delimiters. Problems and
solutions are not truncated. The published solution text is retained apart from trimming
outer whitespace; no answer line or solution was regenerated or appended.

## Source and provenance

- Source: `nvidia/OpenMathInstruct-2`, split `train_1M`.
- Source revision: `469216e3f46f4dacf476b382e192485ea51a143e`.
- Upstream fields: `problem`, `generated_solution`, `expected_answer`, `problem_source`.
- Upstream teacher model: Llama3.1-405B-Instruct, as documented by NVIDIA.
- Tokenizer: `mssfj/qwen25-0.5b-finemath-4plus`, revision
  `0600e6f3b418e77d7323b8934451ac66d49a1185`, using its Qwen2.5 tokenizer and chat template.
- Selection and split seed: 42.

The source paper is [OpenMathInstruct-2: Accelerating AI for Math with Massive
Open-Source Instruction Data](https://arxiv.org/abs/2410.01560),
by Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin,
Alexan Ayrapetyan, and Igor Gitman (2024).

## Selection procedure

1. Scan all 1,000,000 rows in the pinned `train_1M` files in sorted filename order.
2. Reject empty required fields, unknown source categories, problems containing `[asy]`
   diagram markup, and problem/solution text containing chat control-token strings.
3. Require a balanced final `\boxed{...}` or `\fbox{...}` and compare it with
   `expected_answer` after limited whitespace and LaTeX-format normalization.
   This is a string-consistency check, not verification of the mathematical reasoning.
4. Deduplicate normalized problems. Choose the shortest answer-consistent solution by
   character length, with a seeded SHA-256 tie-breaker.
5. Render complete conversations with the pinned chat template and reject lengths
   greater than 2,048 tokenizer tokens.
6. Deduplicate numeric templates. Normalization uses NFKC, lowercase text, removes
   selected LaTeX presentation commands, and keeps letter, number, and math-operator
   tokens. Numeric templates replace numeric literals with `NUM`. Keep the problem with
   the lowest seeded SHA-256 rank per template.
7. Form a deterministic candidate pool of up to 70,000 MATH-related and 30,000
   GSM8K-related examples. Apply the benchmark checks below to this candidate pool.
8. Select 35,000 MATH-related and 15,000 GSM8K-related surviving examples by seeded hash.
   Assign 700 MATH-related and 300 GSM8K-related examples to validation using a separate
   seeded hash ordering, leaving 49,000 training examples. Shuffle file order by another
   deterministic hash.
9. Retokenize every exported example to verify its saved length and check that normalized
   problem IDs and numeric-template group IDs are disjoint between splits.

This favors short complete solutions and template diversity. It is not a representative
random sample of the original corpus or its difficulty distribution. Derived problems
whose templates change substantially may still occur in both train and validation.

## Benchmark overlap filtering

Reference sets:

- `openai/gsm8k`, `main`, `test`: 1,319 problems, revision
  `740312add88f781978c0658806c59bc2815b9866`.
- `HuggingFaceH4/MATH-500`, `test`: 500 problems, revision
  `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.

An example is removed when its problem meets any of these conditions:

- Exact match after problem normalization.
- Exact numeric-template match, targeting number-substitution variants.
- RapidFuzz normalized-problem ratio of at least 90 against any benchmark problem.
- Maximum sentence-embedding cosine similarity of at least 0.94.
- Maximum cosine similarity of at least 0.90, together with a normalized-problem ratio
  of at least 70 or a numeric-template ratio of at least 85 for that nearest benchmark.

Embedding model: `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with normalized embeddings and a maximum
input length of 512 embedding-model tokens. Longer questions are truncated for this
similarity check only, not in the SFT data. These thresholds are heuristics: they can
remove related but distinct problems and miss paraphrases or mathematical equivalents.
**This dataset does not guarantee complete semantic decontamination.**

The filtering checks problem text, not all mentions within generated solutions.
It also does not undo possible benchmark exposure during the base model's prior
FineWeb-Edu or FineMath pretraining, or during the teacher model's training.
GSM8K training problems remain eligible; only the named evaluation sets are exclusion
references. MATH test problems outside MATH-500 were not an additional exclusion set.

`decontamination_audit.jsonl` records removed candidate IDs, matched benchmark row IDs,
and matching scores. It does not reproduce benchmark questions or answers.
`selection_stats.json` records filter counts, split composition, source revisions,
template/script fingerprints, token statistics, and Parquet SHA-256 checksums.

## Fields

| Field | Description |
|---|---|
| `messages` | System/user/assistant messages for SFT |
| `problem` | Selected upstream problem |
| `generated_solution` | Selected upstream solution |
| `expected_answer` | Upstream reference or majority-vote answer |
| `problem_source` | `math`, `augmented_math`, `gsm8k`, or `augmented_gsm8k` |
| `problem_id` | SHA-256 of the normalized problem |
| `problem_group_id` | SHA-256 of its numeric template |
| `token_count` | Complete conversation length under the pinned tokenizer/template |
| `source_dataset`, `source_revision`, `source_split` | Upstream provenance |
| `source_file`, `source_row` | Source Parquet path and zero-based row number |
| `nearest_benchmark_cosine` | Maximum embedding similarity to the exclusion references; audit metadata only |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("mssfj/qwen25-openmathinstruct2-sft-50k")
train = dataset["train"]
validation = dataset["validation"]
print(train[0]["messages"])
```

Use `messages` for training and keep the reference/audit fields out of the prompt.
The system prompt is:

```text
Solve the math problem step by step. Put the final answer in \boxed{...}.
```

Render with the pinned Qwen chat template (`chat_template.jinja` is also supplied),
train with loss on the assistant response and its turn-ending token, and keep the
input embeddings and output weights untied to match the base checkpoint. At inference,
stop on the Qwen assistant turn terminator `<|im_end|>` as well as the base EOS token.
Use this dataset's validation split for model selection; keep GSM8K test and MATH-500
for final evaluation. For TDT comparisons, fix the dataset revision and examples,
initial weights, training token budget, template, and evaluation settings.

## Reproduction

The included `prepare_openmath_sft.py` downloads pinned input revisions and rebuilds
the Parquet files and selection statistics. A CUDA GPU accelerates embedding checks.
Install a suitable PyTorch build for your system, then:

```bash
pip install -r requirements.txt
python prepare_openmath_sft.py \
  --output ./rebuilt-openmath-sft-50k \
  --total 50000 --validation 1000 --math-ratio 0.70 \
  --max-tokens 2048 --seed 42 --candidate-multiplier 2
```

`environment.json` records the generation environment, including the CUDA PyTorch
build. Different numerical environments can change borderline embedding decisions.
Compare output checksums and statistics when reproducing the dataset. The build script
creates data and statistics; this README and dependency/environment files accompany the
published release separately.

## Build statistics

Counters refer to the stage where each filter is applied; they are not all counts over
the same population. Similarity checks apply to the hash-selected candidate pool.

| Counter | Count |
|---|---:|
| `source_rows` | 1,000,000 |
| `diagram_or_control_token` | 21,843 |
| `missing_or_mismatched_final_box` | 993 |
| `duplicate_normalized_problem` | 386,845 |
| `empty_field` | 733 |
| `unique_answer_checked_candidates` | 589,586 |
| `over_token_limit` | 355 |
| `duplicate_numeric_template` | 28,173 |
| `eligible_math` | 481,746 |
| `eligible_gsm8k` | 79,312 |
| `similarity_candidate_pool` | 100,000 |
| `benchmark_fuzzy_ratio_90` | 81 |
| `benchmark_numeric_template_exact` | 11 |
| `embedding_input_over_512_tokens` | 169 |
| `benchmark_semantic_similarity` | 11 |

## License and attribution

This derived dataset follows the upstream OpenMathInstruct-2 dataset's **CC BY 4.0**
license. Attribute NVIDIA and the OpenMathInstruct-2 authors, and identify this release
as an extracted subset with formatting, consistency, length, deduplication, benchmark
overlap filtering, and split changes. Refer to the
[upstream dataset card](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
for the original release and citation.
