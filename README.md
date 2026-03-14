# LLM Eval

Qwen3.5-9B の GPTQ 量子化スクリプトは [`scripts/quantize_qwen35_9b_gptq.py`](/workspace/llm-2026-eval/scripts/quantize_qwen35_9b_gptq.py) を使います。

## Qwen3.5 GPTQ

96GB GPU を前提にした推奨実行例:

```bash
.venv/bin/python scripts/quantize_qwen35_9b_gptq.py \
  --output-dir /workspace/llm-2026-eval/model/Qwen3.5-9B-GPTQ-INT8 \
  --calibration-preset math_qa_cot \
  --max-calibration-samples 128 \
  --max-seq-len 8192 \
  --bits 8
```

現在のデフォルト値:

- `--max-calibration-samples 128`
- `--max-seq-len 8192`
- `--bits 8`

補足:

- `math_qa_cot` の calibration サンプルは長文なので、`max-seq-len 2048` では大きく切り詰められます。
- 量子化前に `vLLM` など GPU を占有するプロセスを停止してください。
- `Qwen3.5` 対応のため、環境の `transformers` は GitHub 版の最新版を使います。セットアップは [`vastai-setup_uv.sh`](/workspace/llm-2026-eval/vastai-setup_uv.sh) を参照してください。
