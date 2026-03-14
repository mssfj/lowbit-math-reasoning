#!/usr/bin/env bash
set -euxo pipefail

# ==== 設定 ====
if [ "$#" -lt 1 ]; then
	echo "プロジェクトのルートディレクトリの指定が必要です。例)bash vastai-setup_uv.sh /workspace/llm-2026-eval,/root/llm-2026-eval" >&2
  exit 1
fi
PROJECT_ROOT="$(realpath -m "$1")"

# ==== 0. 基本パッケージ ====
sudo apt-get update
sudo apt-get install -y \
  git wget curl build-essential \
  python3-dev python3-pip \
  pkg-config nodejs npm unzip

# ==== codexのインストール ====
npm i -g @openai/codex

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install --update
aws configure

aws s3 cp s3://llm-train-dev/codex/auth.json ~/.codex/auth.json

# キャッシュをクリア
npm cache clean -f

# バージョン管理ツール 'n' をインストール
npm install -g n

# 最新のLTS（推奨版）をインストール
n lts

# --- 【ここが修正ポイント：新しいパスを強制認識】 ---
export PATH="/usr/local/bin:$PATH"
hash -r
# ----------------------------------------------

npm install -g @openai/codex@latest

# 反映させるためにシェルを再起動、またはパスを通す
hash -r

# ==== gemini-cliのインストール ====
npm install -g @google/gemini-cli

# ==== 1. uv インストール ====
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# ==== 2. プロジェクトディレクトリ ====
mkdir -p "${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

if [ ! -f pyproject.toml ]; then
  uv init --python 3.10 .
fi

# ==== 3. pyproject.toml を上書き（コンペ用構成・グループ分け） ====
cat > pyproject.toml << 'EOF'
[project]
name = "llm-eval"
version = "0.1.0"
description = "LLM Eval Pipeline"
requires-python = ">=3.10"

# 共通依存
dependencies = [
    "transformers",
    "accelerate",
    "datasets",
    "peft",
    "bitsandbytes",
    "sentencepiece",
    "evaluate",
    # wandb 0.22.3+ supports newer W&B API key formats and works with NumPy 2.x.
    "wandb==0.22.3",
    "tiktoken",
    "scikit-learn",
    "numpy",
    "einops",
    "setuptools",
    "sympy",
    "unsloth",
    "vllm",
    "gptqmodel>=5.7.0",
    "optimum>=2.1.0",
]

# グループ依存（uv の新仕様）
[dependency-groups]
# SFT: 今後 SFT専用をここに追加
sft = [
]

# GRPO / RL / 推論
grpo = [
    "transformers==4.57.6",
    "trl==0.22.2",
    "vllm==0.17.0",
    "math-verify[antlr4_13_2]",
]

# 評価系（math-verify 等）
eval = [
]

# 開発用
dev = [
    "pytest",
    "ipykernel",
]
EOF

# ==== 4. 依存インストール（Torch 以外） ====
uv sync --group sft --group grpo --group dev --group eval

# Qwen3.5 GPTQ 量子化では、公開版 transformers 4.x では model_type=qwen3_5 を
# 認識できないため、GitHub の最新版で上書きする。
uv pip install --python .venv/bin/python --upgrade \
    git+https://github.com/huggingface/transformers.git

# ==== 5. PyTorch (CUDA 12.1 wheel) ====
uv pip install --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.4.0" \
    "torchvision==0.19.0" \
    "torchaudio==2.4.0"

# ==== 6. ディレクトリ構成 ====
# mkdir -p "${PROJECT_ROOT}"/logs "${PROJECT_ROOT}"/checkpoints "${PROJECT_ROOT}"/configs "${PROJECT_ROOT}"/data "${PROJECT_ROOT}"/model
mkdir -p "${PROJECT_ROOT}"/configs "${PROJECT_ROOT}"/outputs

# ==== 7. 動作確認 ====
uv run python - << 'PYCODE'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
PYCODE

echo "=== setup done. ==="
echo "次回以降は:"
echo "  cd ${PROJECT_ROOT}"
echo "  uv run python your_script.py"
echo
echo "Qwen3.5 GPTQ quantization で通った条件:"
echo "  .venv/bin/python scripts/quantize_qwen35_9b_gptq.py \\"
echo "    --output-dir ${PROJECT_ROOT}/model/Qwen3.5-9B-GPTQ-INT8 \\"
echo "    --calibration-preset math_qa_cot \\"
echo "    --max-calibration-samples 32 \\"
echo "    --max-seq-len 512 \\"
echo "    --bits 8"
echo "  ※ vLLM など GPU を占有するプロセスは事前に停止すること。"

# ==== 8.git 初期化 ====
git config --global user.email "mss.fujimoto@gmail.com"
git config --global user.name "Masashi Fujimoto"

# ==== 9.クリーニング ====
rm -r ./aws
rm ./awscliv2.zip
