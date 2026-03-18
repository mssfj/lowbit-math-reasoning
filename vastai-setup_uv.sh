#!/usr/bin/env bash
set -euxo pipefail

# ==== 設定 ====
if [ "$#" -lt 1 ]; then
  echo "プロジェクトのルートディレクトリの指定が必要です。例) bash vastai-setup_uv.sh /workspace/lowbit-math-reasoning" >&2
  exit 1
fi
PROJECT_ROOT="$(realpath -m "$1")"
BUILD_ROOT="${PROJECT_ROOT}/build"

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

mkdir -p ~/.codex
a​ws s3 cp s3://llm-train-dev/codex/auth.json ~/.codex/auth.json

npm cache clean -f
npm install -g n
n lts

export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
hash -r

npm install -g @openai/codex@latest
npm install -g @google/gemini-cli
hash -r

# ==== 1. uv インストール ====
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
hash -r

# ==== 2. プロジェクトディレクトリ ====
mkdir -p "${PROJECT_ROOT}" "${BUILD_ROOT}"
cd "${PROJECT_ROOT}"

# ==== 3. ルート pyproject.toml を作成（評価用） ====
cat > pyproject.toml << 'PYPROJECT_EVAL'
[project]
name = "llm-eval"
version = "0.1.0"
description = "LLM Eval Pipeline"
requires-python = ">=3.10,<3.12"

# 共通依存
dependencies = [
    "accelerate",
    "datasets",
    "peft",
    "bitsandbytes",
    "sentencepiece",
    "evaluate",
    "wandb==0.22.3",
    "tiktoken",
    "scikit-learn",
    "numpy",
    "einops",
    "setuptools",
    "sympy",
    "unsloth",
    "gptqmodel>=5.7.0",
    "optimum>=2.1.0",
]

[dependency-groups]
sft = [
]

grpo = [
]

eval = [
    "transformers @ git+https://github.com/huggingface/transformers.git@v4.57.6",
    "vllm==0.17.0",
]

dev = [
    "pytest",
    "ipykernel",
]
PYPROJECT_EVAL

# ==== 4. build/pyproject.toml を作成（量子化・変換用） ====
cat > "${BUILD_ROOT}/pyproject.toml" << 'PYPROJECT_BUILD'
[project]
name = "llm-build"
version = "0.1.0"
description = "Separate build environment for quantization and model conversion"
requires-python = ">=3.10,<3.12"
dependencies = [
    "transformers @ git+https://github.com/huggingface/transformers.git",
    "accelerate",
    "datasets",
    "sentencepiece",
    "gptqmodel>=5.7.0",
    "optimum>=2.1.0",
    "bitsandbytes",
]

[dependency-groups]
dev = [
    "ipykernel",
]
PYPROJECT_BUILD

# ==== 5. lock と sync ====
uv lock
uv sync --group sft --group grpo --group dev --group eval

cd "${BUILD_ROOT}"
uv lock
uv sync
cd "${PROJECT_ROOT}"

# ==== 6. PyTorch (CUDA 12.1 wheel) ====
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.4.0" \
    "torchvision==0.19.0" \
    "torchaudio==2.4.0"

uv pip install --python "${BUILD_ROOT}/.venv/bin/python" --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.4.0" \
    "torchvision==0.19.0" \
    "torchaudio==2.4.0"

# ==== 7. ディレクトリ構成 ====
mkdir -p "${PROJECT_ROOT}/configs" "${PROJECT_ROOT}/outputs" "${PROJECT_ROOT}/model"

# ==== 8. 動作確認 ====
uv run --group eval python - << 'PYCODE'
import torch
import vllm
import transformers
print("eval torch version:", torch.__version__)
print("eval cuda available:", torch.cuda.is_available())
print("eval cuda version:", torch.version.cuda)
print("eval vllm version:", vllm.__version__)
print("eval transformers version:", transformers.__version__)
PYCODE

(
  cd "${BUILD_ROOT}"
  uv run python - << 'PYCODE'
import torch
import transformers
print("build torch version:", torch.__version__)
print("build cuda available:", torch.cuda.is_available())
print("build cuda version:", torch.version.cuda)
print("build transformers version:", transformers.__version__)
PYCODE
)

echo "=== setup done. ==="
echo "eval環境:"
echo "  cd ${PROJECT_ROOT}"
echo "  uv run --group eval python scripts/gsm8k-eval.py --model-name ./model/Qwen3.5-9B-GPTQ-INT4"
echo
echo "build環境:"
echo "  cd ${BUILD_ROOT}"
echo "  uv run python ../scripts/quantize_qwen35_9b_gptq.py \\\n    --output-dir ${PROJECT_ROOT}/model/Qwen3.5-9B-GPTQ-INT4 \\\n    --calibration-preset math_qa_cot \\\n    --max-calibration-samples 32 \\\n    --max-seq-len 512 \\\n    --bits 4"
echo "  ※ vLLM など GPU を占有するプロセスは事前に停止すること。"

# ==== 9. git 初期化 ====
git config --global user.email "mss.fujimoto@gmail.com"
git config --global user.name "Masashi Fujimoto"

# ==== 10. クリーニング ====
rm -rf ./aws
rm -f ./awscliv2.zip
