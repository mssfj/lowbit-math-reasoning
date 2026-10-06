"""Export a local LitGPT SFT checkpoint for Transformers/vLLM evaluation."""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("/tmp/finemath-eval-model"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    convert_lit_checkpoint(args.checkpoint_dir, args.out_dir)
    state = torch.load(args.out_dir / "model.pth", map_location="cpu", weights_only=True)
    save_file({key: value.to(torch.bfloat16).contiguous() for key, value in state.items()},
              args.out_dir / "model.safetensors", metadata={"format": "pt"})
    (args.out_dir / "model.pth").unlink()
    for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "added_tokens.json", "special_tokens_map.json", "chat_template.jinja"]:
        if (args.base_dir / name).exists():
            shutil.copy2(args.base_dir / name, args.out_dir / name)
    config = json.loads((args.base_dir / "config.json").read_text())
    config.update(tie_word_embeddings=False, torch_dtype="bfloat16", eos_token_id=[151645, 151643])
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out_dir / "generation_config.json").write_text(json.dumps({
        "bos_token_id": 151643, "eos_token_id": [151645, 151643], "pad_token_id": 151643,
        "do_sample": False,
    }, indent=2) + "\n")
    # Native LitGPT generation also stops at the trained assistant-turn terminator.
    for directory in [args.out_dir, args.checkpoint_dir]:
        path = directory / "tokenizer_config.json"
        tokenizer_config = json.loads(path.read_text())
        tokenizer_config["eos_token"] = "<|im_end|>"
        path.write_text(json.dumps(tokenizer_config, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.out_dir), "parameters": sum(x.numel() for x in state.values()),
                      "tie_word_embeddings": False, "eos_token_id": [151645, 151643]}, indent=2))


if __name__ == "__main__":
    main()
