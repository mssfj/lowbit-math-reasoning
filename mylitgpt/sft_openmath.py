"""Run full SFT on the pinned, decontaminated OpenMath subset with local LitGPT."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer

from litgpt.args import EvalArgs, TrainArgs
from litgpt.data.openmath_sft import OpenMathPrompt, OpenMathSFT
from litgpt.finetune.full import setup
import litgpt.finetune.full as full_finetune

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "mssfj/qwen25-openmathinstruct2-sft-50k"
DATASET_REVISION = "95cdc178b2dda77628f98a801b99f4d549db403a"


def prepare(source, destination, checkpoint):
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    stats = {"dataset": DATASET_ID, "revision": DATASET_REVISION, "splits": {}}
    style = OpenMathPrompt()
    for split, name in [("train", "train"), ("validation", "val")]:
        rows = pq.read_table(source / "data" / f"{split}-00000-of-00001.parquet").to_pylist()
        samples, lengths = [], []
        for row in rows:
            answer = row["expected_answer"].strip()
            solution = row["generated_solution"].strip()
            final = "\\boxed{" + answer + "}"
            if solution.splitlines()[-1].strip() != final:
                solution += "\n\n" + final
            prompt = style.apply(row["problem"])
            length = len(tokenizer.encode(prompt, add_special_tokens=False)) + len(
                tokenizer.encode(solution, add_special_tokens=False)
            ) + 1
            if length > 2048:
                raise ValueError(f"Sample {row['problem_id']} would be truncated ({length} tokens)")
            samples.append({"instruction": row["problem"], "output": solution})
            lengths.append(length)
        (destination / f"{name}.json").write_text(json.dumps(samples, ensure_ascii=False))
        stats["splits"][split] = {"rows": len(samples), "tokens": sum(lengths), "max_length": max(lengths)}
    (destination / "preparation.json").write_text(json.dumps(stats, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/tmp/finemath-eval-model"))
    parser.add_argument("--source", type=Path, default=ROOT / "preprocessing/outputs/openmath-sft-50k")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/openmath-sft-litgpt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out/finemath-openmath-sft")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()
    if not args.resume and not args.skip_prepare:
        print(json.dumps(prepare(args.source, args.data_dir, args.checkpoint_dir), indent=2), flush=True)
    if args.prepare_only:
        return
    settings = dict(
        checkpoint_dir=args.checkpoint_dir, out_dir=args.out_dir,
        precision="bf16-mixed", devices=1, seed=42, resume=args.resume,
        data=OpenMathSFT(json_path=args.data_dir, mask_prompt=True, prompt_style=OpenMathPrompt(), seed=42, num_workers=2),
        train=TrainArgs(epochs=1, max_steps=args.max_steps, global_batch_size=32, micro_batch_size=1,
                        max_seq_length=2048, lr_warmup_steps=50, save_interval=250, log_interval=32),
        eval=EvalArgs(interval=250, max_iters=100, max_new_tokens=256, final_validation=True),
        optimizer={"class_path": "torch.optim.AdamW", "init_args": {
            "lr": 2e-5, "betas": [0.9, 0.95], "weight_decay": 0.1, "fused": True}},
        logger_name="csv", activation_checkpointing=True,
    )
    # LitGPT's stock saver reparses sys.argv as its own CLI. This runner calls
    # setup programmatically, so persist the actual arguments instead.
    serialized = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in settings.items()
           if key not in {"data", "train", "eval"}},
        "train": asdict(settings["train"]), "eval": asdict(settings["eval"]),
        "data": {"class_path": "litgpt.data.openmath_sft.OpenMathSFT", "init_args": {
            "json_path": str(args.data_dir), "mask_prompt": True, "seed": 42, "num_workers": 2,
            "prompt_style": {"class_path": "litgpt.data.openmath_sft.OpenMathPrompt"}}},
    }

    def save_actual_hyperparameters(function, checkpoint_dir):
        (checkpoint_dir / "hyperparameters.yaml").write_text(yaml.safe_dump(serialized, sort_keys=False))

    full_finetune.save_hyperparameters = save_actual_hyperparameters
    # Exercise the same metadata saver before spending time on GPU training.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_actual_hyperparameters(setup, args.out_dir)
    setup(**settings)


if __name__ == "__main__":
    main()
