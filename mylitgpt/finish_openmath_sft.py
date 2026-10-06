"""Monitor a training process, then export and smoke-check the completed SFT model.

An explicitly enabled upload publishes both formats; benchmarks are not run here.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_REPO = "mssfj/qwen25-0.5b-finemath-4plus-openmath-sft"
LIT_REPO = HF_REPO + "-lit"


def upload_models(checkpoint, export, run_dir, report):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    results = {}
    for repo_id, directory in [(LIT_REPO, checkpoint), (HF_REPO, export)]:
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=directory,
                                   commit_message="Upload LitGPT OpenMath full SFT model and English model card")
        info = api.model_info(repo_id, revision=commit.oid)
        files = {item.rfilename for item in info.siblings}
        required = {"README.md", "training_report.json", "tokenizer.json", "tokenizer_config.json"}
        required |= {"lit_model.pth", "model_config.yaml", "prompt_style.yaml"} if repo_id == LIT_REPO else {
            "model.safetensors", "config.json", "generation_config.json"}
        if not required.issubset(files):
            raise RuntimeError(f"Missing uploaded files in {repo_id}: {required - files}")
        card = Path(hf_hub_download(repo_id, "README.md", revision=commit.oid)).read_text()
        if card != (directory / "README.md").read_text():
            raise RuntimeError(f"Uploaded model card differs for {repo_id}")
        results[repo_id] = {"url": f"https://huggingface.co/{repo_id}", "revision": commit.oid,
                            "verified_files": sorted(required)}
        (run_dir / "uploads.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(path, state, **kwargs):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"status": state, "updated_at": now(), **kwargs}, indent=2) + "\n")
    temporary.replace(path)


def process_running(pid):
    try:
        # A zombie has exited even if its parent has not reaped it yet.
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def smoke_check(directory):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(ROOT / "eval"))
    from mymath_verify import extract_final_answer_with_meta
    from litgpt.data.openmath_sft import OpenMathPrompt

    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForCausalLM.from_pretrained(directory, torch_dtype=torch.bfloat16).to("cuda").eval()
    outputs = []
    for problem in ["What is 2 + 3?", "What is 18 multiplied by 7?", "Solve for x: 3x + 7 = 22."]:
        inputs = tokenizer(OpenMathPrompt().apply(problem), return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=512)
        text = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        extracted = extract_final_answer_with_meta(text)
        outputs.append({"problem": problem, "response": text, "extracted_answer": extracted.answer,
                        "has_final_answer": extracted.has_final_answer, "format": extracted.source})
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--training-log", type=Path, default=Path("/tmp/finemath-sft-training.log"))
    parser.add_argument("--run-dir", type=Path, default=ROOT / "out/finemath-openmath-sft")
    parser.add_argument("--base-dir", type=Path, default=Path("/tmp/finemath-eval-model"))
    parser.add_argument("--upload", action="store_true", help="Publish both model formats to Hugging Face")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.run_dir / "status.json"
    checkpoint = args.run_dir / "final"
    export = args.run_dir / "huggingface"
    metadata = {"training_pid": args.training_pid, "training_log": str(args.training_log),
                "litgpt_checkpoint": str(checkpoint), "hf_model": str(export),
                "upload_enabled": args.upload, "hf_repo": HF_REPO, "litgpt_repo": LIT_REPO}
    write_status(status_path, "training", **metadata)
    while process_running(args.training_pid):
        time.sleep(30)
    try:
        log = args.training_log.read_text()
        shutil.copy2(args.training_log, args.run_dir / "training.log")
        if not (checkpoint / "lit_model.pth").is_file() or "Traceback (most recent call last)" in log:
            raise RuntimeError("Training did not produce a successful final checkpoint; inspect training.log")
        write_status(status_path, "exporting", **metadata)
        subprocess.run([sys.executable, str(ROOT / "mylitgpt/export_openmath_sft.py"),
                        "--checkpoint-dir", str(checkpoint), "--base-dir", str(args.base_dir),
                        "--out-dir", str(export)], check=True)
        val_losses = re.findall(r"Final evaluation\s*\|\s*val loss:\s*([\d.]+)", log)
        report = {"base_model": "mssfj/qwen25-0.5b-finemath-4plus",
                  "dataset": "mssfj/qwen25-openmathinstruct2-sft-50k",
                  "dataset_revision": "95cdc178b2dda77628f98a801b99f4d549db403a",
                  "training": {"method": "LitGPT full SFT", "epochs": 1, "learning_rate": 2e-5,
                               "global_batch_size": 32, "micro_batch_size": 1,
                               "precision": "bf16-mixed", "seed": 42, "mask_prompt": True,
                               "max_seq_length": 2048, "activation_checkpointing": True},
                  "final_val_loss": float(val_losses[-1]) if val_losses else None,
                  "format": "Last line: \\boxed{ANSWER}", "completed_at": now()}
        data_dir = ROOT / "data/openmath-sft-litgpt"
        for name in ["preparation.json", "verification.json"]:
            if (data_dir / name).exists():
                report[name.removesuffix(".json")] = json.loads((data_dir / name).read_text())
        write_status(status_path, "checking_generation", **metadata)
        report["generation_smoke_check"] = smoke_check(export)
        (args.run_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
        shutil.copy2(args.run_dir / "training_report.json", export / "training_report.json")
        shutil.copy2(ROOT / "model_cards/qwen25-0.5b-finemath-openmath-sft.md", export / "README.md")
        card = (export / "README.md").read_text()
        card += f"\n## Training result\n\nFinal validation loss: {report['final_val_loss']} (LitGPT's printed precision). "
        card += "See `training_report.json` for generation checks and dataset preparation statistics.\n"
        card += f"\nFormats: [Transformers](https://huggingface.co/{HF_REPO}) and [LitGPT](https://huggingface.co/{LIT_REPO}).\n"
        (export / "README.md").write_text(card)
        native_card = card.replace("library_name: transformers", "library_name: litgpt")
        native_card += "\n## Native LitGPT checkpoint\n\nThis repository contains `lit_model.pth` with FP32 trained master weights, "
        native_card += "`model_config.yaml`, tokenizer files, and `prompt_style.yaml`. It contains final model weights, "
        native_card += "without optimizer state. Use the `mylitgpt` implementation in lowbit-math-reasoning, which includes "
        native_card += "the `litgpt.data.openmath_sft.OpenMathPrompt` class referenced by the saved prompt style. "
        native_card += "A copy of that module is included as `openmath_sft.py` for provenance.\n"
        (checkpoint / "README.md").write_text(native_card)
        shutil.copy2(args.run_dir / "training_report.json", checkpoint / "training_report.json")
        shutil.copy2(ROOT / "mylitgpt/litgpt/data/openmath_sft.py", checkpoint / "openmath_sft.py")
        for directory in [checkpoint, export]:
            for name in ["sft_openmath.py", "export_openmath_sft.py"]:
                shutil.copy2(ROOT / "mylitgpt" / name, directory / name)
            for name in ["chat_template.jinja", "special_tokens_map.json", "added_tokens.json", "merges.txt", "vocab.json"]:
                if (args.base_dir / name).exists():
                    shutil.copy2(args.base_dir / name, directory / name)
        uploads = {}
        if args.upload:
            write_status(status_path, "uploading", **metadata)
            uploads = upload_models(checkpoint, export, args.run_dir, report)
        write_status(status_path, "complete", final_val_loss=report["final_val_loss"], uploads=uploads, **metadata)
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        write_status(status_path, "failed", error=str(error), **metadata)
        raise


if __name__ == "__main__":
    main()
