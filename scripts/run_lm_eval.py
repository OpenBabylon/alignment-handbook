#!/usr/bin/env python
# coding=utf-8

import os
import sys
import json
import time
import yaml
import argparse
import torch
import lm_eval
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Single config-based eval script.")
    parser.add_argument("config", type=str, help="Path to the YAML config file.")
    return parser.parse_args()


def evaluate_lm_eval(model, tokenizer, tasks, limit, chat_template):
    """Runs lm_eval on a given model and tokenizer and returns scores."""
    print(f"\n🚀 Starting lm_eval for tasks={tasks}, limit={limit}, chat_template={chat_template}")

    lm_model = lm_eval.models.huggingface.HFLM(
        pretrained=model,  # Passing the already loaded model
        tokenizer=tokenizer
    )

    if limit < 0:
        limit = None

    results = lm_eval.simple_evaluate(
        model=lm_model,
        tasks=list(tasks),
        log_samples=True,
        limit=limit,
        apply_chat_template=bool(chat_template),
    )

    # Extract "acc,none" for each task
    final_scores = {task: results["results"][task]["acc,none"] for task in tasks}
    print("score:", final_scores)
    return final_scores


def main():
    # 1) Parse single YAML config argument
    args = parse_args()

    # 2) Load config from YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 3) Pull out relevant model + evaluation keys
    base_model_path = config.get("base_model_path", "Your-Default-Model")
    eval_save_dir = config.get("eval_save_dir", "eval_results")
    lm_eval_tasks = config.get("tasks", ["zno-language"])
    limit = config.get("limit", -1)
    chat_template = config.get("chat_template", 0)

    # 4) Make sure eval directory exists
    os.makedirs(eval_save_dir, exist_ok=True)

    # 5) Load model/tokenizer from the config's base_model_path
    print("\n🔍 Evaluating Base Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path).to(device)

    # 6) Evaluate base model
    all_results = {}
    all_results["base_model"] = evaluate_lm_eval(
        model=base_model,
        tokenizer=tokenizer,
        tasks=lm_eval_tasks,
        limit=limit,
        chat_template=chat_template
    )

    # 🟡 Free GPU memory used by base_model, if you don't need it anymore
    del base_model
    torch.cuda.empty_cache()


    # 7) Evaluate any training checkpoints found in config["output_dir"]
    checkpoints_dir = config.get("checkpoints_dir", None)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = os.path.join(checkpoints_dir,eval_save_dir, f"lm_eval_results_{timestamp}.json")
    print(f"\n✅ All results will be saved to {result_file}")
    directory = Path(os.path.join(checkpoints_dir,eval_save_dir))
    directory.mkdir(parents=True, exist_ok=True)

    if checkpoints_dir and os.path.isdir(checkpoints_dir):
        checkpoint_dirs = sorted(
            [
                os.path.join(checkpoints_dir, d)
                for d in os.listdir(checkpoints_dir)
                if d.startswith("checkpoint-")
            ],
            key=lambda x: int(x.split("-")[-1])
        )

        for checkpoint_path in checkpoint_dirs:
            print(f"\n🔍 Evaluating {checkpoint_path}...")
            ckpt_model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)

            all_results[checkpoint_path] = evaluate_lm_eval(
                model=ckpt_model,
                tokenizer=tokenizer,
                tasks=lm_eval_tasks,
                limit=limit,
                chat_template=chat_template
            )
            with open(result_file, "w") as f:
                json.dump(all_results, f, indent=4)
            print(f"\n✅ Results saved to {result_file}")

            # 🟡 Immediately delete and clear GPU cache
            del ckpt_model
            torch.cuda.empty_cache()
    else:
        print(f"\n⚠️ No valid checkpoints directory found at: {checkpoints_dir}")

    # 8) Save results to JSON

    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\n✅ All results saved to {result_file}")
    print("\n✅ lm_eval testing complete!")


if __name__ == "__main__":
    main()
