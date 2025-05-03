#!/usr/bin/env python
# grid_runner.py
import subprocess, shutil, time, uuid
from pathlib import Path
from itertools import product
from copy import deepcopy
from ruamel.yaml import YAML

BASE_CFG = Path("recipes/sft_mykola/orpo_mamy.yaml")
OUT_ROOT = Path("data") / "grid_runs"          # everything lives here
WANDB_PROJECT = "Mam﻿ayLM-ORPO-grid"

# ---------------------------------------------------------------------
# 1️⃣  Define the grid
training_modes     = ["full", "lora"]           # full-finetune or LoRA
learning_rates     = [1e-6, 5e-6, 1e-5, 5e-5]
batch_sizes        = [8, 16, 32]
lora_ranks         = [8, 64, 256]               # used only when mode == "lora"
num_epochs         = [3]
# lr schedule is fixed in base yaml (cosine + warm-up)
# ---------------------------------------------------------------------

yaml = YAML(typ="safe")
base_cfg = yaml.load(BASE_CFG)

def make_cfg(m, lr, bs, ep, r=None):
    cfg = deepcopy(base_cfg)

    # --- pieces that always vary ---
    cfg["learning_rate"]              = lr
    cfg["per_device_train_batch_size"] = bs
    cfg["num_train_epochs"]           = ep

    # --- LoRA vs full finetune ---
    if m == "lora":
        cfg["peft_type"]  = "LORA"
        cfg["lora_r"]     = r
        cfg["lora_alpha"] = 2 * r      # common heuristic
        cfg["hub_model_id"] += f"-lora{r}"
    else:  # full-parameter
        cfg.pop("peft_type",  None)
        cfg.pop("lora_r",     None)
        cfg.pop("lora_alpha", None)

    # --- bookkeeping paths & names ---
    run_id   = uuid.uuid4().hex[:6]
    run_name = f"{m}_lr{lr}_bs{bs}" + (f"_r{r}" if m=="lora" else "") + f"_ep{ep}_{run_id}"
    cfg["output_dir"] = str(OUT_ROOT / run_name)
    cfg["logging_steps"] = 10
    cfg["wandb_run_name"] = run_name
    cfg["tags"] = ["grid", m]

    return run_name, cfg

# ---------------------------------------------------------------------
# 2️⃣  Iterate and launch
configs_dir = Path("auto_configs")
configs_dir.mkdir(exist_ok=True)

for mode in training_modes:
    # if full-finetune, we only iterate over one dummy rank
    rank_list = lora_ranks if mode == "lora" else [None]
    for lr, bs, ep, r in product(learning_rates, batch_sizes, num_epochs, rank_list):
        run_name, cfg = make_cfg(mode, lr, bs, ep, r)
        cfg_path = configs_dir / f"{run_name}.yaml"
        yaml.dump(cfg, cfg_path)

        print(f"\n=== Launching {run_name} ===")
        cmd = [
            "accelerate", "launch",
            "--config_file", "recipes/accelerate_configs/deepspeed_zero3.yaml",
            "scripts/run_orpo_mamy.py",
            str(cfg_path),
        ]
        # subprocess.run blocks until training finishes; catch non-zero exits
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            print(f"[{run_name}] failed (exit {completed.returncode}) — skipping")
        time.sleep(10)  # tiny pause so W&B/FS sync can flush
