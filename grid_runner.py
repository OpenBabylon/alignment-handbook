#!/usr/bin/env python
# grid_runner.py
import subprocess, time, uuid, os, torch, tempfile, yaml as pyyaml
from pathlib import Path
from itertools import product
from copy import deepcopy
from ruamel.yaml import YAML

BASE_CFG   = Path("recipes/sft_mykola/orpo_mamy.yaml")
OUT_ROOT   = Path("data") / "grid_runs"
WANDB_PROJ = "MamayLM-ORPO-grid"

# ---------------- grid definition ----------------
training_modes = ["full"]            # or ["full", "lora"]
learning_rates = [1e-6, 5e-6, 1e-5, 5e-5]
batch_sizes    = [8, 16, 32]         # global batch size
lora_ranks     = [8, 64, 256]
num_epochs     = [3]
# -------------------------------------------------

yaml     = YAML(typ="safe")
base_cfg = yaml.load(BASE_CFG)
n_gpu    = torch.cuda.device_count() or 1

def make_cfg(mode, lr, global_bs, ep, r=None):
    cfg = deepcopy(base_cfg)

    # derive per-GPU batch (bs_dev) and accumulation (G)
    bs_dev, G = None, None
    for cand in range(min(global_bs, 8), 0, -1):
        if global_bs % (cand * n_gpu) == 0:
            bs_dev = cand
            G      = global_bs // (cand * n_gpu)
            break
    if bs_dev is None:
        raise ValueError(f"No per-device batch fits global_bs={global_bs} for n_gpu={n_gpu}")

    cfg["per_device_train_batch_size"] = int(bs_dev)
    cfg["gradient_accumulation_steps"] = int(G)
    cfg["learning_rate"]               = float(lr)
    cfg["num_train_epochs"]            = int(ep)

    if mode == "lora":
        cfg.update({"peft_type": "LORA", "lora_r": r, "lora_alpha": 16 * r})
        cfg["hub_model_id"] += f"-lora{r}"
    else:
        for k in ("peft_type", "lora_r", "lora_alpha"):
            cfg.pop(k, None)

    run_id   = uuid.uuid4().hex[:6]
    run_name = (
        f"{mode}_lr{lr:g}_gbs{global_bs}_pdev{bs_dev}_G{G}"
        + (f"_r{r}" if mode == "lora" else "")
        + f"_ep{ep}_{run_id}"
    )
    cfg["output_dir"] = str(OUT_ROOT / run_name)

    return run_name, cfg, bs_dev, G      # ← return bs_dev & G for later

# ---------------- iterate & launch ----------------
OUT_ROOT.mkdir(parents=True, exist_ok=True)
configs_dir = Path("auto_configs"); configs_dir.mkdir(exist_ok=True)

for mode in training_modes:
    ranks = lora_ranks if mode == "lora" else [None]
    for lr, gbs, ep, r in product(learning_rates, batch_sizes, num_epochs, ranks):
        run_name, cfg, bs_dev, G = make_cfg(mode, lr, gbs, ep, r)

        cfg_path = configs_dir / f"{run_name}.yaml"
        yaml.dump(cfg, cfg_path)

        env = os.environ.copy()
        env["WANDB_RUN_NAME"] = run_name
        env["WANDB_TAGS"]     = f"grid,{mode}"

        print(f"\n=== Launching {run_name} ===")

        # ----------  DeepSpeed config resolved on-the-fly  ----------
        ### <<< new block
        ds_cfg = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "DEEPSPEED",
            "mixed_precision": "bf16",
            "num_processes": n_gpu,
            "deepspeed_config": {
                "zero_stage": 3,
                "train_micro_batch_size_per_gpu": int(bs_dev),
                "gradient_accumulation_steps":     int(G),
                "zero3_init_flag": True,
                "zero3_save_16bit_model": True,
            },
        }
        tmp_cfg = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".yaml",
            mode="w",           # ← text mode instead of default binary
            encoding="utf-8",
        )
        pyyaml.safe_dump(ds_cfg, tmp_cfg)
        tmp_cfg.close()
        ### >>> end new block

        cmd = [
            "accelerate", "launch",
            "--config_file", tmp_cfg.name,   # use the resolved DS file
            "scripts/run_orpo_mamy.py",
            str(cfg_path),
        ]
        completed = subprocess.run(cmd, env=env)
        if completed.returncode:
            print(f"[{run_name}] failed (exit {completed.returncode})")
        time.sleep(10)
