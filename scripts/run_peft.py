#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
"""
Supervised fine-tuning script for decoder language models with optional LoRA.
"""

import logging
import random
import sys

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    SFTConfig,
    EvalArguments,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,         # <--- We'll load a LoRA config from here
    get_quantization_config,
    get_tokenizer,
)
from trl import SFTTrainer, setup_chat_format

logger = logging.getLogger(__name__)

def main():
    # Parse YAML + CLI into arguments
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig, EvalArguments))
    model_args, data_args, training_args, eval_args = parser.parse()

    # Set random seed
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Summaries
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Possibly resume from checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming from {last_checkpoint}.")

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label", "conversations"],
    )
    logger.info(
        "Training on the following datasets and proportions: " +
        ", ".join([f"{split}: {dset.num_rows}" for split, dset in raw_datasets.items()])
    )
    train_dataset = raw_datasets["train"]
    # Optionally, there might be a test set or eval set: raw_datasets["test"]

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)

    ###################
    # Load base model
    ###################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=not training_args.gradient_checkpointing,
        device_map=get_kbit_device_map() if quantization_config else None,
        quantization_config=quantization_config,
    )

    # If the model is recognized as ChatML, do special formatting
    if "<|im_start|>" in tokenizer.chat_template and "gemma-tokenizer-chatml" not in tokenizer.name_or_path:
        base_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        base_model, tokenizer = setup_chat_format(base_model, tokenizer)
        model_kwargs = None
    else:
        base_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    ####################
    # Possibly apply LoRA
    ####################
    peft_config = None
    if model_args.use_peft:
        # get_peft_config() returns a LoraConfig based on your YAML
        peft_config = get_peft_config(model_args)
        from peft import get_peft_model
        logger.info("*** Applying LoRA ***")
        # Wrap the base model with LoRA
        model = get_peft_model(base_model, peft_config)
        # Optionally, print how many parameters are trainable
        model.print_trainable_parameters()
    else:
        # If not using LoRA, we just train the base model
        model = base_model

    # Additional transformations or chat template logic can happen here if needed

    #######################
    # Prepare SFT Trainer
    #######################
    training_args.model_init_kwargs = model_kwargs
    training_args.eval_strategy = "no"  # or "steps", etc. if you want periodic eval
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=...
        tokenizer=tokenizer,
        peft_config=peft_config,  # If None, no LoRA
    )

    # Start training
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Save model
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    ############
    # Model card
    ############
    kwargs = {
        "model_name": model_args.model_name_or_path,
        "dataset_name": list(data_args.dataset_mixer.keys()),
        "tags": ["sft"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # Possibly push to hub
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
