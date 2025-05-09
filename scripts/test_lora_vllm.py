from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

HF_LORA_PATH = "OpenBabylon/MamayLM-ORPO-align-lora64"
sql_lora_path = snapshot_download(repo_id=HF_LORA_PATH)
max_num_seqs = 2

import os
print(sql_lora_path)
print(os.listdir(sql_lora_path))

import json

def load_basemodel_name(lora_path):
    config = json.load(
        open(os.path.join(lora_path, "adapter_config.json"), 'r')
            )
    return config['base_model_name_or_path'], config['r']

base_model_name_or_path, rank = load_basemodel_name(sql_lora_path)

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest



llm = LLM(model=base_model_name_or_path, 
        max_lora_rank=rank,
        max_num_seqs=max_num_seqs,
        enable_lora=True)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=10
)

prompts = [{'role': 'user', 'content': 'Hi, my name is Verka Serducka, and what is yours ?'}]
outputs = llm.chat(
    prompts,
    sampling_params,
    lora_request=LoRARequest("lora_adapter", 1, sql_lora_path)
)

print(outputs[0])

# vllm serve meta-llama/Llama-2-7b-hf \
#    --enable-lora \
#    --lora-modules sql-lora=$HOME/.cache/huggingface/hub/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c/


