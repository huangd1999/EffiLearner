from transformers import T5ForConditionalGeneration, AutoTokenizer,GPTNeoForCausalLM,AutoModelForCausalLM,AutoModel, AutoModelForSeq2SeqLM
# from vllm import LLM, SamplingParams
import torch
import json
import tiktoken
from tqdm import tqdm
import markdownify
import os

# checkpoints = ["m-a-p/OpenCodeInterpreter-DS-1.3B", "m-a-p/OpenCodeInterpreter-DS-6.7B", "m-a-p/OpenCodeInterpreter-DS-33B","deepseek-ai/deepseek-coder-1.3b-instruct","deepseek-ai/deepseek-coder-6.7b-instruct","deepseek-ai/deepseek-coder-33b-instruct","codellama/CodeLlama-7b-Instruct-hf","codellama/CodeLlama-13b-Instruct-hf","codellama/CodeLlama-34b-Instruct-hf","codellama/CodeLlama-70b-Instruct-hf","Xwin-LM/XwinCoder-7B","Xwin-LM/XwinCoder-13B","Xwin-LM/XwinCoder-34B","TheBloke/WizardCoder-Python-7B-V1.0-GGUF","TheBloke/WizardCoder-Python-13B-V1.0-GGUF","TheBloke/WizardCoder-Python-34B-V1.0-GGUF","bigcode/starcoder2-3b","bigcode/starcoder2-7b","bigcode/starcoder2-15b"]
checkpoints = [
    "m-a-p/OpenCodeInterpreter-DS-33B",
    "NTQAI/Nxcode-CQ-7B-orpo",
    "Qwen/CodeQwen1.5-7B-Chat",
    "codefuse-ai/CodeFuse-DeepSeek-33B",
    "deepseek-ai/deepseek-coder-33b-instruct",
    "Artigenz/Artigenz-Coder-DS-6.7B",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "m-a-p/OpenCodeInterpreter-DS-6.7B",
    "Phind/Phind-CodeLlama-34B-v2",
    "Phind/Phind-CodeLlama-34B-v1",
    "Phind/Phind-CodeLlama-34B-Python-v1",
    "Qwen/CodeQwen1.5-7B",
    "codellama/CodeLlama-70b-Instruct-hf",
    "codellama/CodeLlama-70b-hf",
    "deepseek-ai/deepseek-coder-33b-base",
    "codellama/CodeLlama-70b-Python-hf",
    "bigcode/starcoder2-15b",
    "codellama/CodeLlama-34b-Instruct-hf",
    "deepseek-ai/deepseek-coder-6.7b-base",
    "codellama/CodeLlama-34b-hf",
    "codellama/CodeLlama-34b-Python-hf",
    "WizardLMTeam/WizardCoder-15B-V1.0",
    "codellama/CodeLlama-13b-Instruct-hf",
    "google/codegemma-7b",
    "codellama/CodeLlama-13b-hf",
    "codellama/CodeLlama-13b-Python-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
    "bigcode/starcoder2-7b",
    "google/codegemma-7b-it",
    "WisdomShell/CodeShell-7B",
    "codellama/CodeLlama-7b-hf",
    "bigcode/octocoder",
    "codellama/CodeLlama-7b-Python-hf",
    "bigcode/starcoder",
    "bigcode/starcoderbase",
    "bigcode/starcoder2-3b",
    "THUDM/codegeex2-6b",
    "bigcode/starcoderbase-7b",
    "bigcode/octogeex",
    "Salesforce/codegen25-7b-multi_P",
    "google/codegemma-2b",
    "smallcloudai/Refact-1_6B-fim",
    "stabilityai/stable-code-3b",
    "deepseek-ai/deepseek-coder-1.3b-base",
    "bigcode/starcoderbase-3b",
    "replit/replit-code-v1-3b",
    "Salesforce/codegen25-7b-mono_P",
    "bigcode/starcoderbase-1b",
    "Salesforce/codegen-16B-multi",
    "stabilityai/stablecode-completion-alpha-3b",
    "Deci/DeciCoder-1b",
    "microsoft/phi-1",
    "bigcode/santacoder",
]

batch_size = 16
def construct_prompt_template(inputs,checkpoint,model,tokenizer):
    device = "cuda"

    tokenizer.pad_token = tokenizer.eos_token
    input_tokens = tokenizer.batch_encode_plus(
    inputs,
    padding=True,
    return_tensors="pt",
    
    ).to(model.device)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(model.device)
    # input_tokens.pop("token_type_ids")
    try:
        sequences = model.generate(
        **input_tokens, max_new_tokens=512, do_sample=True
        )
        generated_texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        for i in range(len(generated_texts)):
            if inputs[i] in generated_texts[i]:
                generated_texts[i] = generated_texts[i].replace(inputs[i], "")
    except:
        generated_texts = ["" for i in range(len(inputs))]

    return generated_texts


# Function to fetch completion
def fetch_completion(data_entry_lists, model,checkpoint,tokenizer):
    return_batchs = data_entry_lists
    inputs_batchs = []
    for data_entry in data_entry_lists:

        test_case = data_entry["small_test_cases"]
        inputs_batchs.append(f"Please complete Python code based on the task description and test cases. # Task description:\n{data_entry['markdown_description']}\n{test_case}\n#Solution:\n")

    completion_lists = construct_prompt_template(inputs_batchs,checkpoint,model,tokenizer)
    for i in range(len(data_entry_lists)):
        data_entry_lists[i]["completion"] = completion_lists[i]

    return data_entry_lists

for checkpoint in checkpoints:
    print(checkpoint)
    end_name = checkpoint.split("/")[-1]
    with open("./dataset.json", "r") as f:
        dataset = json.load(f)


    model = AutoModelForCausalLM.from_pretrained(checkpoint,device_map = "auto",trust_remote_code=True,torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,trust_remote_code=True)

    for i in tqdm(range(0,len(dataset),batch_size)):
        dataset[i:i+batch_size] = fetch_completion(dataset[i:i+batch_size],model,checkpoint,tokenizer)

    end_name = checkpoint.split("/")[-1]
    with open(f"./results/leetcode_{end_name}.json", "w") as f:
        json.dump(dataset, f, indent=4)
