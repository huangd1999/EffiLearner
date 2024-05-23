from transformers import T5ForConditionalGeneration, AutoTokenizer,GPTNeoForCausalLM,AutoModelForCausalLM,AutoModel, AutoModelForSeq2SeqLM
# from vllm import LLM, SamplingParams
import torch
import json
import tiktoken
from tqdm import tqdm
import markdownify
from datasets import load_dataset

batch_size = 8
# "TheBloke/WizardCoder-33B-V1.1-GGUF",
checkpoints = ["m-a-p/OpenCodeInterpreter-DS-1.3B", "m-a-p/OpenCodeInterpreter-DS-6.7B", "m-a-p/OpenCodeInterpreter-DS-33B","deepseek-ai/deepseek-coder-1.3b-instruct","deepseek-ai/deepseek-coder-6.7b-instruct","deepseek-ai/deepseek-coder-33b-instruct","codellama/CodeLlama-7b-Instruct-hf","codellama/CodeLlama-13b-Instruct-hf","codellama/CodeLlama-34b-Instruct-hf","codellama/CodeLlama-70b-Instruct-hf","Xwin-LM/XwinCoder-7B","Xwin-LM/XwinCoder-13B","Xwin-LM/XwinCoder-34B","TheBloke/WizardCoder-Python-7B-V1.0-GPTQ","TheBloke/WizardCoder-Python-13B-V1.0-GPTQ","TheBloke/WizardCoder-Python-34B-V1.0-GPTQ","bigcode/starcoder2-3b","bigcode/starcoder2-7b","bigcode/starcoder2-15b"]
# "Qwen/CodeQwen1.5-7B","deepseek-ai/deepseek-coder-6.7b-instruct","m-a-p/OpenCodeInterpreter-DS-6.7B","Artigenz/Artigenz-Coder-DS-6.7B","ise-uiuc/Magicoder-S-DS-6.7B","Nondzu/Mistral-7B-codealpaca-lora","uukuguy/speechless-starcoder2-15b"
# 
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
    inputs_batchs = []
    for data_entry in data_entry_lists:
        inputs_batchs.append(f"Please complete Python code based on the task description. # Task description:\n{data_entry['prompt']}\n#Solution:\n")

    completion_lists = construct_prompt_template(inputs_batchs,checkpoint,model,tokenizer)
    for i in range(len(data_entry_lists)):
        data_entry_lists[i]["completion"] = completion_lists[i]

    return data_entry_lists

for checkpoint in checkpoints:
    # with open("./dataset.json", "r") as f:
    #     dataset = json.load(f)
    dataset = load_dataset("evalplus/humanevalplus",split="test")
    print(checkpoint)
    dataset = [entry for entry in dataset]
    print(dataset[0].keys())

    model = AutoModelForCausalLM.from_pretrained(checkpoint,device_map = "auto",trust_remote_code=True,torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,trust_remote_code=True)

    for i in tqdm(range(0,len(dataset),batch_size)):
        if i+batch_size > len(dataset):
            dataset[i:] = fetch_completion(dataset[i:],model,checkpoint,tokenizer)
        else:
            dataset[i:i+batch_size] = fetch_completion(dataset[i:i+batch_size],model,checkpoint,tokenizer)

    end_name = checkpoint.split("/")[-1]
    with open(f"./results/humaneval_{end_name}.json", "w") as f:
        json.dump(dataset, f, indent=4)
