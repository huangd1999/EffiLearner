import json
import argparse
import os
import json
from tqdm import tqdm
import copy
from datasets import load_dataset
import openai
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import tiktoken
import time
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer,GPTNeoForCausalLM,AutoModelForCausalLM,AutoModel, AutoModelForSeq2SeqLM
from code_efficiency_calculator import calculate_code_execution_efficiency
from tqdm import tqdm

batch_size = 8
def prompt_construction(task_description, test_case, completion, overhead_prompt):
    prompt = f"""
Optimize the efficiency of the following Python code based on the task, test case, and overhead analysis provided. Ensure the optimized code can pass the given test case.

Task Description:
{task_description}

Test Case:
{test_case}

Original Code:
```python
{completion}
```

Overhead Analysis:
{overhead_prompt}

Optimization Rules:
- Encapsulate the optimized code within a Python code block (i.e., ```python\n[Your Code Here]\n```).
- Do not include the test case within the code block.
- Focus solely on code optimization; test cases are already provided.
- Ensure the provided test case passes with your optimized solution.
"""
    return prompt


def construct_prompt_template(inputs,model,tokenizer):

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
def fetch_completion(data_entry_lists, model,tokenizer):

    inputs_batchs = []
    for data_entry in data_entry_lists:
        if "overhead" not in data_entry.keys():
            overhead = "The code execution failed."
            test_case = "\n".join(data_entry["test_list"])
            completion = data_entry["completion"]
            task_description = data_entry["prompt"]
        else:
            test_case = "\n".join(data_entry["test_list"])
            completion = data_entry["completion"]
            task_description = data_entry["prompt"]
            overhead = data_entry["overhead"]
        prompt = prompt_construction(task_description, test_case, completion, overhead)
        inputs_batchs.append(prompt)

    completion_lists = construct_prompt_template(inputs_batchs,model,tokenizer)
    for i in range(len(data_entry_lists)):
        data_entry_lists[i]["tmp_completion"] = completion_lists[i]
    return data_entry_lists

checkpoints = ["m-a-p/OpenCodeInterpreter-DS-1.3B", "m-a-p/OpenCodeInterpreter-DS-6.7B", "m-a-p/OpenCodeInterpreter-DS-33B","deepseek-ai/deepseek-coder-1.3b-instruct","deepseek-ai/deepseek-coder-6.7b-instruct","deepseek-ai/deepseek-coder-33b-instruct","codellama/CodeLlama-7b-Instruct-hf","codellama/CodeLlama-13b-Instruct-hf","codellama/CodeLlama-34b-Instruct-hf","codellama/CodeLlama-70b-Instruct-hf","Xwin-LM/XwinCoder-7B","Xwin-LM/XwinCoder-13B","Xwin-LM/XwinCoder-34B","TheBloke/WizardCoder-Python-7B-V1.0-GPTQ","TheBloke/WizardCoder-Python-13B-V1.0-GPTQ","TheBloke/WizardCoder-Python-34B-V1.0-GPTQ","bigcode/starcoder2-3b","bigcode/starcoder2-7b","bigcode/starcoder2-15b"]
for checkpoint in checkpoints:
    print(checkpoint)
    model_name = checkpoint.split("/")[-1]
    file_path = f"./results/mbpp_overhead_{model_name}.json"
    if os.path.exists(file_path):
        continue
    model = AutoModelForCausalLM.from_pretrained(checkpoint,device_map = "auto",trust_remote_code=True,torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,trust_remote_code=True)
    overhead_dict = {
        "overhead": [],
        "memory_usage": [],
        "execution_time": [],
        "max_memory_peak": [],
        "correct": [],
    }
    with open(f"./results/mbpp_{model_name}.json", "r") as f:
        dataset = json.load(f)

    for i in range(len(dataset)):
        dataset[i]["dataset"] = "mbpp"
    for i in tqdm(range(len(dataset))):
        overhead, memory_usage, execution_time, max_memory_peak,executable = calculate_code_execution_efficiency(dataset[i],evaluation_code=True,path="./mbpp/")
        if executable:
            dataset[i]["overhead"] = overhead
            dataset[i]["memory_usage"] = memory_usage
            dataset[i]["execution_time"] = execution_time
            dataset[i]["max_memory_peak"] = max_memory_peak
            dataset[i]["executable"] = executable


    dataset = [entry for entry in dataset if "executable" in entry.keys() and entry["executable"]]

    total_memory_usage = 0
    total_execution_time = 0
    total_max_memory_peak = 0
    normalize_total_memory_usage = 0
    normalize_total_execution_time = 0
    normalize_total_max_memory_peak = 0
    correct = 0
    for i in tqdm(range(len(dataset))):
        if "executable" in dataset[i].keys() and dataset[i]["executable"]:
            total_memory_usage += dataset[i]["memory_usage"]
            total_execution_time += dataset[i]["execution_time"]
            total_max_memory_peak += dataset[i]["max_memory_peak"]

            correct+=1
    if correct==0:
        correct+=1
    total_overhead = f"""
The total memory usage during the code execution is: {round(total_memory_usage/correct,2)} MB*s.
The total execution time is: {round(total_execution_time/correct,2)} s.
The maximum memory peak requirement is: {round(total_max_memory_peak/correct,2)} MB.
"""
    overhead_dict["overhead"].append(total_overhead)
    overhead_dict["memory_usage"].append(round(total_memory_usage/correct,2))
    overhead_dict["execution_time"].append(round(total_execution_time/correct,2))
    overhead_dict["max_memory_peak"].append(round(total_max_memory_peak/correct,2))
    overhead_dict["correct"].append(correct)

    epoch = 5

    for current_epoch in range(1, epoch+1):
        for i in tqdm(range(0,len(dataset),batch_size)):
            dataset[i:i+batch_size] = fetch_completion(dataset[i:i+batch_size],model,tokenizer)

        total_memory_usage = 0
        total_execution_time = 0
        total_max_memory_peak = 0
        normalize_total_memory_usage = 0
        normalize_total_execution_time = 0
        normalize_total_max_memory_peak = 0
        correct = 0
        for i in range(len(dataset)):
            tmp_code = dataset[i]["completion"]
            dataset[i]["completion"] = dataset[i]["tmp_completion"]
            overhead, memory_usage, execution_time, max_memory_peak,executable = calculate_code_execution_efficiency(dataset[i],evaluation_code=True)
            if (("memory_usage" not in dataset[i].keys()) or (memory_usage < dataset[i]["memory_usage"])) and executable:
                dataset[i]["memory_usage"] = memory_usage
                dataset[i]["execution_time"] = execution_time
                dataset[i]["max_memory_peak"] = max_memory_peak
                dataset[i]["overhead"] = overhead
                dataset[i]["executable"] = executable

            else:
                dataset[i]["completion"] = tmp_code
            if "executable" in dataset[i].keys() and dataset[i]["executable"]:
                total_memory_usage += dataset[i]["memory_usage"]
                total_execution_time += dataset[i]["execution_time"]
                total_max_memory_peak += dataset[i]["max_memory_peak"]

                correct+=1
        if correct==0:
            correct+=1
        total_overhead = f"""
The total memory usage during the code execution is: {round(total_memory_usage/correct,2)} MB*s.
The total execution time is: {round(total_execution_time/correct,2)} s.
The maximum memory peak requirement is: {round(total_max_memory_peak/correct,2)} MB.
"""
        overhead_dict["overhead"].append(total_overhead)
        overhead_dict["memory_usage"].append(round(total_memory_usage/correct,2))
        overhead_dict["execution_time"].append(round(total_execution_time/correct,2))
        overhead_dict["max_memory_peak"].append(round(total_max_memory_peak/correct,2))
        overhead_dict["correct"].append(correct)

        with open(f"./results/mbpp_{model_name}_{current_epoch}.json", "w") as f:
            json.dump(dataset, f, indent=4)

    with open(f"./results/mbpp_overhead_{model_name}.json", "w") as f:
        json.dump(overhead_dict, f, indent=4)