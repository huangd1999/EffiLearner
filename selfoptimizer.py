import json
import argparse
import os
import json
from tqdm import tqdm
import copy
import openai
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import tiktoken
import time
from code_efficiency_calculator import calculate_code_execution_efficiency
from tqdm import tqdm

# Setting API parameters
openai.api_base = "https://api.aiohub.org/v1"
openai.api_key = 'API KEY'


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


# Function to fetch completion
def fetch_completion(data_entry, model):
    if "small_test_cases" not in data_entry.keys():
        return data_entry
    if "overhead" not in data_entry.keys():
        overhead = "The code execution failed."
        test_case = data_entry["small_test_cases"]
        completion = data_entry["completion"]
        task_description = data_entry["markdown_description"]
    else:
        test_case = data_entry["small_test_cases"]
        completion = data_entry["completion"]
        task_description = data_entry["markdown_description"]
        overhead = data_entry["overhead"]
    prompt = prompt_construction(task_description, test_case, completion, overhead)
    current_try = 0
    while True:
        current_try+=1
        try:
            completions = openai.ChatCompletion.create(
                model=model,
                stream=False,
                messages=[
                    {"role": "system", "content": "You are a code developer expert."},
                    {"role": "user", "content": prompt},
                ],
                request_timeout=100,
            )
            data_entry["tmp_completion"] = completions.choices[0]["message"]["content"]

        except Exception as e:
            # print(repr(e))
            time.sleep(10)
            data_entry["tmp_completion"] = ""
        if data_entry["tmp_completion"] != "":
            break
        if current_try>10:
            break
    return data_entry

# ,"gpt-3.5-turbo-0613","gpt-3.5-turbo-1106","gpt-4-turbo-preview","gpt-4","claude-3-haiku","claude-3-sonnet"
# model_list = ["gpt-4-turbo-preview","gpt-3.5-turbo-0301","gpt-3.5-turbo-0613","gpt-3.5-turbo-1106","gpt-4"]
model_list = ["gpt-3.5-turbo-0301"]
for model in model_list:
    overhead_dict = {
        "overhead": [],
        "memory_usage": [],
        "execution_time": [],
        "max_memory_peak": [],
        "correct": [],
    }
    with open(f"./results/leetcode_{model}.json", "r") as f:
        dataset = json.load(f)


    for i in tqdm(range(len(dataset))):
        overhead, memory_usage, execution_time, max_memory_peak,executable = calculate_code_execution_efficiency(dataset[i],evaluation_code=True)
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
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_entry = {executor.submit(fetch_completion, copy.deepcopy(entry), model): entry for entry in tqdm(dataset)}
            for future in tqdm(concurrent.futures.as_completed(future_to_entry)):
                entry = future_to_entry[future]
                try:
                    updated_entry = future.result()
                    idx = dataset.index(entry)
                    dataset[idx] = updated_entry
                except Exception as e:
                    print(repr(e))

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

        with open(f"./results/leetcode_{model}_{current_epoch}.json", "w") as f:
            json.dump(dataset, f, indent=4)
    with open(f"./results/overhead_{model}.json", "w") as f:
        json.dump(overhead_dict, f, indent=4)