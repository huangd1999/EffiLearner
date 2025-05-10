import json
import openai
import argparse
import os
import json
from tqdm import tqdm
import copy
import openai
from concurrent.futures import ThreadPoolExecutor,as_completed
import concurrent.futures
import tiktoken
import time
from code_efficiency_calculator import calculate_code_execution_efficiency
from tqdm import tqdm

with open("./dataset.json", "r") as f:
    leetcode = json.load(f)


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

def calculate_metrics(entry):
    overhead, memory_usage, execution_time, max_memory_peak, executable = calculate_code_execution_efficiency(entry)
    return overhead, memory_usage, execution_time, max_memory_peak, executable


# Function to fetch completion
def fetch_completion(data_entry, model):
    if "small_test_cases" not in data_entry.keys():
        return data_entry
    if "overhead" not in data_entry.keys():
        overhead = "The code execution failed."
    else:
        overhead = data_entry["overhead"]
    test_case = data_entry["small_test_cases"]
    completion = data_entry["completion"]
    task_description = data_entry["markdown_description"]
    prompt = prompt_construction(task_description, test_case, completion, overhead)
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
        print(repr(e))
        data_entry["tmp_completion"] = "API Error"
    return data_entry

def update_dataset_entry(entry):
    if "tmp_completion" not in entry.keys():
        overhead, memory_usage, execution_time, max_memory_peak, executable = calculate_code_execution_efficiency(entry)
        if executable:
            return {
                "update": True,
                "data": (overhead, memory_usage, execution_time, max_memory_peak, executable)
            }
    else:
        original_completion = entry["completion"]
        entry["completion"] = entry["tmp_completion"]
        overhead, memory_usage, execution_time, max_memory_peak, executable = calculate_code_execution_efficiency(entry)
        if "memory_usage" not in entry.keys():
            return {
                "update": True,
                "data": (overhead, memory_usage, execution_time, max_memory_peak, executable)
            }
        elif executable and memory_usage < entry["memory_usage"]:
            return {
                "update": True,
                "data": (overhead, memory_usage, execution_time, max_memory_peak, executable)
            }
        else:
            entry["completion"] = original_completion
    return {"update": False}


model_list = ["gpt-4"]
overhead_dict = {
    "overhead": [],
    "memory_usage": [],
    "execution_time": [],
    "max_memory_peak": [],
    "correct": []
}

for model in model_list:
    with open(f"./EffiBench_{model}.json", "r") as f:
        dataset = json.load(f)
    for i in range(len(dataset)):
        dataset[i]["small_test_cases"] = leetcode[i]["small_test_cases"]
        dataset[i]["test_case"] = leetcode[i]["test_case"]

    for i in range(5):
        total_memory_usage = 0
        total_execution_time = 0
        total_max_memory_peak = 0
        correct = 0
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(update_dataset_entry, entry): entry for entry in dataset}

            for future in as_completed(futures):
                entry = futures[future]
                result = future.result()
                if result["update"]:

                    overhead, memory_usage, execution_time, max_memory_peak, executable = result["data"]
                    entry["overhead"] = overhead
                    entry["memory_usage"] = memory_usage
                    entry["execution_time"] = execution_time
                    entry["max_memory_peak"] = max_memory_peak
                    entry["executable"] = executable
                    correct += 1
                    total_memory_usage += memory_usage
                    total_execution_time += execution_time
                    total_max_memory_peak += max_memory_peak
                    if "tmp_completion" in entry.keys():
                        entry["completion"] = entry["tmp_completion"]
                else:
                    if "executable" in entry.keys() and entry["executable"]:
                        correct+=1
                        total_memory_usage += entry["memory_usage"]
                        total_execution_time += entry["execution_time"]
                        total_max_memory_peak += entry["max_memory_peak"]
                        if "tmp_completion" in entry.keys():
                            entry["tmp_completion"] = entry["completion"]

        if correct > 0:
        # Calculate and update the overall metrics only if there are any correct entries
            total_overhead = f"""
The total memory usage during the code execution is: {round(total_memory_usage/correct, 2)} MB*s.
The total execution time is: {round(total_execution_time/correct, 2)} s.
The maximum memory peak requirement is: {round(total_max_memory_peak/correct, 2)} MB.
"""
            overhead_dict["overhead"].append(total_overhead)
            overhead_dict["memory_usage"].append(round(total_memory_usage/correct, 2))
            overhead_dict["execution_time"].append(round(total_execution_time/correct, 2))
            overhead_dict["max_memory_peak"].append(round(total_max_memory_peak/correct, 2))
            overhead_dict["correct"].append(correct)
        else:
            # Handle case with no correct entries
            print("No correct entries to calculate overall metrics.")
        print("correct",correct)
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
    correct = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(update_dataset_entry, entry): entry for entry in dataset}
        
        for future in as_completed(futures):
            entry = futures[future]
            result = future.result()
            if result["update"]:
                overhead, memory_usage, execution_time, max_memory_peak, executable = result["data"]
                entry["overhead"] = overhead
                entry["memory_usage"] = memory_usage
                entry["execution_time"] = execution_time
                entry["max_memory_peak"] = max_memory_peak
                entry["executable"] = executable
                if "tmp_completion" in entry.keys():
                    entry["completion"] = entry["tmp_completion"]
                correct += 1
                total_memory_usage += memory_usage
                total_execution_time += execution_time
                total_max_memory_peak += max_memory_peak
            else:
                if "executable" in entry.keys() and entry["executable"]:
                    correct+=1
                    total_memory_usage += entry["memory_usage"]
                    total_execution_time += entry["execution_time"]
                    total_max_memory_peak += entry["max_memory_peak"]
                    if "tmp_completion" in entry.keys():
                        entry["tmp_completion"] = entry["completion"]


    if correct > 0:
        # Calculate and update the overall metrics only if there are any correct entries
        total_overhead = f"""
The total memory usage during the code execution is: {round(total_memory_usage/correct, 2)} MB*s.
The total execution time is: {round(total_execution_time/correct, 2)} s.
The maximum memory peak requirement is: {round(total_max_memory_peak/correct, 2)} MB.
"""
        overhead_dict["overhead"].append(total_overhead)
        overhead_dict["memory_usage"].append(round(total_memory_usage/correct, 2))
        overhead_dict["execution_time"].append(round(total_execution_time/correct, 2))
        overhead_dict["max_memory_peak"].append(round(total_max_memory_peak/correct, 2))
        overhead_dict["correct"].append(correct)

    print(overhead_dict)
    with open(f"./leetcode_{model}_subset.json", "w") as f:
        json.dump(dataset, f, indent=4)
    with open(f"./overhead_{model}_subset.json", "w") as f:
        json.dump(overhead_dict, f, indent=4)
    print(f"Model {model} is done")