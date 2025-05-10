import json
import os
import glob
import numpy as np


def calculate_memory_usage(dat_file_path):
    with open(dat_file_path, 'r') as file:
        prev_time = 0
        prev_mem_mb = 0
        mem_time_mb_s = 0
        next(file)
        for line in file:
            if "__main__." in line:
                continue
            parts = line.split()
            mem_in_mb = float(parts[1])
            timestamp = float(parts[2])
            if prev_time > 0:
                time_interval_s = timestamp - prev_time
                mem_time_mb_s += (prev_mem_mb + mem_in_mb) / 2 * time_interval_s
            prev_time = timestamp
            prev_mem_mb = mem_in_mb
        return mem_time_mb_s


def calculate_runtime(dat_file_path):
    with open(dat_file_path, 'r') as file:
        start_time = float("inf")
        end_time = float("-inf")
        next(file)
        for line in file:
            if "__main__." in line:
                continue
            parts = line.split()
            timestamp = float(parts[2])
            start_time = min(start_time, timestamp)
            end_time = max(end_time, timestamp)
        return max(end_time - start_time, 0)


def report_max_memory_usage(dat_file_path):
    max_memory_usage = 0
    with open(dat_file_path, 'r') as file:
        next(file)
        for line in file:
            if "__main__." in line:
                continue
            parts = line.split()
            mem_in_mb = float(parts[1])
            max_memory_usage = max(max_memory_usage, mem_in_mb)
        return max_memory_usage


steps = 5
models = ["m-a-p/OpenCodeInterpreter-DS-1.3B","deepseek-ai/deepseek-coder-1.3b-instruct","codellama/CodeLlama-7b-Instruct-hf","bigcode/starcoder2-15b","gpt-3.5-turbo-0301","claude-3-sonnet"]
canonical_solution_directory = "./dat_results/canonical_solution"
canonical_solution_memory_usage = {}
canonical_solution_execution_time = {}
canonical_solution_max_memory_usage = {}
for dat_file in glob.glob(os.path.join(canonical_solution_directory, "*.dat")):
    try:
        problem_idx = os.path.basename(dat_file).split('.')[0]
        canonical_solution_memory_usage[int(problem_idx)] = calculate_memory_usage(dat_file)
        canonical_solution_execution_time[int(problem_idx)] = calculate_runtime(dat_file)
        canonical_solution_max_memory_usage[int(problem_idx)] = report_max_memory_usage(dat_file)
    except:
        pass

global_result = {}


common_problem_idxs = {}
for model in models:
    if "/" in model:
        model = model.split("/")[1]
    problem_idxs = set()
    for step in range(1,steps + 1):
        model_step = f"{model}_{step}"

        dat_directory = f"./rebuttal/leetcode_{model_step}_variation"
        step_problem_idxs = set()
        for dat_file in glob.glob(os.path.join(dat_directory, "*.dat")):
            problem_idx = os.path.basename(dat_file).split('.')[0]
            step_problem_idxs.add(int(problem_idx))
        if step == 1:
            problem_idxs = step_problem_idxs
        else:
            problem_idxs = problem_idxs.intersection(step_problem_idxs)
    common_problem_idxs[model] = problem_idxs

import traceback

for model in models:
    if "/" in model:
        model = model.split("/")[1]
    prev_step_result = {}
    for step in range(1,steps + 1):
        model_step = f"{model}_{step}"
        completion_memory_usage = {}
        execution_time = {}
        max_memory_usage = {}
        task_idx = {}
        dat_directory = f"./rebuttal/leetcode_{model_step}_variation"
        print(dat_directory)
        for dat_file in glob.glob(os.path.join(dat_directory, "*.dat")):
            try:
                problem_idx = os.path.basename(dat_file).split('.')[0]
                if int(problem_idx) not in common_problem_idxs[model]:
                    continue
                current_memory_usage = calculate_memory_usage(dat_file)
                current_execution_time = calculate_runtime(dat_file)
                current_max_memory_usage = report_max_memory_usage(dat_file)
                completion_memory_usage[int(problem_idx)] = current_memory_usage
                execution_time[int(problem_idx)] = current_execution_time
                max_memory_usage[int(problem_idx)] = current_max_memory_usage
                task_idx[int(problem_idx)] = dat_file

                prev_step_result[problem_idx] = {
                    "completion_memory_usage": completion_memory_usage[int(problem_idx)],
                    "execution_time": execution_time[int(problem_idx)],
                    "max_memory_usage": max_memory_usage[int(problem_idx)],
                }
            except Exception as e:
                print(f"Error processing {dat_file}:")
                print(traceback.format_exc())
        global_result[model_step] = {
            "completion_memory_usage": completion_memory_usage,
            "execution_time": execution_time,
            "max_memory_usage": max_memory_usage,
            "task_idx": task_idx
        }
        print(len(global_result[model_step]["completion_memory_usage"]))

for model in models:
    if "/" in model:
        model = model.split("/")[1]
    variants_results = []
    for step in range(1,6):
        model_step = f"{model}_{step}"
        completion_memory_usage = global_result[model_step]["completion_memory_usage"]
        execution_time = global_result[model_step]["execution_time"]
        max_memory_usage = global_result[model_step]["max_memory_usage"]

        total_execution_time = 0
        normalized_execution_time = 0
        total_max_memory_usage = 0
        normalized_max_memory_usage = 0
        total_memory_usage = 0
        total_canonical_solution_max_memory_usage = 0
        total_canonical_solution_execution_time = 0
        total_canonical_solution_memory_usage = 0
        normalized_memory_usage = 0
        total_codes = 0
        normalized_execution_time_list = []
        normalized_max_memory_usage_list = []
        normalized_memory_usage_list = []

        for idx in completion_memory_usage.keys():
            if idx not in canonical_solution_memory_usage.keys():
                continue

            total_memory_usage += completion_memory_usage[idx]
            total_execution_time += execution_time[idx]
            total_max_memory_usage += max_memory_usage[idx]
            total_canonical_solution_max_memory_usage += canonical_solution_max_memory_usage[idx]
            total_canonical_solution_memory_usage += canonical_solution_memory_usage[idx]
            total_canonical_solution_execution_time += canonical_solution_execution_time[idx]

            normalized_execution_time += execution_time[idx] / canonical_solution_execution_time[idx]
            normalized_execution_time_list.append(execution_time[idx] / canonical_solution_execution_time[idx])

            normalized_max_memory_usage += max_memory_usage[idx] / canonical_solution_max_memory_usage[idx]
            normalized_max_memory_usage_list.append(max_memory_usage[idx] / canonical_solution_max_memory_usage[idx])

            normalized_memory_usage += completion_memory_usage[idx] / canonical_solution_memory_usage[idx]
            normalized_memory_usage_list.append(completion_memory_usage[idx] / canonical_solution_memory_usage[idx])

            total_codes += 1

        if len(normalized_execution_time_list) == 0:

            continue

        normalized_execution_time = total_execution_time / total_canonical_solution_execution_time
        normalized_max_memory_usage = total_max_memory_usage / total_canonical_solution_max_memory_usage
        normalized_memory_usage = total_memory_usage / total_canonical_solution_memory_usage
        total_execution_time = total_execution_time / len(normalized_execution_time_list)
        total_memory_usage = total_memory_usage / len(normalized_execution_time_list)
        total_max_memory_usage = total_max_memory_usage / len(normalized_execution_time_list)


        variants_results.append([total_execution_time,normalized_execution_time,total_max_memory_usage,normalized_max_memory_usage,total_memory_usage,normalized_memory_usage])

    results = []
    if len(variants_results) == 0:
        continue


    for i in range(len(variants_results[0])):
        values = []
        for j in range(len(variants_results)):
            values.append(variants_results[j][i])

        mean = np.mean(values)
        std_dev = np.std(values)
        coef_of_var = (std_dev / mean) * 100
        results.append(coef_of_var)
    print(f"{model}&{results[0]:.1f}&{results[1]:.1f}&{results[2]:.1f}&{results[3]:.1f}&{results[4]:.1f}&{results[5]:.1f}\\\\")
