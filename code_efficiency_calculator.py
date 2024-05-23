from importlib.metadata import entry_points
import json
import os
import copy
from tqdm import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import os
import re
import shutil
ListNode_text = """
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
"""
TreeNode_text = """
class TreeNode:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

import_pkg = """
from typing import *
from bisect import *
from collections import *
from copy import *
from datetime import *
from heapq import *
from math import *
from re import *
from string import *
from random import *
from itertools import *
from functools import *
from operator import *

import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import itertools
import functools
import operator
"""

memory_profiler_prompt = r"""
def parse_profile_table(profile_table: str):
    table = {"filename": None, "rows": []}
    for line in profile_table.strip().split("\n"):
        if line.startswith("Filename:"):
            table["filename"] = line.split(": ")[1]
        elif re.match(r"^\s*\d+", line):
            parts = re.split(r"\s{2,}", line.strip(), maxsplit=4)
            if len(parts) == 5 and "iB" in parts[1] and "iB" in parts[2]:
                table["rows"].append({
                    "line": int(parts[0]),
                    "mem_usage": parts[1],
                    "increment": parts[2],
                    "occurrences": int(parts[3]),
                    "line_contents": parts[4],
                })
            else:
                parts = re.split(r"\s{2,}", line.strip(), maxsplit=1)
                table["rows"].append({
                    "line": int(parts[0]),
                    "line_contents": parts[1] if len(parts) == 2 else "",
                })
    return table

def print_averaged_results(profile_log: str, precision: int = 1):
    tables = [parse_profile_table(table) for table in profile_log.split("\n\n\n")]
    averaged_table = defaultdict(lambda: defaultdict(list))

    for table in tables:
        filename = table["filename"]
        for row in table["rows"]:
            line = row["line"]
            if "mem_usage" in row:
                mem_usage = float(row["mem_usage"].split()[0])
                increment = float(row["increment"].split()[0])
                occurrences = row["occurrences"]
                averaged_table[filename][line].append((mem_usage, increment, occurrences))
            else:
                averaged_table[filename][line].append(tuple())

    stream = sys.stdout
    template = '{0:>6} {1:>12} {2:>12}  {3:>10}   {4:<}'

    for filename, lines in averaged_table.items():
        header = template.format('Line #', 'Mem usage', 'Increment', 'Occurrences', 'Line Contents')

        stream.write(u'Filename: ' + filename + '\n\n')
        stream.write(header + u'\n')
        stream.write(u'=' * len(header) + '\n')

        all_lines = linecache.getlines(filename)

        float_format = u'{0}.{1}f'.format(precision + 4, precision)
        template_mem = u'{0:' + float_format + '} MiB'

        for lineno, mem_values in lines.items():
            # TODO: should average the rest or not?
            # mem_values = [(50.1, 0.0, 4), (51.1, 0.0, 6), ()]
            if any([len(m) == 0 for m in mem_values]):
                tmp = template.format(lineno, "", "", "", all_lines[lineno - 1])
            else:
                mem_usage_sum = sum(m[0] for m in mem_values)
                increment_sum = sum(m[1] for m in mem_values)
                occurrences_sum = sum(m[2] for m in mem_values)
                count = len(mem_values)

                avg_mem_usage = mem_usage_sum / count
                avg_increment = increment_sum / count
                avg_occurrences = occurrences_sum / count

                avg_mem_usage_str = template_mem.format(avg_mem_usage)
                avg_increment_str = template_mem.format(avg_increment)

                tmp = template.format(lineno, avg_mem_usage_str, avg_increment_str, int(avg_occurrences), all_lines[lineno - 1])
            stream.write(tmp)

print_averaged_results(profile_stream.getvalue(), precision=PROFILE_PRECISION)
"""

memory_profiler_pkgs = r"""
from collections import defaultdict, deque
from memory_profiler import profile
import io
profile_stream = io.StringIO()
PROFILE_PRECISION = 1
import re
import sys
import linecache
"""


def calculate_memory_usage(dat_file_path):
    with open(dat_file_path, 'r') as file:
        prev_time = 0
        prev_mem_mb = 0
        mem_time_mb_s = 0
        next(file)
        for line in file:
            if not line.startswith('MEM'):
                continue  # Skip any line that does not start with 'MEM'
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
            if not line.startswith('MEM'):
                continue  # Skip any line that does not start with 'MEM'
            parts = line.split()
            timestamp = float(parts[2])
            start_time = min(start_time, timestamp)
            end_time = max(end_time, timestamp)
        return max(end_time - start_time,0)

def report_max_memory_usage(dat_file_path):
    max_memory_usage = 0
    with open(dat_file_path, 'r') as file:
        prev_time = 0
        prev_mem_mb = 0
        mem_time_mb_s = 0
        next(file)
        for line in file:
            if not line.startswith('MEM'):
                continue  # Skip any line that does not start with 'MEM'
            parts = line.split()
            mem_in_mb = float(parts[1])
            max_memory_usage = max(max_memory_usage, mem_in_mb)
        return max_memory_usage

def add_profile_decorator_to_python_file(file_path,entry_point):
    """给Python文件中的函数自动添加@profile装饰器。"""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        if "humaneval" in file_path:
            with open(file_path, 'w') as file:
                inside_class = False
                class_indent = 0
                for line in lines:
                    stripped_line = line.lstrip()
                    if stripped_line.startswith(f"def {entry_point}"):
                        inside_class = True
                        class_indent = len(line) - len(stripped_line)
                        file.write('@profile\n')
                        file.write(line)
                        continue
                    if inside_class:
                        if stripped_line and not line[class_indent].isspace():
                            inside_class = False
                        elif stripped_line.startswith("def "):
                            file.write(' ' * class_indent + '@profile\n')
                    file.write(line)
        if "mbpp" in file_path:
            entry_point
            with open(file_path, 'w') as file:
                inside_class = False
                class_indent = 0
                for line in lines:
                    stripped_line = line.lstrip()
                    if stripped_line.startswith(f"def {entry_point}"):
                        inside_class = True
                        class_indent = len(line) - len(stripped_line)
                        file.write('@profile\n')
                        file.write(line)
                        continue
                    if inside_class:
                        if stripped_line and not line[class_indent].isspace():
                            inside_class = False
                        elif stripped_line.startswith("def "):
                            file.write(' ' * class_indent + '@profile\n')
                    file.write(line)
        else:
            with open(file_path, 'w') as file:
                inside_class = False
                class_indent = 0
                for line in lines:
                    stripped_line = line.lstrip()
                    if stripped_line.startswith("class Solution"):
                        inside_class = True
                        class_indent = len(line) - len(stripped_line)
                        file.write(line)
                        continue
                    if inside_class:
                        if stripped_line and not line[class_indent].isspace():
                            inside_class = False
                        elif stripped_line.startswith("def "):
                            file.write(' ' * class_indent + '    @profile\n')
                    file.write(line)
    except Exception as e:
        # print(f"Error during the file processing: {e}")
        pass

def add_profile_for_memory_profiler(code_string,data):
    """给Python代码中的函数自动添加@profile装饰器。"""
    entry_point = ""
    try:
        if "task_id" in data.keys() and "HumanEval" in data["task_id"]:
            entry_point = data["entry_point"]
            lines = code_string.split('\n')
            new_lines = []
            inside_class = False
            class_indent = 0
            first_function = True
            for line in lines:
                stripped_line = line.lstrip()
                if stripped_line.startswith(f"def {entry_point}"):
                    inside_class = True
                    class_indent = len(line) - len(stripped_line)
                    new_lines.append(' ' * class_indent + '@profile(stream=profile_stream, precision=PROFILE_PRECISION)')
                new_lines.append(line)
            return '\n'.join(new_lines)
        elif "task_id" in data.keys():
            entry_point = data["entry_point"]
            lines = code_string.split('\n')
            new_lines = []
            inside_class = False
            class_indent = 0
            first_function = True
            for line in lines:
                stripped_line = line.lstrip()
                if stripped_line.startswith(f"def {entry_point}"):
                    inside_class = True
                    class_indent = len(line) - len(stripped_line)
                    new_lines.append(' ' * class_indent + '@profile(stream=profile_stream, precision=PROFILE_PRECISION)')
                new_lines.append(line)
            return '\n'.join(new_lines)
        else:
            lines = code_string.split('\n')
            new_lines = []
            inside_class = False
            class_indent = 0
            first_function = True
            for line in lines:
                stripped_line = line.lstrip()
                if stripped_line.startswith("class Solution"):
                    inside_class = True
                    class_indent = len(line) - len(stripped_line)
                    new_lines.append(line)
                    continue
                if inside_class:
                    if stripped_line and not line[class_indent].isspace():
                        inside_class = False
                    elif stripped_line.startswith("def ") and first_function:
                        new_lines.append(' ' * class_indent + '    @profile(stream=profile_stream, precision=PROFILE_PRECISION)')
                        first_function = False
                new_lines.append(line)
            return '\n'.join(new_lines)
    except Exception as e:
        return code_string

def calculate_line_efficiency(completion_file,entry_point):
    try:
        path, filename = os.path.split(completion_file)
        tmp_py_script_filename = f"{filename.split('.')[0]}_tmp.py"
        tmp_py_script = os.path.join(path, tmp_py_script_filename)
        tmp_lprof_filename = f"{tmp_py_script_filename}.lprof"  # 期望的lprof文件名
        
        # 复制原始脚本到临时文件，并添加@profile装饰器
        subprocess.run(['cp', completion_file, tmp_py_script],check=True, capture_output=True, text=True)
        add_profile_decorator_to_python_file(tmp_py_script,entry_point)

        subprocess.run(['timeout',"10",'kernprof', '-l', tmp_py_script_filename], cwd=path, capture_output=True, text=True, check=True)
        # 生成性能报告
        overhead_dir = path
        # os.makedirs(overhead_dir, exist_ok=True)
        report_file = os.path.join(overhead_dir, tmp_py_script_filename.replace('.py', '.txt'))
        with open(report_file, 'w') as f:
            subprocess.run(['timeout',"10",'python', '-m', 'line_profiler', tmp_lprof_filename], cwd=path, stdout=f)
        with open(report_file, 'r') as f:
            report_content = f.read()
            # print(report_content)

    except subprocess.CalledProcessError as e:
        # print(f"Error during the execution: {e}")
        report_content = f"Error during the execution: {e}"

    # # 清理临时文件
    if os.path.exists(tmp_py_script):
        os.remove(tmp_py_script)
    if os.path.exists(f"{tmp_py_script}.lprof"):
        os.remove(f"{tmp_py_script}.lprof")

    return report_content

def humaneval_add_string_to_py_file(data,evaluation_code=False, path="./tmp/"):
    if "canonical_solution" in path:
        data["completion"] = data["canonical_solution"]
    if evaluation_code==False:
        test_case = data["test"]
    else: 
        test_case = data["small_test_cases"]
    # test_case = data["small_test_cases"]
    problem_idx = data["task_id"].split("/")[1]
    return_path,full_code = "",""
    try:
        if f"```python" in data["completion"]:
            start_idx = data["completion"].find(f"```python")
            data["completion"] = data["completion"][start_idx+len(f"```python"):]
            if "```" in data["completion"]:
                end_idx = data["completion"].find("```")
                data["completion"] = data["completion"][:end_idx]
        full_code = import_pkg+ "\n"+data["completion"] + "\n" + test_case
        with open(f"./{path}/{problem_idx}.py", "w") as f:
            f.write(full_code)
        return_path = f"./{path}/{problem_idx}.py"

    except Exception as e:
        print(e)
        pass
    # print(return_path,full_code)
    return return_path,full_code


def mbpp_add_string_to_py_file(data,evaluation_code=False, path="./tmp/"):
    if "canonical_solution" in path:
        data["completion"] = data["code"]
    if evaluation_code==False:
        test_case = data["test"]
    else: 
        test_case = "\n".join(data["test_list"])
    # test_case = data["small_test_cases"]
    problem_idx = str(data["task_id"])
    return_path,full_code = "",""
    try:
        if f"```python" in data["completion"]:
            start_idx = data["completion"].find(f"```python")
            data["completion"] = data["completion"][start_idx+len(f"```python"):]
            if "```" in data["completion"]:
                end_idx = data["completion"].find("```")
                data["completion"] = data["completion"][:end_idx]
        full_code = "\n".join(data["test_imports"])+ "\n"+data["completion"] + "\n" + test_case
        with open(f"./{path}/{problem_idx}.py", "w") as f:
            f.write(full_code)
        return_path = f"./{path}/{problem_idx}.py"

    except Exception as e:
        print(e)
    # print(return_path,full_code)
    return return_path,full_code

def add_string_to_py_file(data,evaluation_code=False, path="./tmp/"):
    if "canonical_solution" in path:
        data["completion"] = data["canonical_solution"]
    if evaluation_code==False:
        test_case = data["test_case"]
    else: 
        test_case = data["small_test_cases"]
    # test_case = data["small_test_cases"]
    problem_idx = data["problem_idx"]
    return_path,full_code = "",""
    try:
        if "class Solution" in data["completion"]:
            if "```python" in data["completion"]:
                start_idx = data["completion"].find("```python")
                data["completion"] = data["completion"][start_idx+9:]
                if "```" in data["completion"]:
                    end_idx = data["completion"].find("```")
                    data["completion"] = data["completion"][:end_idx]
            test_case = test_case.split("\n")[:100]
            test_case = "\n".join(test_case)
            # import_pkg
            full_code = import_pkg + "\n"+TreeNode_text + "\n"+ListNode_text + "\n" + data["completion"] + "\nsolution=Solution()\n" + test_case
            with open(f"./{path}/{problem_idx}.py", "w") as f:
                f.write(full_code)
            return_path = f"./{path}/{problem_idx}.py"

    except Exception as e:
        pass
    # print(return_path,full_code)
    return return_path,full_code

def calculate_code_execution_efficiency(data,evaluation_code=False,path="./tmp/",max_execution_time=10):
    entry_point = ""
    try:
        # if "task_id" in data.keys() and "HumanEval" in str(data["task_id"]):
        #     problem_idx = data["task_id"].split("/")[1]
        #     completion_file,full_code = humaneval_add_string_to_py_file(data,evaluation_code=evaluation_code, path=path)
        #     entry_point = data["entry_point"]
        # print(data.keys())
        # print(data["dataset"])
        if data["dataset"]=="mbpp":
            problem_idx = data["task_id"]
            completion_file,full_code = mbpp_add_string_to_py_file(data,evaluation_code=evaluation_code, path=path)
            code_example = data["code"]
            match = re.search(r"def\s+(\w+)\s*\(", code_example)
            if match:
                entry_point = match.group(1)
            else:
                test_example = data["test_list"][0]
                match = re.search(r"assert\s+(\w+)\s*\(", test_example)
                if match:
                    entry_point = match.group(1)
                else: completion_file== None
        else:
            problem_idx = data["problem_idx"]
            completion_file,full_code = add_string_to_py_file(data,evaluation_code=evaluation_code, path=path)
    except Exception as e:
        print(e)
    if completion_file == None:
        # print("test")
        overhead = f"""
The code execution failed.
"""
        canonical_solution_memory_usage = 0
        canonical_solution_execution_time = 0
        canonical_solution_max_memory_usage = 0
        executable = False
        return overhead, canonical_solution_memory_usage, canonical_solution_execution_time, canonical_solution_max_memory_usage, executable

    script_path = './run_code.sh'
    completion_dat_file = f'./{path}/{problem_idx}.dat'
    try:
        subprocess.run([script_path, completion_file, completion_dat_file,str(max_execution_time)], 
                            check=True, capture_output=True, text=True)

        line_profiler_results = calculate_line_efficiency(completion_file,entry_point)
        canonical_solution_memory_usage = calculate_memory_usage(completion_dat_file)
        canonical_solution_execution_time = calculate_runtime(completion_dat_file)
        canonical_solution_max_memory_usage = report_max_memory_usage(completion_dat_file)
        data["entry_point"] = entry_point
        full_code = add_profile_for_memory_profiler(full_code,data)
        completion_code = memory_profiler_pkgs + full_code + memory_profiler_prompt
        path, filename = os.path.split(completion_file)
        tmp_py_script_filename = f"{filename.split('.')[0]}_memory.py"
        tmp_py_script = os.path.join(path, tmp_py_script_filename)

        with open(tmp_py_script, "w") as f:
            f.write(completion_code)

        try:
            memory_report = subprocess.run(['timeout',"10","python", tmp_py_script], capture_output=True, text=True, check=True, timeout=10).stdout
        except:
            memory_report= "The script didn't finish within the timeout period."

        executable = True
        overhead = f"""
The total memory usage during the code execution is: {canonical_solution_memory_usage} MB*s.
The total execution time is: {canonical_solution_execution_time} s.
The maximum memory peak requirement is: {canonical_solution_max_memory_usage} MB.
The line_profiler results are: 
{line_profiler_results}
The memory profiler results are: 
{memory_report}
"""
    except Exception as e:
        # print(e)
        overhead = f"""
The code execution failed.
"""
        canonical_solution_memory_usage = 0
        canonical_solution_execution_time = 0
        canonical_solution_max_memory_usage = 0
        executable = False
    return overhead, canonical_solution_memory_usage, canonical_solution_execution_time, canonical_solution_max_memory_usage, executable
    
    
def fetch_completion(dataset,model):
    with ThreadPoolExecutor() as executor:
            future_to_entry = {executor.submit(calculate_code_execution_efficiency, copy.deepcopy(entry),False, path=model,max_execution_time=10): entry for entry in tqdm(dataset)}
            for future in tqdm(concurrent.futures.as_completed(future_to_entry)):
                entry = future_to_entry[future]
                try:
                    updated_entry = future.result()
                    idx = dataset.index(entry)
                    dataset[idx] = updated_entry
                except Exception as e:
                    pass
    return dataset

if __name__ == "__main__":
    # models = ["canonical_solution","m-a-p/OpenCodeInterpreter-DS-33B","codefuse-ai/CodeFuse-DeepSeek-33B","deepseek-ai/deepseek-coder-33b-instruct","Phind/Phind-CodeLlama-34B-v2","codellama/CodeLlama-70b-Instruct-hf","codellama/CodeLlama-34b-hf","Xwin-LM/XwinCoder-34B","deepseek-ai/deepseek-coder-6.7b-instruct","m-a-p/OpenCodeInterpreter-DS-6.7B","Artigenz/Artigenz-Coder-DS-6.7B","ise-uiuc/Magicoder-S-DS-6.7B","Nondzu/Mistral-7B-codealpaca-lora","uukuguy/speechless-starcoder2-15b","gpt-3.5-turbo-0301","gpt-3.5-turbo-0613","gpt-3.5-turbo-1106","gpt-4-turbo-preview","gpt-4","claude-3-haiku","claude-3-sonnet"]
    # "m-a-p/OpenCodeInterpreter-DS-33B","codefuse-ai/CodeFuse-DeepSeek-33B","deepseek-ai/deepseek-coder-33b-instruct","Phind/Phind-CodeLlama-34B-v2","codellama/CodeLlama-70b-Instruct-hf","codellama/CodeLlama-34b-hf",
    # "gpt-3.5-turbo-0301","gpt-3.5-turbo-0613","gpt-3.5-turbo-1106","gpt-4-turbo-preview","gpt-4", "palm-2-chat-bison",
    # models = ["gpt-3.5-turbo-0301","gpt-3.5-turbo-0613","gpt-3.5-turbo-1106","gpt-4-turbo-preview","gpt-4", "palm-2-chat-bison","gemini-pro"]
    # models = ["canonical_solution","incoder-1B","incoder-6B","starcoder","codegen-2B-mono","codegen-6B-mono","Magicoder-S-CL-7B","Magicoder-S-DS-6.7B","WizardCoder-15B-V1.0","instructcodet5p-16b","Mistral-7B-Instruct-v0.2","Mistral-7B-v0.1", "CodeLlama-7b-Python-hf", "CodeLlama-13b-Python-hf","gpt-3.5-turbo-0301","gpt-3.5-turbo-0613","gpt-3.5-turbo-1106","gpt-4-turbo-preview","gpt-4", "palm-2-chat-bison","claude-instant-1","gemini-pro"]
    # models = ["canonical_solution"]
    # models = ["codellama/CodeLlama-70b-Instruct-hf", "gpt-3.5-turbo-0301"]
    # models = ["m-a-p/OpenCodeInterpreter-DS-33B","codefuse-ai/CodeFuse-DeepSeek-33B","deepseek-ai/deepseek-coder-33b-instruct","Phind/Phind-CodeLlama-34B-v2","codellama/CodeLlama-70b-Instruct-hf","codellama/CodeLlama-34b-hf","Xwin-LM/XwinCoder-34B","deepseek-ai/deepseek-coder-6.7b-instruct","m-a-p/OpenCodeInterpreter-DS-6.7B","Artigenz/Artigenz-Coder-DS-6.7B","ise-uiuc/Magicoder-S-DS-6.7B","Nondzu/Mistral-7B-codealpaca-lora","uukuguy/speechless-starcoder2-15b"]
    models = ["m-a-p/OpenCodeInterpreter-DS-1.3B", "m-a-p/OpenCodeInterpreter-DS-6.7B", "m-a-p/OpenCodeInterpreter-DS-33B","deepseek-ai/deepseek-coder-1.3b-instruct","deepseek-ai/deepseek-coder-6.7b-instruct","deepseek-ai/deepseek-coder-33b-instruct","codellama/CodeLlama-7b-Instruct-hf","codellama/CodeLlama-13b-Instruct-hf","codellama/CodeLlama-34b-Instruct-hf","codellama/CodeLlama-70b-Instruct-hf","Xwin-LM/XwinCoder-7B","Xwin-LM/XwinCoder-13B","Xwin-LM/XwinCoder-34B","TheBloke/WizardCoder-Python-7B-V1.0-GPTQ","TheBloke/WizardCoder-Python-13B-V1.0-GPTQ","TheBloke/WizardCoder-Python-34B-V1.0-GPTQ","bigcode/starcoder2-3b","bigcode/starcoder2-7b","bigcode/starcoder2-15b"]
    
    # steps = 6
    # for step in range(1,6):
    #     for model in models:
    #         if "/" in model:
    #             model = model.split("/")[1]
    #         dat_path = f"./dat_results/leetcode_{model}_{step}"
    #         if os.path.exists(dat_path):
    #             continue
    #         try:
    #             with open(f"./results/leetcode_{model}_{step}.json", "r") as f:
    #                 dataset = json.load(f)
    #         except Exception as e:
    #             print(e)
    #             continue

    #         import os
    #         import glob
    #         import shutil

    #         if os.path.exists(dat_path):
    #             shutil.rmtree(dat_path)
    #             # print(f"目录 '{dat_path}' 已成功删除。")
                
    #         os.makedirs(dat_path)
    #         # print(f"目录 '{dat_path}' 已成功创建。")

    #         fetch_completion(dataset,dat_path)

    for model in models:
        print(model)
        if "/" in model:
            model = model.split("/")[1]
        dat_path = f"./dat_results/mbpp_{model}_timeout10_5"
        # if os.path.exists(dat_path):
        #     continue
        try:
            if model == "canonical_solution":
                with open(f"./results/mbpp_OpenCodeInterpreter-DS-1.3B.json", "r") as f:
                    dataset = json.load(f)
            else:
                with open(f"./results/mbpp_{model}_5.json", "r") as f:
                    dataset = json.load(f)
        except Exception as e:
            print(e)
            continue

        if os.path.exists(dat_path):
            shutil.rmtree(dat_path)
            
        os.makedirs(dat_path)
        for i in range(len(dataset)):
            dataset[i]["dataset"] = "mbpp"
        fetch_completion(dataset,dat_path)
