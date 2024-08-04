# Prompts for SOAP and baselines are mainly different in the overhead prompt, while other parts are follow the same architecture of prompt_template. 
# Unsupervised self-refine does not contain Overhead Analysis part.

prompt_template = f"""
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


SOAP_overhead_prompt_part = f"""
The total memory usage during the code execution is: {memory_usage} MB*s.
The total execution time is: {execution_time} s.
The maximum memory peak requirement is: {max_memory_usage} MB.
The line_profiler results are: 
{line_profiler_results}
The memory profiler results are: 
{memory_report}
"""

Execution_time_profiler_overhead_prompt_part = f"""
The total memory usage during the code execution is: {memory_usage} MB*s.
The total execution time is: {execution_time} s.
The maximum memory peak requirement is: {max_memory_usage} MB.
The line_profiler results are: 
{line_profiler_results}
"""

memory_profiler_overhead_prompt_part = f"""
The total memory usage during the code execution is: {memory_usage} MB*s.
The total execution time is: {execution_time} s.
The maximum memory peak requirement is: {max_memory_usage} MB.
The memory profiler results are: 
{memory_report}
"""

result_Aware_self_refine_overhead_prompt_part = f"""
The total memory usage during the code execution is: {memory_usage} MB*s.
The total execution time is: {execution_time} s.
The maximum memory peak requirement is: {max_memory_usage} MB.
"""


unsupervised_self_refine_prompt = f"""
Optimize the efficiency of the following Python code based on the task, test case, and overhead analysis provided. Ensure the optimized code can pass the given test case.

Task Description:
{task_description}

Test Case:
{test_case}

Original Code:
```python
{completion}
```

Optimization Rules:
- Encapsulate the optimized code within a Python code block (i.e., ```python\n[Your Code Here]\n```).
- Do not include the test case within the code block.
- Focus solely on code optimization; test cases are already provided.
- Ensure the provided test case passes with your optimized solution.
"""