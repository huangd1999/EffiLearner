# SOAP: Self-Optimization Improves the Efficiency of Code Generation

## Reproduce paper results

We first generate initial code based on the task description:

```
# mbpp
python mbpp_generation.py

# humaneval
python humaneval_generation.py

# EffiBench
python effibench_generation.py
```

Then, SOAP will optimize the efficiency of LLM generated code:

```
# mbpp
python mbpp_selfoptimizer.py

# humaneval
python humaneval_selfoptimizer.py

# EffiBench
python effibench_self_optimizer.py
```

## Report Efficiency results

To report efficiency metrics of each dataset, we will first run:

```
python code_efficiency_calculator.py
```

to calculate the efficiency metric for each sample.

Then, run:

```
python calculator_memory_usage.py
```

to report the efficiency results.