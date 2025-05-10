# EffiLearner: Enhancing Efficiency of Generated Code via Self-Optimization

## Installation

```
git clone https://github.com/huangd1999/EffiLearner
cd EffiLearner
pip install -r requirements.txt

# For GPT models
export OPENAI_API_KEY=[your api key]
export OPENAI_API_BASE=[your custom api base url]
```

## Usage

We first generate initial inefficient code by runing the following commands:
```
cd src
# for open source LLMs
python open_llm_generation.py --checkpoint [huggingface model name] --dataset EffiBench

# For GPT models
python gpt_generation.py --checkpoint gpt-4 --dataset EffiBench
```

Then, we can use EffiLearner to optimize the efficiency of the initial code with the following commands:

```
# for open source LLMs
python EffiLearner.py --checkpoint [huggingface model name] --dataset EffiBench

# For GPT models
python gpt_EffiLearner.py --checkpoint gpt-4 --dataset EffiBench
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
