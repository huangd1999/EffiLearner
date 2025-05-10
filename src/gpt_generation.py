import json
import openai
import json
from tqdm import tqdm
import copy
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

with open("../datasets/dataset.json", "r") as f:
    dataset = json.load(f)


# Function to fetch completion
def fetch_completion(data_entry, model):
    test_case = data_entry["small_test_cases"]
    try:
        completions = openai.ChatCompletion.create(
            model=model,
            stream=False,
            messages=[
                {"role": "user", "content": f"Please complete Python code based on the task description and test cases. # Task description:\n{data_entry['markdown_description']}\n{test_case}\n#Solution:\n"},
            ],
            request_timeout=100,
        )
        data_entry["completion"] = completions.choices[0]["message"]["content"]
    except Exception as e:
        print(repr(e))
        data_entry["completion"] = "API Error"
    return data_entry

model_list = ["gpt-4"]
for model in model_list:
    with ThreadPoolExecutor() as executor:
        future_to_entry = {executor.submit(fetch_completion, copy.deepcopy(entry), model): entry for entry in tqdm(dataset)}
        for future in tqdm(concurrent.futures.as_completed(future_to_entry)):
            entry = future_to_entry[future]
            try:
                updated_entry = future.result()
                idx = dataset.index(entry)
                dataset[idx] = updated_entry
            except Exception as e:
                print(repr(e))


    with open(f"./EffiBench_{model}.json", "w") as f:
        json.dump(dataset, f, indent=4)