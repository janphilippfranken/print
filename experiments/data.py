import json
import ast
from tqdm import tqdm

from datasets import Dataset, load_dataset


def get_apps_dataset(
    name: str="codeparrot/apps", 
    split: str="test", 
    difficulty: str="introductory"
):
    """
    Loads the APPS dataset, filters based on split and difficulty, and evaluates strings. 
    """
    dataset = load_dataset(name, split=split)
    with open(f"{difficulty}.json", "r") as f:
        difficulty_json = json.load(f)
    difficulty_json = [str(int(i)) for i in difficulty_json]
    filtered_data = dataset.filter(lambda example: str(example["problem_id"]) in difficulty_json)
    print(f"Loaded {len(filtered_data)} {difficulty} examples.")
    modified_data_list = []
    for example in tqdm(filtered_data):
        modified_example = example.copy()
        for key, value in example.items():
            if isinstance(value, str):  
                try:
                    modified_example[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
        modified_data_list.append(modified_example)
    dataset = Dataset.from_list(modified_data_list)
    return dataset