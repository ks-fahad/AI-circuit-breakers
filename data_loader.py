import json
from datasets import Dataset
from sklearn.model_selection import train_test_split

def load_json_dataset(train_file_path, val_file_path):
    with open(train_file_path, 'r') as f:
        train_data = json.load(f)
    
    with open(val_file_path, 'r') as f:
        val_data = json.load(f)
    
    train_prompts = [item['prompt'] for item in train_data]
    train_outputs = [item['output'] for item in train_data]
    
    val_prompts = [item['prompt'] for item in val_data]
    val_outputs = [item['output'] for item in val_data]
    
    train_dataset = Dataset.from_dict({'prompt': train_prompts, 'output': train_outputs})
    val_dataset = Dataset.from_dict({'prompt': val_prompts, 'output': val_outputs})
    
    return train_dataset, val_dataset
