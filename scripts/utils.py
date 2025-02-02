import json
from datasets import Dataset, load_dataset
import random
import os


def save_as_jsonl(data, filename):
    """
    Saves a list of dictionaries to a JSONL file.

    Args:
    data (list of dict): Data to save, where each dictionary in the list represents a separate JSON object.
    filename (str): Name of the file to save the data to.

    Returns:
    None
    """
    # Extract the directory from the full file path
    directory = os.path.dirname(filename)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, 'w', encoding='utf-8') as file:
        for entry in data:
            json_string = json.dumps(entry)
            file.write(json_string + '\n')


from datasets import Dataset, load_dataset
import random
import os

def save_as_hf_dataset(data, directory):
    """
    Splits data into a training and test dataset, then saves both using the Hugging Face datasets library.

    Parameters:
        data (list): A list of dictionaries representing the data.
        directory (str): Base directory path where the datasets will be saved.
    """
    # Transform list of dictionaries to a dictionary of lists
    transformed_data = {key: [dic[key] for dic in data if key in dic] for key in set().union(*data)}

    # Convert dictionary of lists into a Hugging Face Dataset
    dataset = Dataset.from_dict(transformed_data)
    # Shuffle the dataset
    dataset = dataset.shuffle(seed=random.randint(0, 10000))
    
    # Split the dataset into training and test datasets
    test_size = 1000  # fixed size for the test set
    if len(dataset) > test_size:
        train_size = len(dataset) - test_size
    else:
        raise ValueError("Data size must be greater than 1000 for splitting")
    
    # Using Dataset.train_test_split to handle the split
    split_datasets = dataset.train_test_split(test_size=test_size, train_size=train_size)
    
    # Prepare directories for saving datasets
    train_dir = os.path.join(directory, "train")
    test_dir = os.path.join(directory, "test")
    
    # Saving the datasets to disk
    split_datasets['train'].save_to_disk(train_dir)
    split_datasets['test'].save_to_disk(test_dir)
    
    print(f"Datasets saved: Train dataset at {train_dir}, Test dataset at {test_dir}")
