from datasets import load_dataset
dataset = load_dataset("Helsinki-NLP/mu-shroom", "all")

# Access splits
train = dataset["train_unlabeled"]
val = dataset["validation"]
test = dataset["test"]
def load_mu_shroom(dataset_name="Helsinki-NLP/mu-shroom", lang="all"):
    """
    Load the Mu-Shroom dataset from Hugging Face.

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset to load. Options are "train_unlabeled", "validation", and "test".

    Returns:
        Dataset: The loaded dataset split.
    """
    dataset = load_dataset(dataset_name, split=split)
    return dataset
def add_id(example, idx):
    example["id"] = idx
    return example