from datasets import load_dataset

def load_data():
    dataset = load_dataset("wmt/wmt14", "hi-en")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    return train_dataset, test_dataset
