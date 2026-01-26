from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/syntax/train/train_qwen.json",
        "validation": "data/syntax/val/val_qwen.json",
    }
)

print(dataset)
print("Sample:")
print(dataset["train"][0])
