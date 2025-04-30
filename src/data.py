from datasets import load_dataset

def load_and_preprocess_dataset(tokenizer, max_length=128):
    dataset = load_dataset("glue", "sst2")

    def tokenize_function(example):
        return tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
    tokenized_dataset.set_format(type="torch")

    return tokenized_dataset
