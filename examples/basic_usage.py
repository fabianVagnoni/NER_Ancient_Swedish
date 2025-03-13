"""
Basic usage example for the ner_ancient_swedish package.
"""

from ner_ancient_swedish import EUROBERT_NER
from ner_ancient_swedish.utils.ner_utils import prepare_dataset, map_labels_to_ids, tokenize_and_align_labels
from ner_ancient_swedish.utils.pytorch_utils import create_dataloaders

from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    # Load your dataset
    dataset = load_dataset("csv", data_files={"train": "../A2_train.csv", "test": "../A2_test.csv"})

    # Prepare the dataset
    processed_dataset = prepare_dataset(dataset)

    # Define your label map
    label_map = {"O": 0, "PER": 1, "LOC": 2, "ORG": 3}

    # Map labels to IDs
    processed_dataset = processed_dataset.map(lambda x: map_labels_to_ids(x, label_map))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EuroBERT/EuroBERT-2.1B")

    # Tokenize and align labels
    tokenized_dataset = processed_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True
    )

    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        tokenized_dataset["train"],
        tokenized_dataset["test"],
        batch_size=16,
        tokenizer=tokenizer
    )

    # Initialize the model
    model = EUROBERT_NER(
        model_name="eurobert-ner",
        num_labels=len(label_map)
    )

    # Train the model
    model.train(train_dataloader, val_dataloader, num_epochs=3)

    # Evaluate the model
    metrics = model.evaluate(val_dataloader)
    print(metrics)

    # Make predictions
    predictions = model.predict(val_dataloader)
    print(predictions[:5])  # Print first 5 predictions

if __name__ == "__main__":
    main() 