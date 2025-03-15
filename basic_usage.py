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
    dataset = load_dataset("csv", data_files="../A2_train.csv")

    # Process Dataset
    processed_dataset = prepare_dataset(dataset)

    # Define your label map
    label_list = ['O', 'EVN', 'LOC', 'MSR-AREA', 'MSR-DIST', 'MSR-LEN', 'MSR-MON',
                'MSR-OTH', 'MSR-VOL', 'MSR-WEI', 'OCC', 'ORG-COMP', 'ORG-INST',
                'ORG-OTH', 'PER', 'SYMP', 'TME-DATE', 'TME-INTRV', 'TME-TIME', 'WRK']
    label_map = {label: i for i, label in enumerate(label_list)}

    # Map labels to IDs
    df = processed_dataset.map(lambda example: map_labels_to_ids(example, label_map))

    # Define Model Checkpoint 
    model_checkpoint = "EuroBERT/EuroBERT-610m"

    # Load tokenizer
    batch_size = 2
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                          add_prefix_space=True)

    # Tokenize and align the labels for the entire dataset
    max_len = 512
    tokenized_datasets = df.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, max_len),
        batched=True,
        remove_columns=df["train"].column_names
    )


    # Split the dataset into train and validation sets & create loaders
    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    train_val_split = train_dataset.train_test_split(test_size=0.1)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    train_dataloader, val_dataloader = create_dataloaders(train_dataset, val_dataset, batch_size, tokenizer)


    # Initialize the model
    model = EUROBERT_NER(
        model_name="eurobert-ner",
        num_labels=len(label_map),
        model_checkpoint=model_checkpoint
    )

    # Train the model
    model.fit(train_dataloader, 
          val_dataloader)


    # Evaluate the model
    metrics = model.evaluate(val_dataloader)

if __name__ == "__main__":
    main() 