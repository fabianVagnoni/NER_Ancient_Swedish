import pandas as pd
import numpy as np
import json
from transformers import pipeline
import torch
from ner_ancient_swedish.models.eurobert_ner import EUROBERT_NER
from transformers import AutoTokenizer

def prepare_test_dataset(test_dataset):
    """Prepare the test dataset by splitting text into words with their positions."""
    processed_examples = []

    for example in test_dataset["train"]:
        text = example["text"]
        words = text.split()

        # Track character positions for each word
        start_positions = []
        current_pos = 0
        for word in words:
            start = text.find(word, current_pos)
            if start == -1:
                raise ValueError(f"Word '{word}' not found in text starting from position {current_pos}")
            start_positions.append(start)
            current_pos = start + len(word)

        end_positions = [start + len(word) for start, word in zip(start_positions, words)]

        processed_examples.append({
            "id": example["id"],
            "text": text,
            "words": words,
            "start_positions": start_positions,
            "end_positions": end_positions
        })

    return processed_examples


def create_ner_predictions_csv(trainer, tokenizer, test_dataset, id2label, output_file="predictions.csv"):
    """
    Create a CSV file with NER predictions in the required format.

    Args:
        trainer: The trained Hugging Face Trainer object
        tokenizer: The tokenizer used during training
        test_dataset: The test dataset
        id2label: Dictionary mapping from label IDs to label names
        output_file: Path to save the output CSV file
    """
    # Prepare test dataset
    processed_examples = prepare_test_dataset(test_dataset)

    # Create NER pipeline using the trained model
    ner_pipeline = pipeline(
        "token-classification",
        model=trainer.model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  # Group tokens into named entities
    )

    # List to store results
    results = []

    # Process each example
    for example in processed_examples:
        text = example["text"]
        words = example["words"]
        start_positions = example["start_positions"]
        end_positions = example["end_positions"]
        example_id = example["id"]

        # Get predictions
        entities = ner_pipeline(text)

        # Filter out 'O' predictions and format the entities
        formatted_entities = []
        for entity in entities:
            if entity["entity_group"] != "O":
                formatted_entities.append({
                    "label": entity["entity_group"],
                    "start": entity["start"],
                    "end": entity["end"]
                })

        # Add to results
        results.append({
            "id": example_id,
            "entities": json.dumps([formatted_entities])
        })

    # Create and save the DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    return results_df


def create_torch_ner_predictions_csv(model, tokenizer, test_dataset, label_list, max_length=512, output_file="predictions.csv", device="cuda"):
    """
    Create a CSV file with NER predictions from a PyTorch model in the required format.

    Args:
        model: The trained PyTorch model (e.g., EUROBERT_NER instance)
        tokenizer: The tokenizer used during training
        test_dataset: The test dataset
        label_list: List of label names corresponding to label indices
        output_file: Path to save the output CSV file
        device: Device to run predictions on ('cuda' or 'cpu')
    """
    # Prepare test dataset
    processed_examples = prepare_test_dataset(test_dataset)
    
    # List to store results
    results = []
    
    # Process each example
    for example in processed_examples:
        text = example["text"]
        words = example["words"]
        start_positions = example["start_positions"]
        end_positions = example["end_positions"]
        example_id = example["id"]
        
        # Tokenize the input text
        tokens = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True
        )
        
        # Move tensors to device
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        
        # Get word-to-token alignment
        offset_mapping = tokens.offset_mapping[0].numpy()
        
        # Get predictions from the model
        model.eval()
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
        
        # Map tokens to words and identify entities
        current_entity = None
        formatted_entities = []
        
        for token_idx, (pred_idx, offset) in enumerate(zip(predictions, offset_mapping)):
            # Skip special tokens (CLS, SEP, PAD, etc.)
            if offset[0] == 0 and offset[1] == 0:
                continue
                
            pred_label = label_list[pred_idx]
            
            # Skip 'O' (non-entity) predictions
            if pred_label == "O":
                if current_entity:
                    # End of an entity
                    formatted_entities.append(current_entity)
                    current_entity = None
                continue
            
            # Convert token offsets to character positions
            token_start, token_end = offset
            
            if current_entity is None:
                # Start of a new entity
                current_entity = {
                    "label": pred_label,
                    "start": int(token_start),
                    "end": int(token_end)
                }
            elif current_entity["label"] == pred_label:
                # Continuation of an entity - extend the end position
                current_entity["end"] = int(token_end)
            else:
                # End of previous entity, start of a new one
                formatted_entities.append(current_entity)
                current_entity = {
                    "label": pred_label,
                    "start": int(token_start),
                    "end": int(token_end)
                }
        
        # Add the last entity if exists
        if current_entity:
            formatted_entities.append(current_entity)
        
        # Add to results
        results.append({
            "id": example_id,
            "entities": json.dumps([formatted_entities])
        })
    
    # Create and save the DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return results_df
