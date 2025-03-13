import ast


def prepare_dataset(dataset):
    def process_example(example):
        # Parse the 'entities' string into a list of dictionaries
        entities_str = example['entities']
        cleaned_entities_str = "[" + entities_str.replace("\n", ",").strip() + "]"
        entities_list = ast.literal_eval(cleaned_entities_str)

        # Split 'text' into words and compute their character positions
        text = example['text']
        words = text.split()
        start_positions = []
        current_pos = 0
        for word in words:
            start = text.find(word, current_pos)
            if start == -1:
                raise ValueError(f"Word '{word}' not found in text starting from position {current_pos}")
            start_positions.append(start)
            current_pos = start + len(word)

        # Assign labels to each word based on entity spans
        labels = []
        for i, word in enumerate(words):
            word_start = start_positions[i]
            word_end = word_start + len(word)
            label = "O"  # Default label for non-entities
            for entity in entities_list[0]:
                if entity['start'] <= word_start and word_end <= entity['end']:
                    label = entity['label']
                    break
            labels.append(label)

        # Return the updated example
        return {
            'text': words,
            'labels': labels,
            'id': example['id'],
        }

    # Apply the processing function to the dataset
    processed_dataset = dataset.map(process_example)
    return processed_dataset


def map_labels_to_ids(example, label_map):
    example['labels'] = [label_map[label] for label in example['labels']]
    return example


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs