import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
import torch
from torch.utils.data import TensorDataset, DataLoader

def stratified_ner_split(texts, labels, val_size=0.2, random_state=42):
    """
    Create a stratified split for NER data by ensuring proper entity distribution.
    
    Args:
        texts: List of tokenized texts or input_ids
        labels: List of token labels corresponding to texts (can be strings or numeric indices)
        val_size: Proportion of validation set size
        random_state: Random seed for reproducibility
        
    Returns:
        train_texts, val_texts, train_labels, val_labels
    """
    # Handle empty dataset or single example
    if len(texts) <= 1:
        raise ValueError("Dataset must contain at least 2 examples for splitting")
    
    # Calculate entity distributions per document
    doc_entity_counts = []
    entity_types = set()
    
    for doc_labels in labels:
        # Skip empty documents
        if len(doc_labels) == 0:
            doc_entity_counts.append(Counter())
            continue
            
        # Handle both string labels and numeric indices
        if isinstance(doc_labels[0], (int, np.integer)) or (hasattr(doc_labels[0], 'item') and isinstance(doc_labels[0].item(), (int, np.integer))):
            # For numeric labels, we can't directly identify 'O' (background), 
            # so we'll assume label 0 is the background class (common convention)
            entity_counter = Counter([label for label in doc_labels if label != 0 and label != -100])
        else:
            # For string labels, exclude 'O' (outside/background) labels
            entity_counter = Counter([label for label in doc_labels if label != 'O' and label != -100])
        
        doc_entity_counts.append(entity_counter)
        entity_types.update(entity_counter.keys())
    
    # If no entity types found, fall back to random split
    if not entity_types:
        print("Warning: No entity types found for stratification. Falling back to random split.")
        indices = np.arange(len(texts))
        np.random.seed(random_state)
        np.random.shuffle(indices)
        split_idx = int(len(indices) * (1 - val_size))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
    else:
        # Create document features based on entity distribution
        entity_types = sorted(list(entity_types))
        doc_features = np.zeros((len(texts), len(entity_types)))
        
        for i, counter in enumerate(doc_entity_counts):
            for j, entity_type in enumerate(entity_types):
                # Get count of this entity type and normalize by document length
                doc_features[i, j] = counter.get(entity_type, 0) / max(1, len([l for l in labels[i] if l != -100]))
        
        # Compute document weights to prioritize rare entities
        entity_type_counts = Counter()
        for counter in doc_entity_counts:
            entity_type_counts.update(counter)
        
        # Calculate inverse frequency weights for entity types
        total_entities = sum(entity_type_counts.values())
        entity_weights = {entity: total_entities / max(1, count) for entity, count in entity_type_counts.items()}
        
        # Calculate weighted importance score for each document
        doc_importance = np.zeros(len(texts))
        for i, counter in enumerate(doc_entity_counts):
            score = sum(counter.get(entity, 0) * entity_weights.get(entity, 1) for entity in entity_types)
            doc_importance[i] = score
        
        # Use importance scores to influence the stratification
        # Create balanced groups for stratification
        n_groups = min(5, len(texts))  # Number of strata, but not more than dataset size
        doc_groups = np.zeros(len(texts), dtype=int)
        
        # Sort documents by importance and assign to groups in round-robin fashion
        sorted_indices = np.argsort(doc_importance)
        for i, idx in enumerate(sorted_indices):
            doc_groups[idx] = i % n_groups
        
        # Perform stratified split based on the assigned groups
        train_indices, val_indices = train_test_split(
            np.arange(len(texts)),
            test_size=val_size,
            random_state=random_state,
            stratify=doc_groups
        )
    
    # Create the split datasets
    train_texts = [texts[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    
    # Verify entity distribution in train and validation sets
    train_entity_counts = sum_entity_counts([doc_entity_counts[i] for i in train_indices])
    val_entity_counts = sum_entity_counts([doc_entity_counts[i] for i in val_indices])
    
    print("Entity distribution in training set:")
    for entity, count in train_entity_counts.items():
        print(f"  {entity}: {count}")
    
    print("\nEntity distribution in validation set:")
    for entity, count in val_entity_counts.items():
        print(f"  {entity}: {count}")
    
    return train_texts, val_texts, train_labels, val_labels

def sum_entity_counts(doc_entity_counts):
    """Sum entity counts across multiple documents."""
    total_counts = Counter()
    for counter in doc_entity_counts:
        total_counts.update(counter)
    return total_counts

def compute_class_weights(labels, num_classes=None, label_map=None):
    """
    Compute class weights for imbalanced NER data.
    
    Args:
        labels: List of label sequences (can be strings or numeric indices)
        num_classes: Total number of classes (if not provided, will be inferred)
        label_map: Optional mapping from label names to indices
        
    Returns:
        torch.Tensor: Weights for each class, can be used in loss functions
    """
    import torch
    from collections import Counter
    
    # Flatten all labels and count occurrences
    all_labels = []
    for doc_labels in labels:
        if len(doc_labels) == 0:
            continue
            
        # Handle both string and numeric labels
        if isinstance(doc_labels[0], str):
            # For string labels, convert to indices if label_map is provided
            if label_map:
                doc_labels = [label_map.get(l, -100) for l in doc_labels]
        
        all_labels.extend([l for l in doc_labels if l != -100])  # Exclude padding
    
    # Count label occurrences
    label_counts = Counter(all_labels)
    
    # Infer number of classes if not provided
    if num_classes is None:
        if label_map:
            num_classes = len(label_map)
        else:
            num_classes = max(label_counts.keys()) + 1 if label_counts else 1
    
    # Compute inverse frequency weights
    total_samples = sum(label_counts.values())
    class_weights = torch.ones(num_classes)
    
    for label, count in label_counts.items():
        if isinstance(label, str) and label_map:
            # Convert string label to index using label_map
            label_idx = label_map[label]
        else:
            label_idx = label
        
        # Inverse frequency weighting (higher weight for less frequent classes)
        class_weights[label_idx] = total_samples / (count * num_classes)
    
    # Normalize weights
    if class_weights.sum() > 0:
        class_weights = class_weights / class_weights.sum() * num_classes
    
    return class_weights

def generate_stratified_data_splits(texts, labels, val_size=0.2, test_size=0.1, random_state=42):
    """
    Generate train, validation, and test splits with stratification for NER data.
    
    Args:
        texts: List of tokenized texts or input_ids
        labels: List of token labels corresponding to texts
        val_size: Proportion of validation set size relative to the whole dataset
        test_size: Proportion of test set size relative to the whole dataset
        random_state: Random seed for reproducibility
        
    Returns:
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels
    """
    # First split off the test set
    train_val_texts, test_texts, train_val_labels, test_labels = stratified_ner_split(
        texts, labels, val_size=test_size, random_state=random_state
    )
    
    # Then split the remaining data into train and validation sets
    # Adjust val_size to account for already removed test set
    adjusted_val_size = val_size / (1 - test_size)
    train_texts, val_texts, train_labels, val_labels = stratified_ner_split(
        train_val_texts, train_val_labels, val_size=adjusted_val_size, random_state=random_state
    )
    
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

def prepare_stratified_data_for_model(texts, labels, tokenizer, max_len, label_map=None):
    """
    Prepare stratified data split for the model using the existing ner_utils functions.
    Compatible with the current implementation of ner_utils.py.
    
    Args:
        texts: List of tokenized texts
        labels: List of labels
        tokenizer: Tokenizer from transformers
        max_len: Maximum sequence length
        label_map: Optional mapping from label names to indices
        
    Returns:
        Dictionary with tokenized inputs and labels
    """
    from ner_ancient_swedish.utils.ner_utils import tokenize_and_align_labels
    
    # Create examples in the format expected by ner_utils.py
    examples = {
        "text": texts,
        "labels": labels
    }
    
    # Convert string labels to IDs if label_map is provided
    if label_map is not None and isinstance(labels[0][0], str):
        from ner_ancient_swedish.utils.ner_utils import map_labels_to_ids
        examples = map_labels_to_ids(examples, label_map)
    
    # Tokenize and align labels using the existing ner_utils function
    tokenized_inputs = tokenize_and_align_labels(examples, tokenizer, max_len)
    
    return tokenized_inputs

def create_data_loaders_from_stratified_split(texts, labels, tokenizer, max_len, batch_size, 
                                           val_size=0.2, test_size=0.1, label_map=None, 
                                           random_state=42, return_class_weights=True):
    """
    Create data loaders for training from a stratified split of the data.
    Compatible with the current implementation of all files.
    
    Args:
        texts: List of tokenized texts
        labels: List of labels
        tokenizer: Tokenizer from transformers
        max_len: Maximum sequence length
        batch_size: Batch size for training
        val_size: Proportion of validation set size
        test_size: Proportion of test set size
        label_map: Optional mapping from label names to indices
        random_state: Random seed for reproducibility
        return_class_weights: Whether to return class weights
        
    Returns:
        train_loader, val_loader, test_loader, (class_weights if return_class_weights=True)
    """
    # Perform stratified split
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = generate_stratified_data_splits(
        texts, labels, val_size=val_size, test_size=test_size, random_state=random_state
    )
    
    # Prepare data for model
    train_inputs = prepare_stratified_data_for_model(train_texts, train_labels, tokenizer, max_len, label_map)
    val_inputs = prepare_stratified_data_for_model(val_texts, val_labels, tokenizer, max_len, label_map)
    test_inputs = prepare_stratified_data_for_model(test_texts, test_labels, tokenizer, max_len, label_map)
    
    # Convert to tensors and create datasets
    # The model expects input_ids, attention_mask, and labels as tensors
    train_dataset = create_tensor_dataset(train_inputs)
    val_dataset = create_tensor_dataset(val_inputs)
    test_dataset = create_tensor_dataset(test_inputs)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Compute class weights if requested
    if return_class_weights:
        num_classes = len(label_map) if label_map else max(max([max(l) for l in train_labels if len(l) > 0], default=0), 0) + 1
        class_weights = compute_class_weights(train_labels, num_classes=num_classes, label_map=label_map)
        return train_loader, val_loader, test_loader, class_weights
    else:
        return train_loader, val_loader, test_loader

def create_tensor_dataset(tokenized_inputs):
    """
    Convert tokenized inputs to a TensorDataset.
    Compatible with the current implementation of eurobert_ner.py.
    
    Args:
        tokenized_inputs: Dictionary with tokenized inputs and labels
        
    Returns:
        TensorDataset compatible with the model
    """
    # Convert list of lists to padded tensors if needed
    input_ids = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]
    labels = tokenized_inputs["labels"]
    
    # Create a compatible dataset format for eurobert_ner.py
    # The model expects DataLoader yielding batches with these keys
    class NERDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, attention_mask, labels):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.labels = labels
        
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'labels': self.labels[idx]
            }
    
    return NERDataset(input_ids, attention_mask, labels)

def process_dataset_for_ner(dataset, tokenizer, max_len, label_map, 
                           val_size=0.2, test_size=0.1, batch_size=16, 
                           use_stratification=True, random_state=42):
    """
    Process a dataset for NER training using the existing ner_utils.py functions.
    
    Args:
        dataset: Dataset with 'text' and 'entities' fields
        tokenizer: Tokenizer for tokenizing text
        max_len: Maximum sequence length
        label_map: Mapping from label strings to indices
        val_size: Proportion of validation set
        test_size: Proportion of test set
        batch_size: Batch size for training
        use_stratification: Whether to use stratified split
        random_state: Random seed
        
    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    from ner_ancient_swedish.utils.ner_utils import prepare_dataset
    
    # Process raw dataset to extract token-level labels
    processed_dataset = prepare_dataset(dataset)
    
    if use_stratification:
        # Extract texts and labels for stratification
        texts = [example['text'] for example in processed_dataset]
        labels = [example['labels'] for example in processed_dataset]
        
        # Create data loaders with stratified split
        return create_data_loaders_from_stratified_split(
            texts, labels, tokenizer, max_len, batch_size, 
            val_size, test_size, label_map, random_state
        )
    else:
        # Use random split
        from torch.utils.data import random_split
        
        # Map labels to ids
        from ner_ancient_swedish.utils.ner_utils import map_labels_to_ids
        
        dataset_size = len(processed_dataset)
        train_size = int(dataset_size * (1 - val_size - test_size))
        val_size_abs = int(dataset_size * val_size)
        test_size_abs = dataset_size - train_size - val_size_abs
        
        # Create random split
        train_dataset, val_dataset, test_dataset = random_split(
            processed_dataset, 
            [train_size, val_size_abs, test_size_abs],
            generator=torch.Generator().manual_seed(random_state)
        )
        
        # Process each split
        train_examples = []
        for example in train_dataset:
            example = map_labels_to_ids(example, label_map)
            train_examples.append(example)
        
        val_examples = []
        for example in val_dataset:
            example = map_labels_to_ids(example, label_map)
            val_examples.append(example)
        
        test_examples = []
        for example in test_dataset:
            example = map_labels_to_ids(example, label_map)
            test_examples.append(example)
        
        # Tokenize and prepare for model
        from ner_ancient_swedish.utils.ner_utils import tokenize_and_align_labels
        
        train_features = []
        for example in train_examples:
            train_features.append(tokenize_and_align_labels({
                "text": example['text'],
                "labels": example['labels']
            }, tokenizer, max_len))
        
        val_features = []
        for example in val_examples:
            val_features.append(tokenize_and_align_labels({
                "text": example['text'],
                "labels": example['labels']
            }, tokenizer, max_len))
        
        test_features = []
        for example in test_examples:
            test_features.append(tokenize_and_align_labels({
                "text": example['text'],
                "labels": example['labels']
            }, tokenizer, max_len))
        
        # Create tensor datasets
        train_dataset = create_combined_dataset(train_features)
        val_dataset = create_combined_dataset(val_features)
        test_dataset = create_combined_dataset(test_features)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Compute class weights
        train_labels = [example['labels'] for example in train_examples]
        class_weights = compute_class_weights(train_labels, num_classes=len(label_map))
        
        return train_loader, val_loader, test_loader, class_weights

def create_combined_dataset(features):
    """
    Combine multiple tokenized features into a single dataset.
    
    Args:
        features: List of tokenized features
        
    Returns:
        Dataset compatible with the model
    """
    # Create a compatible dataset format
    class NERDataset(torch.utils.data.Dataset):
        def __init__(self, features):
            self.features = features
        
        def __len__(self):
            return len(self.features)
        
        def __getitem__(self, idx):
            feature = self.features[idx]
            return {
                'input_ids': feature['input_ids'],
                'attention_mask': feature['attention_mask'],
                'labels': feature['labels']
            }
    
    return NERDataset(features) 