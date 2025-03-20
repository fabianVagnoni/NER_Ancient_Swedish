import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import json
import ast

# First, ensure entities column contains proper Python objects
def parse_entities(entities):
    if isinstance(entities, str):
        try:
            return json.loads(entities)
        except json.JSONDecodeError:
            try:
                entities = entities.replace("\n", ',')
                return ast.literal_eval(entities)
            except (ValueError, SyntaxError):
                return []
    return entities if isinstance(entities, list) else []


# Create a column with entity category distribution for stratification
def get_entity_distribution(entities_list):
    counts = Counter()
    for entity in entities_list:
        counts[entity['label']] += 1
    return counts


# Create stratification labels based on the most dominant entity type
# or a combination of entity types when multiple are present
def create_strat_label(entity_counts):
    
    # For samples with multiple entity types, use the most frequent ones
    sorted_entities = sorted(entity_counts.items(), key=lambda x: (-x[1], x[0]))
    
    # Get the dominant entity type(s)
    dominant_count = sorted_entities[0][1]
    dominant_types = [e_type for e_type, count in sorted_entities if count == dominant_count]
    
    # Create a composite label for stratification
    return "_".join(sorted(dominant_types))


# Create a combined stratification feature based on weak entities
def create_weak_entity_signature(row, weak_entities, entity_frequency):
    """
    Create a stratification signature based on the rarest entity in each text.
    
    Parameters:
    -----------
    row : pandas.Series
        Row containing binary indicators for weak entities
    weak_entities : list
        List of weak entity types
    entity_frequency : dict
        Dictionary mapping entity types to their frequencies
        
    Returns:
    --------
    str
        Stratification signature based on the rarest entity
    """
    # Check which weak entities are present in this text
    present_entities = []
    for entity in weak_entities:
        column_name = f"has_{entity.replace('-', '_')}"
        if row[column_name] == 1:
            present_entities.append(entity)
    
    if not present_entities:
        return "no_weak_entities"
    
    # Sort present entities by frequency (prioritize the rarest)
    present_entities.sort(key=lambda e: entity_frequency[e])
    
    # Return the rarest entity as the signature
    return f"has_{present_entities[0]}"


def create_stratified_train_test_split(label_list, dataset, weak_entity_threshold=300, test_size=0.2, random_state=42):
    """
    Create a stratified train-test split of the dataset based on entity categories.
    Ensures that rare/weak entities are properly distributed between train and test sets.
    
    Parameters:
    -----------
    label_list : list
        List of entity categories to stratify by
    dataset : pandas.DataFrame
        DataFrame containing the text and entities columns
    weak_entity_threshold : int, optional
        Threshold below which an entity is considered 'weak/rare'
    test_size : float, optional
        Proportion of the dataset to include in the test split
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    train_dataset : pandas.DataFrame
        Training dataset
    test_dataset : pandas.DataFrame
        Testing dataset
    """
    
    # Create a stratification column based on entity distribution
    dataset = dataset.copy()
    
    # Convert string representations to proper Python objects
    dataset['entities'] = dataset['entities'].apply(parse_entities)
    
    dataset['entity_counts'] = dataset['entities'].apply(get_entity_distribution)

    # Get overall counts of each entity type
    overall_counts = Counter()
    for entity_counts in dataset['entity_counts']:
        overall_counts.update(entity_counts)
    print("Overall counts:")
    print(overall_counts)
    
    # Identify weak entities (those with fewer occurrences than the threshold)
    # Sort them by frequency (ascending) to prioritize the rarest entities
    weak_entities = [entity for entity, count in overall_counts.items() 
                    if count < weak_entity_threshold]
    # Sort weak entities by frequency (ascending)
    weak_entities.sort(key=lambda e: overall_counts[e])
    
    print(f"\nWeak entities (count < {weak_entity_threshold}), sorted by rarity:")
    for entity in weak_entities:
        print(f"{entity}: {overall_counts[entity]}")
    
    # Create binary features for each weak entity
    for entity in weak_entities:
        column_name = f"has_{entity.replace('-', '_')}"
        dataset[column_name] = dataset['entity_counts'].apply(lambda x: 1 if entity in x else 0)
    
    print("\nTexts containing each weak entity:")
    for entity in weak_entities:
        column_name = f"has_{entity.replace('-', '_')}"
        count = dataset[column_name].sum()
        print(f"{entity}: {count} texts ({count/len(dataset)*100:.2f}%)")
    
    # Create a stratification signature based on the rarest entity in each text
    dataset['weak_entity_signature'] = dataset.apply(
        lambda row: create_weak_entity_signature(row, weak_entities, overall_counts), 
        axis=1
    )
    
    # Check the distribution of our stratification groups
    signature_counts = dataset['weak_entity_signature'].value_counts()
    print("\nStratification group distribution:")
    print(signature_counts)
    
    # Check if any group is too small for splitting
    min_group_size = signature_counts.min()
    if min_group_size < 5:  # If smallest group has fewer than 5 samples
        print(f"\nWarning: Smallest stratification group has only {min_group_size} samples.")
        print("Some groups might be too small for reliable splitting.")
        
        # For very small groups, we might need to merge them
        if min_group_size < 2:
            print("Some groups are too small for splitting. Implementing fallback strategy...")
            
            # Identify the groups that are too small
            small_groups = signature_counts[signature_counts < 2].index.tolist()
            
            # Create a merged group for these
            def merge_small_groups(signature):
                if signature in small_groups:
                    return "merged_rare_entities"
                return signature
            
            # Apply the merging
            dataset['weak_entity_signature'] = dataset['weak_entity_signature'].apply(merge_small_groups)
            
            # Print the updated distribution
            print("\nUpdated stratification group distribution after merging small groups:")
            print(dataset['weak_entity_signature'].value_counts())
    
    # Perform stratified split
    try:
        train_indices, test_indices = train_test_split(
            np.arange(len(dataset)),
            test_size=test_size,
            random_state=random_state,
            stratify=dataset['weak_entity_signature']
        )
    except ValueError as e:
        print(f"\nError in stratification: {e}")
        print("Falling back to a simpler stratification approach...")
        
        # Create a simpler stratification based on presence of any weak entity
        dataset['has_any_weak'] = (dataset['weak_entity_signature'] != "no_weak_entities").astype(int)
        
        # Also use the dominant entity type as additional stratification
        dataset['strat_label'] = dataset['entity_counts'].apply(create_strat_label)
        dataset['combined_strat'] = dataset['strat_label'] + "_" + dataset['has_any_weak'].astype(str)
        
        print("\nFallback stratification distribution:")
        print(dataset['combined_strat'].value_counts())
        
        # Try the split with the fallback approach
        train_indices, test_indices = train_test_split(
            np.arange(len(dataset)),
            test_size=test_size,
            random_state=random_state,
            stratify=dataset['combined_strat']
        )
    
    # Drop the temporary columns we created
    cols_to_drop = ['entity_counts', 'weak_entity_signature']
    cols_to_drop.extend([f"has_{entity.replace('-', '_')}" for entity in weak_entities])
    if 'strat_label' in dataset.columns:
        cols_to_drop.append('strat_label')
    if 'has_any_weak' in dataset.columns:
        cols_to_drop.append('has_any_weak')
    if 'combined_strat' in dataset.columns:
        cols_to_drop.append('combined_strat')
    
    train_dataset = dataset.iloc[train_indices].drop(cols_to_drop, axis=1)
    test_dataset = dataset.iloc[test_indices].drop(cols_to_drop, axis=1)
    
    # Print distribution statistics
    print("\nEntity distribution in original dataset:")
    original_counts = Counter()
    for entities in dataset['entities']:
        for entity in entities:
            original_counts[entity['label']] += 1
    
    print("Original:", {label: original_counts.get(label, 0) for label in label_list})
    
    train_counts = Counter()
    for entities in train_dataset['entities']:
        for entity in entities:
            train_counts[entity['label']] += 1
    
    test_counts = Counter()
    for entities in test_dataset['entities']:
        for entity in entities:
            test_counts[entity['label']] += 1
    
    print("\nTrain:", {label: train_counts.get(label, 0) for label in label_list})
    print("Test:", {label: test_counts.get(label, 0) for label in label_list})
    
    # Calculate and print the percentage of each entity in train and test
    print("\nPercentage of entities in train set:")
    for label in label_list:
        orig_count = original_counts.get(label, 0)
        if orig_count > 0:
            train_percent = (train_counts.get(label, 0) / orig_count) * 100
            test_percent = (test_counts.get(label, 0) / orig_count) * 100
            print(f"{label}: {train_percent:.1f}% in train, {test_percent:.1f}% in test")
    
    # Print statistics specifically for weak entities
    print("\nDistribution of weak entities:")
    for entity in weak_entities:
        orig_count = original_counts.get(entity, 0)
        if orig_count > 0:
            train_count = train_counts.get(entity, 0)
            test_count = test_counts.get(entity, 0)
            train_percent = (train_count / orig_count) * 100
            test_percent = (test_count / orig_count) * 100
            print(f"{entity}: {orig_count} total, {train_count} in train ({train_percent:.1f}%), {test_count} in test ({test_percent:.1f}%)")
    
    return train_dataset, test_dataset

# Example usage
def example_usage():
    a = pd.read_csv("C:/Users/fabia/OneDrive/Documentos/GitHub/NER_Ancient_Swedish/A2_train.csv")
    label_list = ['O', 'EVN', 'LOC', 'MSR-AREA', 'MSR-DIST', 'MSR-LEN', 'MSR-MON',
                'MSR-OTH', 'MSR-VOL', 'MSR-WEI', 'OCC', 'ORG-COMP', 'ORG-INST',
                'ORG-OTH', 'PER', 'SYMP', 'TME-DATE', 'TME-INTRV', 'TME-TIME', 'WRK']
    create_stratified_train_test_split(label_list, a, weak_entity_threshold=300, test_size=0.2, random_state=42)
