from nervaluate import Evaluator
import numpy as np

def compute_metrics(p, label_list1):
    predictions, labels = p
    
    # Check if predictions is a list of arrays (variable length)
    if isinstance(predictions, list):
        # Handle variable length sequences
        predictions_argmax = [np.argmax(pred, axis=1) if pred.ndim > 1 else pred for pred in predictions]
    else:
        # Original case - predictions is a single array
        predictions_argmax = np.argmax(predictions, axis=2)
        # Convert to list format for consistency
        predictions_argmax = [pred for pred in predictions_argmax]
        labels = [label for label in labels]

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list1[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions_argmax, labels)
    ]
    true_labels = [
        [label_list1[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions_argmax, labels)
    ]

    # Convert to nervaluate format (list of entities with start/end positions)
    true_entities = []
    pred_entities = []

    for doc_true_labels, doc_true_preds in zip(true_labels, true_predictions):
        # Process ground truth entities
        doc_entities = []
        current_entity = None

        for i, label in enumerate(doc_true_labels):
            if label == 'O' and current_entity:
                doc_entities.append(current_entity)
                current_entity = None
            elif label.startswith('B-') or (label != 'O' and (current_entity is None or current_entity['label'] != label)):
                if current_entity:
                    doc_entities.append(current_entity)
                current_entity = {'label': label, 'start': i, 'end': i}
            elif label != 'O' and current_entity and current_entity['label'] == label:
                current_entity['end'] = i

        if current_entity:
            doc_entities.append(current_entity)

        true_entities.append(doc_entities)

        # Process predicted entities
        doc_entities = []
        current_entity = None

        for i, label in enumerate(doc_true_preds):
            if label == 'O' and current_entity:
                doc_entities.append(current_entity)
                current_entity = None
            elif label.startswith('B-') or (label != 'O' and (current_entity is None or current_entity['label'] != label)):
                if current_entity:
                    doc_entities.append(current_entity)
                current_entity = {'label': label, 'start': i, 'end': i}
            elif label != 'O' and current_entity and current_entity['label'] == label:
                current_entity['end'] = i

        if current_entity:
            doc_entities.append(current_entity)

        pred_entities.append(doc_entities)

    # Create evaluator and compute metrics
    evaluator = Evaluator(true_entities, pred_entities, tags=list(set(label_list1) - {'O'}))
    eval_results = evaluator.evaluate()
    results = eval_results[0]

    # Extract overall metrics and account for possible division by zero
    prec = results['strict']['precision']
    reca = results['strict']['recall']
    denominator = prec + reca
    f1 = 2 * (prec * reca / denominator) if denominator != 0 else 0

    return {
        "precision": prec,
        "recall": reca,
        "f1": f1,
        "accuracy": calculate_token_accuracy(true_labels, true_predictions),
    }


def calculate_token_accuracy(true_labels, true_predictions):
    """Calculate token-level accuracy manually since nervaluate doesn't provide it"""
    correct = 0
    total = 0

    for doc_labels, doc_preds in zip(true_labels, true_predictions):
        for label, pred in zip(doc_labels, doc_preds):
            if label == pred:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0

