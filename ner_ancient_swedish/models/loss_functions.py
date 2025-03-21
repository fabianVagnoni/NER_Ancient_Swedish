import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import Counter

class FocalLoss(nn.Module):
    """
    Focal Loss for dealing with class imbalance in NER tagging.
    
    Reference:
    Lin et al., "Focal Loss for Dense Object Detection", 
    https://arxiv.org/abs/1708.02002
    
    Args:
        num_classes: Number of classes in the dataset
        alpha: Optional weighting factor for class balance (tensor or float)
        gamma: Focusing parameter to reduce the relative loss for well-classified examples
        ignore_index: Index to ignore (e.g., padding tokens)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, num_classes, alpha=None, gamma=2.0, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # Handle class weights
        if alpha is None:
            self.alpha = None
        else:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.FloatTensor(alpha)
            else:
                self.alpha = torch.tensor([alpha] * num_classes)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, seq_len, num_classes] or [batch_size * seq_len, num_classes]
            targets: [batch_size, seq_len] or [batch_size * seq_len]
            
        Returns:
            loss: Scalar loss value
        """
        
        # Reshape if needed for sequence data
        if len(logits.shape) == 3:
            # [batch_size, seq_len, num_classes] -> [batch_size * seq_len, num_classes]
            logits = logits.reshape(-1, self.num_classes)
            # [batch_size, seq_len] -> [batch_size * seq_len]
            targets = targets.reshape(-1)
        
        # Create mask for valid positions (not ignore_index)
        valid_mask = (targets != self.ignore_index)
        valid_targets = targets[valid_mask]
        valid_logits = logits[valid_mask]
        
        # Compute probabilities via softmax
        probs = F.softmax(valid_logits, dim=1)
        
        # Get probability for the target classes
        target_probs = probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weights
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != valid_targets.device:
                self.alpha = self.alpha.to(valid_targets.device)
            alpha_weight = self.alpha.gather(0, valid_targets)
            focal_weight = focal_weight * alpha_weight
        
        # Compute cross entropy loss with focal weighting
        ce_loss = -torch.log(target_probs)
        loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy with label smoothing for NER.
    
    Label smoothing helps prevent the model from becoming overconfident and
    improves generalization.
    
    Args:
        num_classes: Number of classes in the dataset
        epsilon: Smoothing parameter (0 means no smoothing)
        ignore_index: Index to ignore (e.g., padding tokens)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, num_classes, epsilon=0.1, ignore_index=-100, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, seq_len, num_classes] or [batch_size * seq_len, num_classes]
            targets: [batch_size, seq_len] or [batch_size * seq_len]
            
        Returns:
            loss: Scalar loss value
        """
        
        # Reshape if needed for sequence data
        if len(logits.shape) == 3:
            # [batch_size, seq_len, num_classes] -> [batch_size * seq_len, num_classes]
            logits = logits.reshape(-1, self.num_classes)
            # [batch_size, seq_len] -> [batch_size * seq_len]
            targets = targets.reshape(-1)
        
        # Create mask for valid positions (not ignore_index)
        valid_mask = (targets != self.ignore_index)
        valid_targets = targets[valid_mask]
        valid_logits = logits[valid_mask]
        
        if valid_targets.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Create smoothed one-hot vectors
        with torch.no_grad():
            label_smoothed = torch.zeros_like(valid_logits)
            label_smoothed.scatter_(1, valid_targets.unsqueeze(1), 1.0)
            # Smooth the labels
            label_smoothed = label_smoothed * (1.0 - self.epsilon) + self.epsilon / self.num_classes
        
        # Compute smoothed cross entropy loss
        log_probs = F.log_softmax(valid_logits, dim=1)
        loss = -torch.sum(log_probs * label_smoothed, dim=1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for handling imbalanced NER classes.
    
    The Dice coefficient measures the overlap between predictions and ground truth
    and is less affected by class imbalance.
    
    Args:
        num_classes: Number of classes in the dataset
        ignore_index: Index to ignore (e.g., padding tokens)
        smooth: Smoothing parameter to avoid division by zero
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, num_classes, ignore_index=-100, smooth=1.0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, seq_len, num_classes] or [batch_size * seq_len, num_classes]
            targets: [batch_size, seq_len] or [batch_size * seq_len]
            
        Returns:
            loss: Scalar loss value
        """
        
        # Reshape if needed for sequence data
        if len(logits.shape) == 3:
            # [batch_size, seq_len, num_classes] -> [batch_size * seq_len, num_classes]
            logits = logits.reshape(-1, self.num_classes)
            # [batch_size, seq_len] -> [batch_size * seq_len]
            targets = targets.reshape(-1)
        
        # Create mask for valid positions (not ignore_index)
        valid_mask = (targets != self.ignore_index)
        valid_targets = targets[valid_mask]
        valid_logits = logits[valid_mask]
        
        if valid_targets.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Get probabilities
        probs = F.softmax(valid_logits, dim=1)
        
        # Create one-hot targets
        targets_one_hot = F.one_hot(valid_targets, self.num_classes).float()
        
        # Calculate Dice loss for each class
        # Compute intersections and unions
        intersect = torch.sum(probs * targets_one_hot, dim=0)
        cardinality = torch.sum(probs + targets_one_hot, dim=0)
        
        # Calculate Dice scores and loss
        dice_scores = (2. * intersect + self.smooth) / (cardinality + self.smooth)
        
        # Convert to loss (1 - dice_score)
        loss = 1 - dice_scores
        
        # Exclude background class (index 0) if it exists
        if self.num_classes > 1:
            loss = loss[1:]  # Exclude background class
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Parameter calculation functions

def calculate_focal_loss_params(train_labels, label_map=None, num_classes=None):
    """
    Calculate optimal parameters for Focal Loss based on label distribution.
    
    Args:
        train_labels: List of label sequences from training data
        label_map: Optional mapping from label names to indices
        num_classes: Number of classes (if None, will be inferred)
        
    Returns:
        dict: Dictionary with optimal alpha and gamma values
    """
    from ner_ancient_swedish.utils.data_utils import compute_class_weights
    
    # Compute class weights (alpha)
    alpha = compute_class_weights(train_labels, num_classes=num_classes, label_map=label_map)
    
    # Calculate class frequencies for gamma determination
    all_labels = []
    for doc_labels in train_labels:
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
    
    # Calculate class imbalance ratio
    if not label_counts:
        # Default value if no valid labels
        imbalance_ratio = 1.0
    else:
        most_common = label_counts.most_common(1)[0][1]
        least_common = min(label_counts.values())
        imbalance_ratio = most_common / max(1, least_common)
    
    # Adjust gamma based on imbalance
    # Higher imbalance -> higher gamma (more focus on hard examples)
    # Use a logarithmic scale to prevent extremely high values
    gamma = min(5.0, 1.0 + math.log(max(imbalance_ratio, 1.0), 10))
    
    return {
        'alpha': alpha,
        'gamma': gamma
    }


def calculate_label_smoothing_params(train_labels, label_map=None, num_classes=None):
    """
    Calculate optimal epsilon for Label Smoothing based on data distribution.
    
    Args:
        train_labels: List of label sequences from training data
        label_map: Optional mapping from label names to indices
        num_classes: Number of classes (if None, will be inferred)
        
    Returns:
        dict: Dictionary with optimal epsilon value
    """
    # Calculate class frequencies
    all_labels = []
    for doc_labels in train_labels:
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
    
    # Calculate number of classes with reasonable representation
    if not label_counts:
        # Default epsilon if no valid labels
        return {'epsilon': 0.1}
        
    # Calculate total count and distribution entropy
    total_count = sum(label_counts.values())
    
    # Calculate distribution entropy
    probs = [count / total_count for count in label_counts.values()]
    entropy = -sum(p * math.log(p, 2) for p in probs)
    max_entropy = math.log(len(label_counts), 2)
    
    # Normalize entropy to [0, 1]
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Adjust epsilon based on entropy
    # Higher entropy (more uniform distribution) -> lower epsilon
    # Lower entropy (more imbalanced) -> higher epsilon
    epsilon = min(0.3, max(0.05, 0.15 * (1 + (1 - norm_entropy))))
    
    return {'epsilon': epsilon}


def calculate_dice_loss_params(train_labels, label_map=None, num_classes=None):
    """
    Calculate optimal smoothing parameter for Dice Loss based on data distribution.
    
    Args:
        train_labels: List of label sequences from training data
        label_map: Optional mapping from label names to indices
        num_classes: Number of classes (if None, will be inferred)
        
    Returns:
        dict: Dictionary with optimal smooth value
    """
    # Calculate class frequencies
    all_labels = []
    for doc_labels in train_labels:
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
    
    if not label_counts:
        # Default smooth value if no valid labels
        return {'smooth': 1.0}
    
    # Calculate class imbalance ratio
    most_common = label_counts.most_common(1)[0][1]
    least_common = min(label_counts.values())
    imbalance_ratio = most_common / max(1, least_common)
    
    # Adjust smooth parameter based on imbalance
    # Higher imbalance -> higher smooth value (more smoothing for rare classes)
    # Scale to a reasonable range (0.5 - 2.0)
    smooth = min(2.0, max(0.5, 1.0 * math.log(imbalance_ratio, 10)))
    
    return {'smooth': smooth}


def calculate_weighted_ce_params(train_labels, label_map=None, num_classes=None):
    """
    Calculate optimal weights for Weighted Cross Entropy Loss based on data distribution.
    
    Args:
        train_labels: List of label sequences from training data
        label_map: Optional mapping from label names to indices
        num_classes: Number of classes (if None, will be inferred)
        
    Returns:
        dict: Dictionary with optimal weights
    """
    from ner_ancient_swedish.utils.data_utils import compute_class_weights
    
    # Calculate class frequencies and determine best weighting method
    all_labels = []
    for doc_labels in train_labels:
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
    
    if not label_counts:
        # Default weights if no valid labels
        return {'weights': None}
    
    # Calculate class imbalance ratio
    most_common = label_counts.most_common(1)[0][1]
    least_common = min(label_counts.values())
    imbalance_ratio = most_common / max(1, least_common)
    
    # Select weighting method based on imbalance ratio
    if imbalance_ratio > 50:  # Extreme imbalance
        # For extreme imbalance, use effective samples method
        method = 'balanced'
        beta = 0.9999  # High beta for extreme imbalance
    elif imbalance_ratio > 10:  # Significant imbalance
        # For significant imbalance, use direct inverse frequency
        method = 'inverse'
        beta = 0.999
    else:  # Moderate imbalance
        # For moderate imbalance, use square root of inverse frequency
        method = 'inverse_sqrt'
        beta = 0.99
    
    # Compute class weights
    weights = compute_class_weights(
        train_labels, 
        num_classes=num_classes, 
        label_map=label_map,
        method=method,
        beta=beta
    )
    
    return {'weights': weights, 'method': method}