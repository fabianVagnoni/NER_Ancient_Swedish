import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
# Import the custom evaluation function
from ner_ancient_swedish.utils.evaluation_utils import compute_metrics
# Import custom loss functions
from ner_ancient_swedish.models.loss_functions import FocalLoss, LabelSmoothingCrossEntropy, DiceLoss

class EUROBERT_LSTM(nn.Module):
    def __init__(self, model_name, 
                 num_labels, 
                 hidden_dim=128,
                 bidirectional=True,
                 num_layers=2,
                 model_checkpoint="EuroBERT/EuroBERT-610m"):
        super(EUROBERT_LSTM, self).__init__()
        self.model_checkpoint = model_checkpoint
        self.model_name = model_name
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.model = AutoModel.from_pretrained(model_checkpoint)
        self.lstm = nn.LSTM(self.model.config.hidden_size,
                             hidden_dim, 
                             batch_first=True,
                             num_layers=num_layers,
                             bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, num_labels)
        else:
            self.fc = nn.Linear(hidden_dim, num_labels)
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]
        # output: [batch_size, seq_len, num_labels]
        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        if self.bidirectional:
            lstm_output = lstm_output[:, :, :self.hidden_dim] + lstm_output[:, :, self.hidden_dim:]
            lstm_output = lstm_output.contiguous().view(lstm_output.size(0), -1, self.hidden_dim)
        else:
            lstm_output = lstm_output[:, :, :self.hidden_dim]
        logits = self.fc(lstm_output)
        return logits

    def fit(self, train_loader, val_loader, 
              criterion,
              num_epochs=10, 
              initial_lr=1e-3, 
              finetune_lr=5e-5, 
              finetune_after_epoch=3, 
              device='cuda',
              label_list=None,
              class_weights=None):  # Optional class weights for imbalanced data
        """
        Two-phase training strategy:
        1. First phase: Freeze pretrained model, train only the classification head
        2. Second phase: Unfreeze pretrained model, train entire model with lower learning rate
        
        Args:
            train_loader: DataLoader for training set
            val_loader: DataLoader for validation set
            num_epochs: total number of epochs to train
            initial_lr: learning rate for the first phase (only classification head)
            finetune_lr: learning rate for the second phase (full model fine-tuning)
            finetune_after_epoch: after which epoch to start fine-tuning the pretrained model
            device: device to train on ('cuda' or 'cpu')
            label_list: list of label names corresponding to label indices
            loss_type: type of loss function ('ce', 'focal', 'dice', 'label_smoothing')
            class_weights: weights for each class (for weighted loss functions)
        """

        self.to(device)
        
        # Initialize the selected loss function
        criterion = criterion
            
        best_f1 = 0.0
        
        # Phase 1: Freeze pretrained model, train only classification head
        print("Phase 1: Training only the classification head")
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Only optimize classification head parameters
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=initial_lr)
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Phase 2: Unfreeze pretrained model after specified epoch
            if epoch == finetune_after_epoch:
                print(f"Phase 2: Unfreezing pretrained model and reducing learning rate")
                for param in self.model.parameters():
                    param.requires_grad = True
                
                # Reset optimizer with lower learning rate for all parameters
                optimizer = optim.AdamW([
                    {'params': self.model.parameters(), 'lr': finetune_lr},  # Lower LR for pretrained
                    {'params': self.fc.parameters(), 'lr': initial_lr}       # Keep initial LR for head
                ])
            
            # Training loop
            self.train_epoch(train_loader, optimizer, criterion, device, epoch)
            
            # Validation
            val_metrics = self.evaluate(val_loader, criterion, device, label_list)
            
            print(f"Validation metrics: " + 
                  f"Loss: {val_metrics['loss']:.4f}, " +
                  f"Accuracy: {val_metrics['accuracy']:.4f}, " +
                  f"F1: {val_metrics['f1']:.4f}")
            
            # Save the best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(self.state_dict(), f"{self.model_name}_best.pt")
                print(f"Saved best model with F1: {best_f1:.4f}")
    
    def train_epoch(self, train_loader, optimizer, criterion, device, epoch):
        """Train for one epoch"""
        self.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = self(input_ids, attention_mask)
            
            # Calculate loss using the selected criterion
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if (batch_idx + 1) % 50 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self, val_loader, criterion=None, device='cuda', label_list=None):
        """
        Evaluate the model on validation data using custom evaluation metrics
        
        Args:
            val_loader: DataLoader for validation set
            criterion: loss function (if None, only metrics are computed)
            device: device to evaluate on
            label_list: list of label names corresponding to label indices
            
        Returns:
            dict: metrics including accuracy, precision, recall, F1 score
        """
        if label_list is None:
            # Fallback to default evaluation if label_list is not provided
            return self._evaluate_default(val_loader, criterion, device)
            
        self.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                logits = self(input_ids, attention_mask)
                
                # Calculate loss if criterion is provided
                if criterion:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, 
                        labels.view(-1), 
                        torch.tensor(criterion.ignore_index).type_as(labels)
                    )
                    
                    loss = criterion(active_logits, active_labels)
                    total_loss += loss.item()
                
                # Process each batch item individually
                batch_preds = torch.argmax(logits, dim=2).cpu().numpy()
                batch_labels = labels.cpu().numpy()
                
                all_predictions.extend(batch_preds)
                all_labels.extend(batch_labels)
        
        # Prepare metrics
        metrics = {}
        if criterion:
            metrics['loss'] = total_loss / len(val_loader)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(all_predictions, all_labels)
        ]
        
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(all_predictions, all_labels)
        ]
        
        # Calculate token-level accuracy
        token_accuracy = self._calculate_token_accuracy(true_labels, true_predictions)
        metrics['accuracy'] = token_accuracy
        
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
                elif label != 'O' and (current_entity is None or current_entity['label'] != label):
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
                elif label != 'O' and (current_entity is None or current_entity['label'] != label):
                    if current_entity:
                        doc_entities.append(current_entity)
                    current_entity = {'label': label, 'start': i, 'end': i}
                elif label != 'O' and current_entity and current_entity['label'] == label:
                    current_entity['end'] = i
            
            if current_entity:
                doc_entities.append(current_entity)
            
            pred_entities.append(doc_entities)
        
        # Create evaluator and compute metrics
        from nervaluate import Evaluator
        evaluator = Evaluator(true_entities, pred_entities, tags=list(set(label_list) - {'O'}))
        results_all = evaluator.evaluate()
        
        results = results_all[0]  # Overall results
        results_per_tag = results_all[1]  # Results per tag
        
        # Extract overall metrics
        metrics['precision'] = results['strict']['precision']
        metrics['recall'] = results['strict']['recall']
        metrics['f1'] = results['strict']['f1']
        
        # Print per-tag results
        print("\nResults per entity type:")
        for key, value in results_per_tag.items():
            print(f"{key}: {results_per_tag[key]['strict']}")
        
        return metrics
    
    def _calculate_token_accuracy(self, true_labels, true_predictions):
        """Calculate token-level accuracy"""
        correct = 0
        total = 0
        
        for doc_labels, doc_preds in zip(true_labels, true_predictions):
            for label, pred in zip(doc_labels, doc_preds):
                if label == pred:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0

    def _evaluate_default(self, val_loader, criterion=None, device='cuda'):
        """
        Original evaluation method using token-level metrics (fallback method)
        """
        self.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                logits = self(input_ids, attention_mask)
                
                # Calculate loss if criterion is provided
                if criterion:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, 
                        labels.view(-1), 
                        torch.tensor(criterion.ignore_index).type_as(labels)
                    )
                    
                    loss = criterion(active_logits, active_labels)
                    total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=2)
                
                # Collect predictions and labels for metrics calculation
                for i, mask in enumerate(attention_mask):
                    active_preds = predictions[i, mask == 1].cpu().numpy()
                    active_labels_i = labels[i, mask == 1].cpu().numpy()
                    
                    # Exclude ignored indices
                    valid_indices = active_labels_i != -100
                    if valid_indices.any():
                        all_predictions.extend(active_preds[valid_indices])
                        all_labels.extend(active_labels_i[valid_indices])
        
        # Calculate metrics
        metrics = {}
        if criterion:
            metrics['loss'] = total_loss / len(val_loader)
        
        if all_predictions:
            metrics['accuracy'] = accuracy_score(all_labels, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted'
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
        
        return metrics

    def predict(self, test_loader, device='cuda', label_list=None):
        """
        Generate predictions for test data
        
        Args:
            test_loader: DataLoader for test set
            device: device to predict on
            label_list: list of label names corresponding to label indices
            
        Returns:
            list: predictions for each token in the test set
            list: token ids from the input (for mapping back to original text)
            dict: metrics if labels are provided in the test_loader
        """
        self.eval()
        all_predictions = []
        all_token_ids = []
        all_labels = []
        has_labels = False
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Check if labels are provided
                if 'labels' in batch:
                    has_labels = True
                    labels = batch['labels'].to(device)
                
                # Forward pass
                logits = self(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=2)
                
                # Collect predictions (accounting for padding)
                for i, mask in enumerate(attention_mask):
                    active_indices = mask == 1
                    active_preds = predictions[i, active_indices].cpu().numpy()
                    active_ids = input_ids[i, active_indices].cpu().numpy()
                    
                    all_predictions.append(active_preds)
                    all_token_ids.append(active_ids)
                
                # Store labels if available
                if has_labels:
                    batch_labels = labels.cpu().numpy()
                    all_labels.extend([labels[i, mask == 1].cpu().numpy() for i, mask in enumerate(attention_mask)])
        
        # If we have labels and a label list, compute metrics
        if has_labels and label_list is not None:
            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(all_predictions, all_labels)
            ]
            
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(all_predictions, all_labels)
            ]
            
            # Calculate token-level accuracy
            token_accuracy = self._calculate_token_accuracy(true_labels, true_predictions)
            
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
                    elif label != 'O' and (current_entity is None or current_entity['label'] != label):
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
                    elif label != 'O' and (current_entity is None or current_entity['label'] != label):
                        if current_entity:
                            doc_entities.append(current_entity)
                        current_entity = {'label': label, 'start': i, 'end': i}
                    elif label != 'O' and current_entity and current_entity['label'] == label:
                        current_entity['end'] = i
                
                if current_entity:
                    doc_entities.append(current_entity)
                
                pred_entities.append(doc_entities)
            
            # Create evaluator and compute metrics
            from nervaluate import Evaluator
            evaluator = Evaluator(true_entities, pred_entities, tags=list(set(label_list) - {'O'}))
            results_all = evaluator.evaluate()
            
            results = results_all[0]  # Overall results
            results_per_tag = results_all[1]  # Results per tag
            
            # Create metrics dict
            metrics = {
                'accuracy': token_accuracy,
                'precision': results['strict']['precision'],
                'recall': results['strict']['recall'],
                'f1': results['strict']['f1']
            }
            
            # Print per-tag results
            print("\nResults per entity type:")
            for key, value in results_per_tag.items():
                print(f"{key}: {results_per_tag[key]['strict']}")
            
            return all_predictions, all_token_ids, metrics
        
        return all_predictions, all_token_ids