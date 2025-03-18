import os
import torch
import optuna
from optuna.trial import Trial
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import json
from tqdm import tqdm
import numpy as np
from transformers import AutoModel
import matplotlib.pyplot as plt

from ner_ancient_swedish.models.eurobert_ner import EUROBERT_NER

logger = logging.getLogger(__name__)

class OptunaHyperparameterTuner:
    """
    Class for hyperparameter optimization of the EUROBERT_NER model using Optuna.
    """
    def __init__(
        self,
        train_dataset,
        val_dataset,
        model_name,
        num_labels,
        label_list,
        model_checkpoint="EuroBERT/EuroBERT-610m",
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir="optuna_results",
        n_trials=100,
        timeout=None,  # in seconds, None means no timeout
    ):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            model_name: Name of the model
            num_labels: Number of labels for classification
            label_list: List of label names
            model_checkpoint: Pretrained model checkpoint
            device: Device to run the model on
            output_dir: Directory to save the results
            n_trials: Number of trials for hyperparameter search
            timeout: Timeout for the search in seconds
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_name = model_name
        self.num_labels = num_labels
        self.label_list = label_list
        self.model_checkpoint = model_checkpoint
        self.device = device
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.timeout = timeout
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def objective(self, trial: Trial):
        """
        Objective function for Optuna to minimize.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            float: Validation loss (to be minimized)
        """
        # Sample hyperparameters
        hyperparams = self.sample_hyperparameters(trial)
        
        # Create data loaders with the sampled batch size
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=hyperparams["batch_size"],
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn if hasattr(self.train_dataset, "collate_fn") else None
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=hyperparams["batch_size"],
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn if hasattr(self.val_dataset, "collate_fn") else None
        )
        
        # Create model with the sampled architecture
        model = self.create_model(hyperparams)
        
        # Train the model
        model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=hyperparams["num_epochs"],
            initial_lr=hyperparams["initial_lr"],
            finetune_lr=hyperparams["finetune_lr"],
            finetune_after_epoch=hyperparams["finetune_after_epoch"],
            device=self.device,
            label_list=self.label_list
        )
        
        # Evaluate on validation set
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        metrics = model.evaluate(val_loader, criterion, self.device, self.label_list)
        
        # Report intermediate values
        trial.report(metrics["f1"], hyperparams["num_epochs"])
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return 1 - metrics["f1"]  # We want to maximize F1 score
    
    def sample_hyperparameters(self, trial: Trial):
        """
        Sample hyperparameters for the current trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            dict: Sampled hyperparameters
        """
        # Training hyperparameters
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        num_epochs = trial.suggest_int("num_epochs", 3, 15)
        
        # Learning rates
        initial_lr = trial.suggest_float("initial_lr", 1e-5, 1e-2, log=True)
        finetune_lr = trial.suggest_float("finetune_lr", 1e-6, 1e-4, log=True)
        finetune_after_epoch = trial.suggest_int("finetune_after_epoch", 1, max(1, num_epochs - 1))
        
        # Architecture hyperparameters
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
        
        # Additional architecture hyperparameters
        use_extra_layer = trial.suggest_categorical("use_extra_layer", [True, False])
        if use_extra_layer:
            extra_layer_dim = trial.suggest_categorical("extra_layer_dim", [64, 128, 256])
        else:
            extra_layer_dim = None
            
        # Optimizer hyperparameters
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        
        return {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "initial_lr": initial_lr,
            "finetune_lr": finetune_lr,
            "finetune_after_epoch": finetune_after_epoch,
            "hidden_dim": hidden_dim,
            "dropout_rate": dropout_rate,
            "use_extra_layer": use_extra_layer,
            "extra_layer_dim": extra_layer_dim,
            "weight_decay": weight_decay
        }
    
    def create_model(self, hyperparams):
        """
        Create a model with the given hyperparameters.
        
        Args:
            hyperparams: Dictionary of hyperparameters
            
        Returns:
            EUROBERT_NER: Model with the specified hyperparameters
        """
        # Create custom model with the sampled architecture
        class CustomEUROBERT_NER(EUROBERT_NER):
            def __init__(self, model_name, num_labels, hidden_dim, dropout_rate, 
                         use_extra_layer, extra_layer_dim, model_checkpoint):
                nn.Module.__init__(self)
                self.model_checkpoint = model_checkpoint
                self.model_name = model_name
                self.num_labels = num_labels
                self.hidden_dim = hidden_dim
                self.model = AutoModel.from_pretrained(model_checkpoint)
                
                # Customize the classifier head based on hyperparameters
                if use_extra_layer and extra_layer_dim is not None:
                    self.classifier = nn.Sequential(
                        nn.Linear(self.model.config.hidden_size, extra_layer_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.Linear(extra_layer_dim, num_labels)
                    )
                else:
                    self.classifier = nn.Sequential(
                        nn.Dropout(dropout_rate),
                        nn.Linear(self.model.config.hidden_size, num_labels)
                    )
            
            def forward(self, input_ids, attention_mask=None):
                outputs = self.model(input_ids, attention_mask=attention_mask)
                sequence_output = outputs.last_hidden_state
                logits = self.classifier(sequence_output)
                return logits
        
        # Create model with the sampled hyperparameters
        model = CustomEUROBERT_NER(
            model_name=self.model_name,
            num_labels=self.num_labels,
            hidden_dim=hyperparams["hidden_dim"],
            dropout_rate=hyperparams["dropout_rate"],
            use_extra_layer=hyperparams["use_extra_layer"],
            extra_layer_dim=hyperparams["extra_layer_dim"],
            model_checkpoint=self.model_checkpoint
        )
        
        return model
    
    def optimize(self):
        """
        Run the hyperparameter optimization.
        
        Returns:
            dict: Best hyperparameters
        """
        # Create a pruner
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        
        # Create a study
        study = optuna.create_study(
            direction="minimize", 
            pruner=pruner,
            study_name=f"{self.model_name}_optimization"
        )
        
        # Optimize
        study.optimize(
            self.objective, 
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        # Get best hyperparameters
        best_hyperparams = study.best_params
        best_f1 = 1 - study.best_value
        
        # Print and save results
        print("Best hyperparameters:", best_hyperparams)
        print(f"Best F1 score: {best_f1:.4f}")
        
        # Save the best hyperparameters
        with open(os.path.join(self.output_dir, "best_hyperparams.json"), "w") as f:
            json.dump(best_hyperparams, f, indent=4)
        
        # Save the study
        with open(os.path.join(self.output_dir, "study.pkl"), "wb") as f:
            import pickle
            pickle.dump(study, f)
        
        # Plot optimization history
        try:
            import matplotlib.pyplot as plt
            
            # Plot optimization history
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.savefig(os.path.join(self.output_dir, "optimization_history.png"))
            
            # Plot parameter importances
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.savefig(os.path.join(self.output_dir, "param_importances.png"))
            
        except Exception as e:
            print(f"Warning: Could not create visualization plots. Error: {e}")
        
        return best_hyperparams
    
    def train_with_best_params(self, output_model_path=None):
        """
        Train a model with the best hyperparameters.
        
        Args:
            output_model_path: Path to save the trained model
            
        Returns:
            EUROBERT_NER: Trained model
        """
        # Load best hyperparameters
        best_hyperparams_path = os.path.join(self.output_dir, "best_hyperparams.json")
        if not os.path.exists(best_hyperparams_path):
            raise FileNotFoundError(f"Best hyperparameters file not found at {best_hyperparams_path}. "
                                   "Run optimize() first.")
        
        with open(best_hyperparams_path, "r") as f:
            best_hyperparams = json.load(f)
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=best_hyperparams["batch_size"],
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn if hasattr(self.train_dataset, "collate_fn") else None
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=best_hyperparams["batch_size"],
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn if hasattr(self.val_dataset, "collate_fn") else None
        )
        
        # Create model
        model = self.create_model(best_hyperparams)
        
        # Train the model
        model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=best_hyperparams["num_epochs"],
            initial_lr=best_hyperparams["initial_lr"],
            finetune_lr=best_hyperparams["finetune_lr"],
            finetune_after_epoch=best_hyperparams["finetune_after_epoch"],
            device=self.device,
            label_list=self.label_list
        )
        
        # Save the model if path is provided
        if output_model_path:
            torch.save(model.state_dict(), output_model_path)
            print(f"Model saved to {output_model_path}")
        
        return model


def run_hyperparameter_search(
    train_dataset,
    val_dataset,
    model_name,
    num_labels,
    label_list,
    model_checkpoint="EuroBERT/EuroBERT-610m",
    n_trials=50,
    output_dir="optuna_results",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Run hyperparameter search and return the best hyperparameters.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        model_name: Name of the model
        num_labels: Number of labels for classification
        label_list: List of label names
        model_checkpoint: Pretrained model checkpoint
        n_trials: Number of trials for hyperparameter search
        output_dir: Directory to save the results
        device: Device to run the model on
        
    Returns:
        dict: Best hyperparameters
        EUROBERT_NER: Trained model with best hyperparameters
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "hyperparameter_search.log")),
            logging.StreamHandler()
        ]
    )
    
    # Create hyperparameter tuner
    tuner = OptunaHyperparameterTuner(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name=model_name,
        num_labels=num_labels,
        label_list=label_list,
        model_checkpoint=model_checkpoint,
        device=device,
        output_dir=output_dir,
        n_trials=n_trials
    )
    
    # Run optimization
    best_hyperparams = tuner.optimize()
    
    # Train model with best hyperparameters
    model = tuner.train_with_best_params(output_model_path=os.path.join(output_dir, f"{model_name}_best.pt"))
    
    return best_hyperparams, model


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    
    # This should be replaced with your actual dataset
    class DummyDataset:
        def __init__(self, data, tokenizer, max_length=128):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            item = self.data[idx]
            encoding = self.tokenizer(
                item["text"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Remove batch dimension added by tokenizer
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            
            # Create labels tensor
            labels = torch.tensor(item["labels"], dtype=torch.long)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
        def collate_fn(self, batch):
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            labels = torch.stack([item["labels"] for item in batch])
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
    
    # Replace this with your actual data loading code
    def load_dummy_data():
        tokenizer = AutoTokenizer.from_pretrained("EuroBERT/EuroBERT-610m")
        
        # Example data format
        train_data = [
            {"text": "Sample text 1", "labels": [0, 1, 2, 0]},
            {"text": "Sample text 2", "labels": [1, 0, 0, 2]}
        ]
        
        val_data = [
            {"text": "Sample validation text", "labels": [0, 2, 1, 0]}
        ]
        
        train_dataset = DummyDataset(train_data, tokenizer)
        val_dataset = DummyDataset(val_data, tokenizer)
        
        return train_dataset, val_dataset
    
    # Replace with your actual label list
    label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    
    print("This is an example script. Replace the dummy data with your actual data.")
    print("Uncomment the following code to run the hyperparameter search:")
    """
    # Load datasets
    train_dataset, val_dataset = load_dummy_data()
    
    # Run hyperparameter search
    best_hyperparams, best_model = run_hyperparameter_search(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name="eurobert_ner",
        num_labels=len(label_list),
        label_list=label_list,
        n_trials=50,
        output_dir="optuna_results"
    )
    
    print("Best hyperparameters:", best_hyperparams)
    """
