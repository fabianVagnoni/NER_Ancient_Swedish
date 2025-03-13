from datasets import load_dataset
from ner_utils import prepare_dataset, map_labels_to_ids, tokenize_and_align_labels
from transformers import AutoTokenizer
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from pytorch_utils import create_dataloaders

dataset = load_dataset("csv", data_files="A2_train.csv")
print(dataset)

processed_dataset = prepare_dataset(dataset)
print(processed_dataset)

from datasets import ClassLabel, Sequence

label_list = ['O', 'EVN', 'LOC', 'MSR-AREA', 'MSR-DIST', 'MSR-LEN', 'MSR-MON',
              'MSR-OTH', 'MSR-VOL', 'MSR-WEI', 'OCC', 'ORG-COMP', 'ORG-INST',
              'ORG-OTH', 'PER', 'SYMP', 'TME-DATE', 'TME-INTRV', 'TME-TIME', 'WRK']  # Replace with your actual label list

# Create a label map to convert string labels to integers
label_map = {label: i for i, label in enumerate(label_list)}

# Pass the label_map as a parameter to the map function
df = processed_dataset.map(lambda example: map_labels_to_ids(example, label_map))

# Update the 'labels' feature type to ClassLabel
#df = df.cast_column("labels", ClassLabel(names=label_list))
df = df.cast_column("labels", Sequence(feature=ClassLabel(names=label_list)))
print(df)

label_list1 = df["train"].features["labels"].feature.names
print(label_list1)

model_checkpoint = "distilbert-base-uncased"
batch_size = 8
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenize and align the labels for the entire dataset
tokenized_datasets = df.map(
    lambda examples: tokenize_and_align_labels(examples, tokenizer),
    batched=True,
    remove_columns=df["train"].column_names
)

# Split the dataset into train and validation sets
train_dataset = tokenized_datasets["train"].shuffle(seed=42)
train_val_split = train_dataset.train_test_split(test_size=0.1)
train_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]

train_dataloader, val_dataloader = create_dataloaders(train_dataset, val_dataset, batch_size, tokenizer)


# Define a simple NER model
class NERModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, pad_idx=0):
        super(NERModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len]
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # Pack padded sequence if we have an attention mask
        if attention_mask is not None:
            # Get sequence lengths from attention mask
            seq_lengths = attention_mask.sum(dim=1).cpu()
            
            # Pack the embedded sequence
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, seq_lengths, batch_first=True, enforce_sorted=False
            )
            
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            
            # Unpack the sequence
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(embedded)
        
        output = self.dropout(output)
        predictions = self.fc(output)  # [batch_size, seq_len, num_labels]
        
        return predictions

# Initialize the model
vocab_size = tokenizer.vocab_size
embedding_dim = 128
hidden_dim = 256
num_labels = len(label_list)
pad_idx = tokenizer.pad_token_id

model = NERModel(vocab_size, embedding_dim, hidden_dim, num_labels, pad_idx)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding tokens (-100)
optimizer = optim.Adam(model.parameters(), lr=2e-3)

# Training function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get data
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask)
        
        # Reshape for loss calculation
        predictions = predictions.view(-1, predictions.shape[-1])
        labels = labels.view(-1)
        
        # Calculate loss
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            predictions = model(input_ids, attention_mask)
            
            # Save predictions and labels for metrics
            batch_predictions = torch.argmax(predictions, dim=-1)
            
            # Only consider non-padding tokens
            for i in range(len(labels)):
                valid_indices = labels[i] != -100
                all_predictions.extend(batch_predictions[i, valid_indices].cpu().numpy())
                all_labels.extend(labels[i, valid_indices].cpu().numpy())
            
            # Reshape for loss calculation
            predictions = predictions.view(-1, predictions.shape[-1])
            labels = labels.view(-1)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
    
    # Calculate accuracy
    accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels) if all_labels else 0
    
    return epoch_loss / len(dataloader), accuracy

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

# Training loop
num_epochs = 5
best_val_loss = float('inf')

print("Starting training...")
for epoch in range(num_epochs):
    # Train
    train_loss = train(model, train_dataloader, optimizer, criterion, device)
    
    # Evaluate
    val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_ner_model.pt')
        print("  Saved best model!")

print("Training complete!")

# Load the best model
model.load_state_dict(torch.load('best_ner_model.pt'))

# Function for inference
def predict_ner(text, model, tokenizer, label_list, device):
    model.eval()
    words = text.split()
    
    # Tokenize
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=-1)
    
    # Convert predictions to labels
    word_ids = inputs.word_ids()
    previous_word_idx = None
    predicted_labels = []
    
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        
        predicted_labels.append(label_list[predictions[0, idx].item()])
        previous_word_idx = word_idx
    
    # Combine words and labels
    result = []
    for word, label in zip(words, predicted_labels):
        result.append((word, label))
    
    return result

# Test the model with a sample text
if len(df["train"]) > 0:
    sample_text = " ".join(df["train"][0]["text"])
    print(f"Sample text: {sample_text}")
    predictions = predict_ner(sample_text, model, tokenizer, label_list, device)
    print("Predictions:")
    for word, label in predictions:
        print(f"{word}: {label}")

