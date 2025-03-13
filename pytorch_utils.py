import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# Create a PyTorch Dataset class
class NERDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Convert to tensors
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
        labels = torch.tensor(item['labels'], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    

# Custom collate function to handle variable length sequences
def collate_fn(batch, tokenizer):
    # Extract all input_ids, attention_masks, and labels from the batch
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Pad sequences to the maximum length in the batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # Use -100 for padding labels
    
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels_padded
    }


# Create DataLoaders with the custom collate function
def create_dataloaders(train_dataset, val_dataset, batch_size, tokenizer):
    train_pytorch_dataset = NERDataset(train_dataset)
    val_pytorch_dataset = NERDataset(val_dataset)

    train_dataloader = DataLoader(
        train_pytorch_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    val_dataloader = DataLoader(
        val_pytorch_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    return train_dataloader, val_dataloader


    
