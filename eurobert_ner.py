import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class EUROBERT_NER(nn.Module):
    def __init__(self, model_name, num_labels, hidden_dim=128,
                 model_checkpoint="EuroBERT/EuroBERT-2.1B"):
        super(EUROBERT_NER, self).__init__()
        self.model_checkpoint = model_checkpoint
        self.model_name = model_name
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
    
    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]
        # output: [batch_size, seq_len, num_labels]
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs

    def train(self, train_loader, val_loader, num_epochs=3, lr=0.001):
        # train_loader: DataLoader for training set
        # val_loader: DataLoader for validation set
        # num_epochs: number of epochs to train
        # lr: learning rate
        pass

    def evaluate(self, val_loader):
        # val_loader: DataLoader for validation set
        # return: accuracy, precision, recall, F1 score
        pass

    def predict(self, test_loader):
        # test_loader: DataLoader for test set
        # return: predictions
        pass



