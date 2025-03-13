"""NER for Ancient Swedish texts using EuroBERT.

This package provides tools for Named Entity Recognition (NER) in Ancient Swedish texts
using the EuroBERT model.
"""

__version__ = "0.1.0"

# Import main classes and functions for easier access
from .models import EUROBERT_NER
from .utils.ner_utils import prepare_dataset, map_labels_to_ids, tokenize_and_align_labels
from .utils.pytorch_utils import NERDataset, create_dataloaders 