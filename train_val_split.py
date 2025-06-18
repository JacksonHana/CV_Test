import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from preprocessing import TrayDishDataset, label_map
DATA_DIR = 'Dataset/Classification'

dataset = TrayDishDataset(DATA_DIR, label_map, transform=None)

# get indices and labels
indices = list(range(len(dataset)))
labels  = [lbl for _, lbl in dataset.samples]

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

train_ds = Subset(dataset, train_idx)
val_ds   = Subset(dataset, val_idx)