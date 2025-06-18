import torch.nn as nn
import torchvision.models as models
import torch
from preprocessing import num_classes

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.resnet18(pretrained=True)
in_feats = model.fc.in_features
model.fc = nn.Linear(in_feats, num_classes)
model = model.to(device)
