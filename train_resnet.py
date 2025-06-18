import torch, torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models
from preprocessing import num_classes
from transform_dataloader import train_loader, val_loader


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.resnet18(pretrained=True)
in_feats = model.fc.in_features
model.fc = nn.Linear(in_feats, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_one_epoch():
    model.train()
    running = 0.0
    for imgs, lbls in tqdm(train_loader):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, lbls)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / len(train_loader)

def validate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
    return correct / total


if __name__ == "__main__":
    epochs = 100
    best_acc = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch()
        val_acc    = validate()
        print(f"Epoch {epoch} • Train loss: {train_loss:.4f} • Val acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_traydish_resnet18.pth')
            print("Saved new best model")