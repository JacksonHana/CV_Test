import os
from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = 'Dataset/Classification'

label_map = {
    ('tray','empty'):      0,
    ('tray','not_empty'):  1,
    ('tray','kakigori'):   2,
    ('dish','empty'):      3,
    ('dish','not_empty'):  4,
    ('dish','kakigori'):   5,
}
num_classes = len(label_map)  # =6

class TrayDishDataset(Dataset):
    def __init__(self, root_dir, label_map, transform=None):
        self.transform = transform
        self.samples = []
        for super_cls in os.listdir(root_dir):
            sc_path = os.path.join(root_dir, super_cls)
            if not os.path.isdir(sc_path): continue
            for sub_cls in os.listdir(sc_path):
                sub_path = os.path.join(sc_path, sub_cls)
                if not os.path.isdir(sub_path): continue
                lbl = label_map[(super_cls, sub_cls)]
                for fname in os.listdir(sub_path):
                    if fname.lower().endswith('.jpg'):
                        self.samples.append((os.path.join(sub_path, fname), lbl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
    

dataset = TrayDishDataset(DATA_DIR, label_map, transform=None)
print("Total images:", len(dataset))