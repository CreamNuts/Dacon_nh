import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from kobert_transformers import get_tokenizer

class NewsDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels.values if labels is not None else None
    
    def __len__(self):
        return len(self.encodings[list(self.encodings.keys())[0]])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

def create_loader(data_dir, mode, batch_size, ratio=0.8):
    dataset = pd.read_csv(data_dir)
    tokenizer = get_tokenizer()
    if mode == 'train':
        data, data_label = dataset['content'], dataset['info']
        encodings = tokenizer(list(data.values), truncation=True, padding=True)
        dataset = NewsDataset(encodings, data_label)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return data_loader, None
    elif mode == 'val':
        data, val, data_label, val_label = train_test_split(
            dataset['content'], dataset['info'], train_size=ratio, stratify=dataset['info']
            )
        encodings = tokenizer(list(data.values), truncation=True, padding=True)
        val_encodings = tokenizer(list(val.values), truncation=True, padding=True)  
        dataset = NewsDataset(encodings, data_label)
        val_dataset = NewsDataset(val_encodings, val_label)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return data_loader, val_loader
    else:
        data = dataset['content']
        encodings = tokenizer(list(data.values), truncation=True, padding=True)
        dataset = NewsDataset(encodings)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return data_loader, None