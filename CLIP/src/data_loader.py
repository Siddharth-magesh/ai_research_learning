
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

def simple_tokenizer(text, max_length=77):
    tokens = text.lower().split()
    idxs = [min(hash(w) % 49408, 49407) for w in tokens][:max_length]
    arr = np.zeros(max_length, dtype=np.int64)
    arr[:len(idxs)] = idxs
    return torch.tensor(arr, dtype=torch.long)


class Flickr30kDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None, max_length=77, max_samples=None):
        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length
        self.image_files = []
        self.captions = []
        exts = ['*.jpg', '*.jpeg', '*.png']
        for ext in exts:
            self.image_files += glob.glob(os.path.join(image_dir, '**', ext), recursive=True)
        self.image_files = sorted(set(self.image_files))
        self.caption_dict = {}
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img, caption = parts
                    self.caption_dict[img] = caption

        self.paired = []
        for img_path in self.image_files:
            img_name = os.path.basename(img_path)
            if img_name in self.caption_dict:
                self.paired.append((img_path, self.caption_dict[img_name]))

        if max_samples is not None and len(self.paired) > max_samples:
            self.paired = self.paired[:max_samples]

    def __len__(self):
        return len(self.paired)

    def __getitem__(self, idx):
        img_path, caption = self.paired[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        text = simple_tokenizer(caption, self.max_length)
        return image, text

def get_data_loader(image_dir, captions_file, batch_size=32, num_workers=2, max_samples=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = Flickr30kDataset(image_dir, captions_file, transform, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader
