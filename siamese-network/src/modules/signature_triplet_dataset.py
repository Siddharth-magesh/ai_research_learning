import torch
import random
import os
from PIL import Image
from torch.utils.data import Dataset

class SignatureTripletDataset(Dataset):
    def __init__(self, base_data_dir, triplets_per_user=100, transform=None, signature_map=None, user_ids=None):
        self.base_data_dir = base_data_dir
        self.triplets_per_user = triplets_per_user
        self.transform = transform
        
        if signature_map is None:
            self.signature_map = self._create_signature_map()
        else:
            self.signature_map = signature_map
            
        if user_ids is None:
            self.user_ids = sorted(list(self.signature_map.keys()))
        else:
            self.user_ids = list(user_ids)
    
    def __len__(self):
        return len(self.user_ids) * self.triplets_per_user
    
    def __getitem__(self, index):
        user_id = random.choice(self.user_ids)
        
        real_paths = self.signature_map[user_id]["real"]
        anchor_path, positive_path = random.sample(real_paths, 2)
        
        fake_paths = self.signature_map[user_id]["fake"]
        negative_path = random.choice(fake_paths)
        
        anchor_img = self._load_image(anchor_path)
        positive_img = self._load_image(positive_path)
        negative_img = self._load_image(negative_path)
        
        return anchor_img, positive_img, negative_img
    
    def _create_signature_map(self):
        signature_map = {}
        
        real_dir = os.path.join(self.base_data_dir, "Real")
        fake_dir = os.path.join(self.base_data_dir, "Fake")
        
        for user_id in os.listdir(real_dir):
            user_real_dir = os.path.join(real_dir, user_id)
            if not os.path.isdir(user_real_dir):
                continue
            
            real_paths = [
                os.path.join(user_real_dir, f)
                for f in os.listdir(user_real_dir)
                if f.lower().endswith((".png",".jpg",".jpeg"))
            ]
            
            user_fake_dir = os.path.join(fake_dir, user_id)
            fake_paths = []
            if os.path.isdir(user_fake_dir):
                fake_paths = [
                    os.path.join(user_fake_dir, f)
                    for f in os.listdir(user_fake_dir)
                    if f.lower().endswith((".png",".jpg","jpeg"))
                ]
            if len(real_paths) >= 2 and len(fake_paths) >= 1:
                signature_map[user_id] = {
                    "real" : real_paths,
                    "fake" : fake_paths
                }
        return signature_map
    
    def _load_image(self, path):
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
        return img
    
def create_signature_datasets_splits(full_dataset, train_split=0.8, train_transform=None, val_transform=None):
    all_users = full_dataset.user_ids.copy()
    random.shuffle(all_users)
    
    n_train = int(len(all_users) * train_split)
    train_users = all_users[:n_train]
    val_users = all_users[n_train:]
    
    train_dataset = SignatureTripletDataset(
        base_data_dir=full_dataset.base_data_dir,
        triplets_per_user=full_dataset.triplets_per_user,
        transform=train_transform,
        signature_map=full_dataset.signature_map,
        user_ids=train_users
    )
    
    val_dataset = SignatureTripletDataset(
        base_data_dir=full_dataset.base_data_dir,
        triplets_per_user=full_dataset.triplets_per_user,
        transform=val_transform,
        signature_map=full_dataset.signature_map,
        user_ids=val_users
    )
    
    return train_dataset, val_dataset