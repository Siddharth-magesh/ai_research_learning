from modules.signature_triplet_dataset import create_signature_datasets_splits
from modules.signature_triplet_dataset import SignatureTripletDataset
from modules.transformation import train_transform, val_transform
from modules.embedding_network import SimpleEmbeddingNetwork
from visual import show_random_triplet
from train import training_loop_signature
from siamese_network import SiameseNetwork
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

signature_data_dir = kagglehub.dataset_download("siddharthmagesh/signature-verfication")
print("Signature data directory:", signature_data_dir)
full_dataset = SignatureTripletDataset(
    base_data_dir=signature_data_dir,
    triplets_per_user=100,
    transform=None
)

train_dataset, val_dataset = create_signature_datasets_splits(
    full_dataset=full_dataset,
    train_split=0.8,
    train_transform=train_transform,
    val_transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

print("Number of training samples:", len(train_dataset))
print("Number of validation samples:", len(val_dataset))

embedding_dim=128
embedding_net = SimpleEmbeddingNetwork(embedding_dim=embedding_dim)
siamese_model = SiameseNetwork(embedding_network=embedding_net).to(device)

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.Adam(params=siamese_model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)

num_epochs = 5
threshold_dist = 0.8

trained_siamese = training_loop_signature(
    model=siamese_model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fcn=triplet_loss,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=num_epochs,
    threshold=threshold_dist,
    device=device
)

show_random_triplet(train_loader)