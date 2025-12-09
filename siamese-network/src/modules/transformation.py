from torchvision import transforms

mean = [0.861, 0.861, 0.861]
std = [0.274, 0.274, 0.274]

train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=0,
        shear=10,
        translate=(0.1,0.1)
    ),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])