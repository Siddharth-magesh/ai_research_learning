import random
import matplotlib.pyplot as plt

def show_random_triplet(data_loader):
    mean = [0.861, 0.861, 0.861]
    std = [0.274, 0.274, 0.274]
    model_batch = next(iter(data_loader))
    anchor, positive, negative = model_batch
    idx = random.randint(0, anchor.size(0), -1)
    
    imgs = [anchor[idx], positive[idx], negative[idx]]
    titles = ["Anchor (Real)", "Positive (Real)", "Negative (Fake)"]
    plt.figure(figsize=(8, 3))
    for i, (img, title) in enumerate(zip(imgs, titles), 1):
        plt.subplot(1, 3, i)
        img_np = img.permute(1, 2, 0).cpu.numpy()
        img_np = img_np * std[0] + mean[0]
        img_np = img_np.clip(0, 1)
        plt.imshow(img_np, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.show()