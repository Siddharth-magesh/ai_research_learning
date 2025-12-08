import torch
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def dataset_summary(image_dir, captions_file):
    import os
    import random
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    print("\n--- Dataset Directory Structure ---")
    for root, dirs, files in os.walk(image_dir):
        level = root.replace(image_dir, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        # Only print the number of files in each directory, not all filenames
        subindent = ' ' * 4 * (level + 1)
        print(f"{subindent}[{len(files)} files]")

    # Count images
    exts = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(exts)]
    print(f"\nImage count: {len(image_files)}")


    # Count captions and display a few
    with open(captions_file, 'r', encoding='utf-8') as f:
        captions = [line.strip() for line in f if line.strip()]
    print(f"Captions count: {len(captions)}")
    print("\nSample captions:")
    for cap in captions[:10]:
        print(f"  {cap}")

    # Display first 10 images and their captions
    print("\nDisplaying first 10 images and their captions:")
    plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(2, 5)
    # Build a mapping from image name to list of captions
    from collections import defaultdict
    img2caps = defaultdict(list)
    for line in captions:
        if '\t' in line:
            img, cap = line.split('\t', 1)
            img2caps[img].append(cap.strip())
        elif ',' in line and not line.startswith('image'):
            img, cap = line.split(',', 1)
            img2caps[img.strip()] .append(cap.strip())
    for idx, fname in enumerate(image_files[:10]):
        img_path = os.path.join(image_dir, fname)
        try:
            img = Image.open(img_path)
            ax = plt.subplot(gs[idx])
            ax.imshow(img)
            ax.set_title(fname)
            ax.axis('off')
            # Print up to 2 captions for each image
            caps = img2caps.get(fname, [])
            for cidx, cap in enumerate(caps[:2]):
                print(f"{fname} [{cidx+1}]: {cap}")
        except Exception as e:
            print(f"Failed to open {fname}: {e}")
    plt.tight_layout()
    plt.show()

    # EDA: Check for images without captions and vice versa
    caption_names = set([line.split('\t')[0] for line in captions if '\t' in line])
    image_names = set(image_files)
    missing_captions = image_names - caption_names
    missing_images = caption_names - image_names
    print(f"\nImages without captions: {len(missing_captions)}")
    if missing_captions:
        print(f"  {list(missing_captions)[:5]}")
    print(f"Captions without images: {len(missing_images)}")
    if missing_images:
        print(f"  {list(missing_images)[:5]}")