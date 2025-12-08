import torch
from src.clip import CLIP
from src.data_loader import get_data_loaders
from src.vision_transformer import VisionTransformer
from src.text_transformer import TextTransformer

def evaluate(model, data_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, captions in data_loader:
            image_features, text_features = model(images, captions)
            # Example: cosine similarity for retrieval
            similarity = torch.matmul(image_features, text_features.T)
            preds = similarity.argmax(dim=1)
            correct += (preds == torch.arange(len(captions))).sum().item()
            total += len(captions)
    print(f"Accuracy: {correct/total:.4f}")

if __name__ == "__main__":
    images_dir = 'path/to/flickr30k/images'  # Update with actual path
    captions_file = 'path/to/flickr30k/captions.txt'  # Update with actual path
    data_loader = get_data_loaders(images_dir, captions_file)
    vision_encoder = VisionTransformer()
    text_encoder = TextTransformer()
    model = CLIP(vision_encoder, text_encoder)
    # Optionally load trained weights
    # model.load_state_dict(torch.load('clip_model.pth'))
    evaluate(model, data_loader)
