import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_network):
        super().__init__()
        self.embedding_network = embedding_network
        
    def forward(self, *inputs, triplet_bool=True):
        if triplet_bool:
            anchor, positive, negative = inputs
            z_a = self.embedding_network(anchor)
            z_p = self.embedding_network(positive)
            z_n = self.embedding_network(negative)
            return z_a, z_p, z_n
        else:
            img1, img2 = inputs
            z1 = self.embedding_network(img1)
            z2 = self.embedding_network(img2)
            return z1, z2
    
    def get_embedding(self, image):
        return self.embedding_network(image)