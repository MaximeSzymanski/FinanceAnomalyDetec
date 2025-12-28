import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNAutoEncoder(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64, latent_dim=32):
        super(GNNAutoEncoder, self).__init__()
        
        # --- Encoder ---
        # Compresse l'information en prenant compte des voisins
        self.enc1 = GCNConv(num_features, hidden_dim)
        self.enc2 = GCNConv(hidden_dim, latent_dim)
        
        # --- Decoder ---
        # Tente de reconstruire les features initiales
        # On peut utiliser GCNConv ici aussi pour diffuser l'info reconstruite
        self.dec1 = GCNConv(latent_dim, hidden_dim)
        self.dec2 = GCNConv(hidden_dim, num_features)

    def forward(self, x, edge_index):
        # x shape: [Num_Nodes, Window_Size]
        
        # Encoding
        z = self.enc1(x, edge_index)
        z = F.relu(z)
        z = self.enc2(z, edge_index)
        # z est maintenant la représentation latente "compacte" du marché
        
        # Decoding
        x_hat = self.dec1(z, edge_index)
        x_hat = F.relu(x_hat)
        x_hat = self.dec2(x_hat, edge_index)
        
        return x_hat, z