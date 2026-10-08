import torch
from torch import nn


class DeepSetsStateEncoder(nn.Module):
    def __init__(self, glob_dim, cand_dim, hidden_dim, output_dim):
        super().__init__()
        self.glob_enc = nn.Sequential(
            nn.Linear(glob_dim, hidden_dim), nn.ReLU()
        )
        self.cand_enc = nn.Sequential(
            nn.Linear(cand_dim, hidden_dim), nn.ReLU()
        )
        # mean + max pool doubles the pooled dim
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

    def forward(self, x_glob, x_cand):
        g = self.glob_enc(x_glob)                    # (B, H)
        c = self.cand_enc(x_cand)                    # (B, M, H)
        c_mean = c.mean(dim=1)                       # (B, H)
        c_max  = c.max(dim=1).values                 # (B, H)
        x = torch.cat([g, c_mean, c_max], dim=-1)   # (B, 3H)
        return self.fusion(x)                        # (B, output_dim)
    
class FlatStateEncoder(nn.Module):
    def __init__(self, glob_dim, bin_dim, nbins, glob_hidden, bin_hidden, bin_out, output_dim, activation=None):
        super().__init__()
        self.activation = nn.ReLU if activation is None else activation
        # Global encoder
        self.state_enc = nn.Sequential(
            nn.Linear(glob_dim, glob_hidden),
            self.activation(),
            nn.Linear(glob_hidden, glob_hidden),
            self.activation()
        )
        # Bin encoder - processes each bin's features independently and then flattens
        self.bin_enc = nn.Sequential(
            nn.Linear(bin_dim, bin_hidden),
            self.activation(),
            nn.Linear(bin_hidden, bin_out),
            self.activation()
        )
        # Concatenate global and entire sky features (ie, the flattened bin features) and output latent state representation
        self.fusion_net = nn.Sequential(
            nn.Linear(glob_hidden + bin_out * nbins, output_dim),
            self.activation()
        )
        
    def forward(self, x_glob, x_bin):
        x_context = self.glob_enc(x_glob) # shape (batch, glob_hidden)
        x_binfeats = self.bin_enc(x_bin) # shape (batch, nbinbs, bin_hidden)
        x_binfeats_flat = x_binfeats.view(x_binfeats.size(0), -1) # Shape: (batch_size, nbins * bin_hidden)
        x = torch.cat([x_context, x_binfeats_flat], dim=-1) #shape (batch, glob_hidden + nbins * bin_hidden)
        x_latent = self.fusion_net(x)
        return x_latent