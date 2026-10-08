import numpy as np
import torch.nn as nn
import torch
import logging

from skyway.rl.neural_nets.encoders import FlatStateEncoder
logger = logging.getLogger(__name__)

def build_mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int,
              layernorm: bool = True,
              activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    """MLP with optional LayerNorm per hidden layer (important for RL stability)."""
    layers: list[nn.Module] = []
    d = in_dim
    if len(hidden) < 1:
        raise ValueError("Number of layers cannot be less than 1")
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation())
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int,
                 layernorm: bool = True,
                 activation: type[nn.Module] = nn.ReLU):
        super().__init__()
        self.net = build_mlp(in_dim, hidden, out_dim, layernorm, activation)

    def forward(self, x):
        return self.net(x)

class StateValueMLP(nn.Module):
    """State-value network V(s) over the global state and the flattened candidate (bin or field) states.

    Parameters
    ----------
    glob_dim : int
        Global state dimension D_glob.
    cand_dim : int
        Flattened candidate state dimension n_candidates * D_cand; 0 when there are none.
    hidden : tuple[int, ...]
        Hidden layer widths.
    layernorm : bool
        Whether to apply LayerNorm after each hidden layer.
    activation : type[nn.Module]
        Activation class.
    """

    def __init__(self, glob_dim: int, cand_dim: int, hidden: tuple[int, ...],
                 layernorm: bool = True, activation: type[nn.Module] = nn.ReLU):
        super().__init__()
        self.cand_dim = cand_dim
        self.net = build_mlp(glob_dim + cand_dim, hidden, 1, layernorm, activation)

    def forward(self, x_glob: torch.Tensor, x_cand: torch.Tensor) -> torch.Tensor:
        """Predict V(s).

        Parameters
        ----------
        x_glob : torch.Tensor
            Global state, shape (batch, D_glob).
        x_cand : torch.Tensor
            Candidate states, shape (batch, n_candidates, D_cand); ignored when cand_dim == 0.

        Returns
        -------
        torch.Tensor
            State value, shape (batch, 1).
        """
        if self.cand_dim > 0:
            x_glob = torch.cat([x_glob, x_cand.flatten(start_dim=1)], dim=1)  # [batch, D_glob + n_candidates * D_cand]
        return self.net(x_glob)

class ContextualScoreMLP(nn.Module):
    """
    Scores each candidate from [encoded global state, candidate features].

    Candidates are the rows of `x_cand`: HEALPix bins for bin action spaces, survey fields for field_filter.
    Rows are scored independently, so any number of candidates is accepted.
    """

    def __init__(self, global_dim: int, cand_feat_dim: int,
                 hidden_dim: int = 256,
                 nlayers: int = 2,
                 score_dim: int = 1,
                 global_enc_dim: int | None = 128,
                 use_contextual_gating: bool = False,
                 layernorm: bool = True,
                 activation: type[nn.Module] = nn.LeakyReLU):
        super().__init__()
        self.score_dim = score_dim
        self.use_contextual_gating = use_contextual_gating
        hidden = (hidden_dim,) * nlayers

        # --- global encoder ---------------------------------------------------
        if global_enc_dim is None:
            self.global_encoder = None
            g_dim = global_dim
        else:
            self.global_encoder = build_mlp(
                global_dim, (hidden_dim,), global_enc_dim,
                layernorm=layernorm, activation=activation,
            )
            g_dim = global_enc_dim

        # --- contextual gating (candidate feats, conditioned on g) ------------
        if use_contextual_gating:
            self.gate_net = nn.Sequential(
                nn.Linear(g_dim, cand_feat_dim),
                nn.Sigmoid(),
            )

        # --- shared per-candidate scorer -------------------------------------
        self.net = build_mlp(
            g_dim + cand_feat_dim, hidden, score_dim,
            layernorm=layernorm, activation=activation,
        )

    def forward(self, x_glob, x_cand):
        batch_size, n_cands, _ = x_cand.shape

        # encode global state once, then expand across candidates
        g = x_glob if self.global_encoder is None else self.global_encoder(x_glob)
        g = g.unsqueeze(1)                       # (batch, 1, g_dim)
        g_exp = g.expand(-1, n_cands, -1)        # (batch, n_cands, g_dim)

        if self.use_contextual_gating:
            gate_mask = self.gate_net(g_exp)
            x_cand = x_cand * gate_mask

        x = torch.cat((g_exp, x_cand), dim=-1)   # (batch, n_cands, g_dim + cand_feat_dim)
        scores = self.net(x)
        joint_action_scores = scores.view(batch_size, -1)
        return joint_action_scores


class DualStreamMLP(nn.Module):
    def __init__(self, global_dim, cand_feat_dim, hidden_dim, score_dim=1, activation=None,
                 use_layer_norm=True):
        super().__init__()
        self.activation = nn.LeakyReLU if activation is None else activation
        self.glob_enc = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            self.activation()
        )
        self.cand_enc = nn.Sequential(
            nn.Linear(cand_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            self.activation()
        )
        self.net = build_mlp(hidden_dim * 2, (hidden_dim,), score_dim,
                             layernorm=use_layer_norm, activation=self.activation)

    def forward(self, x_glob, x_cand):
        batch_size, n_cands, _ = x_cand.shape

        # Process independently
        g_emb = self.glob_enc(x_glob)
        g_emb = g_emb.unsqueeze(1).expand(-1, n_cands, -1) # Shape: (Batch, Candidates, Hidden)

        c_emb = self.cand_enc(x_cand) # Shape: (Batch, Candidates, Hidden)

        # Fuse deep in the network
        fused = torch.cat([g_emb, c_emb], dim=-1) # Shape: (Batch, Candidates, Hidden * 2)

        # Output scores
        scores = self.net(fused)
        return scores.view(batch_size, -1)




from torch.distributions import Categorical

class AutoregressiveNet(nn.Module):
    def __init__(self, glob_dim, bin_dim, action_dims, glob_hidden, bin_hidden,
                 nbins, nfilters, bin_out, state_latent_dim=256, activation=None, emb_dim=None, bin_first=False
                 ):
        """
        Args:
            state_dim (int): Dimension of the input state.
            action_dims (list of int): A list containing the number of discrete
                                       choices for each action dimension.
            hidden_dim (int): Number of hidden units for the state encoder.
            emb_dim (int): Embedding size for previously selected actions.
        """
        super().__init__()
        self.action_dims = action_dims
        self.num_actions = len(action_dims)
        self.activation = nn.ReLU if activation is None else activation
        self._filt_idx = int(bin_first)
        self._bin_idx = int(not bin_first)

        if emb_dim is None:
            emb_dim = state_latent_dim // 8 # about 11% of latent state size - tune later

        # State encoder
        self.state_encoder = FlatStateEncoder(glob_dim, bin_dim, nbins=nbins, glob_hidden=glob_hidden, bin_hidden=bin_hidden, bin_out=bin_out, output_dim=state_latent_dim, activation=self.activation)

        # 2. Embeddings for previously sampled actions (we don't need one for the final action)
        if bin_first:
            self.action_embeddings = nn.ModuleList([
                nn.Embedding(nbins, emb_dim)
            ])
        else: # Filter first
            self.action_embeddings = nn.ModuleList([
                nn.Embedding(nfilters, emb_dim)
            ])

        # Autoregressive action heads
        self.action_heads = nn.ModuleList()
        for i in range(self.num_actions):
            # Input to the i-th head is the state features + embeddings of all prior actions
            input_dim = state_latent_dim + (i * emb_dim)
            self.action_heads.append(nn.Linear(input_dim, action_dims[i]))


    def forward(self, x_glob, x_cand, action_mask, action=None):
        # GET LATENT SPACE REPRESENTATION
        x_latent = self.state_encoder(x_glob, x_cand)

        pred_actions = []
        log_probs = []
        entropies = []

        x_current = x_latent

        # For each head, COMPUTE LOGITS, PRED ACTIONS, and APPEND EMBEDDING for next head
        for i in range(self.num_actions):
            # 1. COMPUTE LOGITS
            logits = self.action_heads[i](x_current)

            if action_mask is not None:
                # Unflatten mask (ie, flat_idx = (bin * nfilters) + filt)
                batch_size = logits.size(0)
                nfilters = self.action_dims[self._filt_idx]
                nbins = self.action_dims[self._bin_idx]

                mask_2d = action_mask.view(batch_size, nbins, nfilters)

                # For first head, mask is just the original mask. For second head, must index into the mask based on the first head's sampled action
                if i == 0:
                    if self._filt_idx == 0:
                        step_mask = mask_2d.any(dim=1) # shape: (batch, nfilters)
                    else:
                        step_mask = mask_2d.any(dim=2) # shape: (batch, nbins)
                else:
                    first_choice = pred_actions[0]
                    batch_idx = torch.arange(batch_size, device=logits.device)

                    if self._bin_idx == 1:
                        step_mask = mask_2d[batch_idx, :, first_choice]
                    else:
                        step_mask = mask_2d[batch_idx, first_choice, :]

                # 2. PREDICT ACTIONS
                mask_value = torch.finfo(logits.dtype).min
                logits = logits.masked_fill(~step_mask, mask_value)

            dist = Categorical(logits=logits)

            if action is None: # ie, inference
                a_i = logits.argmax(dim=-1)
            else:
                a_i = action[:, i] # training

            pred_actions.append(a_i)
            log_probs.append(dist.log_prob(a_i))
            entropies.append(dist.entropy())

            # 3. APPEND EMBEDDING
            if i < self.num_actions - 1:
                emb = self.action_embeddings[i](a_i)
                x_current = torch.cat([x_current, emb], dim=-1)

        pred_actions = torch.stack(pred_actions, dim=1)

        # Joint log probability and entropy are the sum of the individual step calculations
        joint_log_prob = torch.stack(log_probs, dim=1).sum(dim=1)
        joint_entropy = torch.stack(entropies, dim=1).sum(dim=1)

        return pred_actions, joint_log_prob, joint_entropy


class CNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(num_features, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        # x shape: (Batch, Features, Lat, Lon)
        return self.cnn(x)
