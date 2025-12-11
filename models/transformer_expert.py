import torch
import torch.nn as nn
from einops import repeat
from models.model_utils import Attention, FeedForward, PreNorm


class Projection(nn.Module):
    """Optional input projection, shared or private."""
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, proj_dim, bias=True),
            nn.ReLU()
        )
    def forward(self, x):
        return self.fc(x)


class TransformerBlocks(nn.Module):
    """Stack of Transformer encoder blocks (Attention + FFN)."""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim * 2, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TransformerExpert(nn.Module):
    """One expert aggregator."""
    def __init__(self,
                 num_classes,
                 input_dim=2048,
                 dim=512,
                 depth=2,
                 heads=8,
                 mlp_dim=512,
                 dim_head=64,
                 dropout=0.1,
                 emb_dropout=0.1,
                 pool='cls',
                 pos_enc=None,
                 mode="separate",
                 shared_proj=None,
                 use_local_head=True):   
        super().__init__()

        # === projection ===
        if mode == "separate":
            self.projection = nn.Sequential(nn.Linear(input_dim, heads*dim_head, bias=True), nn.ReLU())
        elif mode == "shared":
            self.projection = shared_proj
        elif mode == "shared_adapter":
            # Standard adapter: shared projection + expert-specific bottleneck MLP
            bottleneck_dim = heads * dim_head // 4   
            self.projection = nn.Sequential(
                shared_proj,
                nn.Linear(heads * dim_head, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, heads * dim_head)
            )
        else:
            raise ValueError("Unknown mode")

        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = TransformerBlocks(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.norm = nn.LayerNorm(dim)
        self.pos_enc = pos_enc

        self.use_local_head = use_local_head
        if use_local_head:
            self.mlp_head = nn.Sequential(
                nn.Linear(dim, 64),  # Fixed: was mlp_dim, should be dim (latent dimension)
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        else:
            self.mlp_head = None  # logits will be None if no local head

    def forward(self, x, coords=None):
        b, _, _ = x.shape

        # Project features
        x = self.projection(x)
        if self.pos_enc is not None:
            x = x + self.pos_enc(coords)

        # Add CLS token if needed
        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        # Transformer stack
        x = self.dropout(x)
        x = self.transformer(x)

        # Pool
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # Latent + logits
        latent = self.norm(x)
        logits = self.mlp_head(latent) if self.use_local_head else None

        return latent, logits
