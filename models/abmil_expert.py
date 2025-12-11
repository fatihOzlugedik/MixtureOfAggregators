import torch
import torch.nn as nn
import torch.nn.functional as F

from models.aggregator import BaseAggregator

class ABMILExpert(nn.Module):
    def __init__(self,
                 input_dim=None,
                 size_arg="big",
                 n_heads=1,
                 dropout=0.1,
                 num_classes=4,
                 mode="separate",
                 shared_proj=None,
                 use_local_head=True):
        r"""
        Expert variant of classical ABMIL with minimal changes:
        - adds `mode` in {"separate","shared","shared_adapter"}
        - supports `shared_proj` injection
        - optional `use_local_head`
        Everything else mirrors ABMIL.
        """
        # Properly initialize nn.Module
        super().__init__()
        self.size_dict = {"small": [input_dim, 256, 64], "big": [512, 256, 64]}

        self.input_dim = input_dim
        self.use_local_head = use_local_head
        self.mode = mode

        # ---- Projection (mirrors classical: 512-d) ----
        if mode == "separate":
            if input_dim is None:
                raise ValueError("input_dim must be provided when mode='separate'.")
            self.projection = nn.Sequential(nn.Linear(self.input_dim, 512, bias=True), nn.ReLU())
        elif mode == "shared":
            if shared_proj is None:
                raise ValueError("shared_proj must be provided for mode='shared'.")
            self.projection = shared_proj  # expected to output 512
        elif mode == "shared_adapter":
            if shared_proj is None:
                raise ValueError("shared_proj must be provided for mode='shared_adapter'.")
            self.projection = nn.Sequential(
                shared_proj,
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 512),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        size = self.size_dict[size_arg]

        # ---- Attention (single-head, identical to classical) ----
        fc = []
        attention_net = Attn_Net_Gated(L=size[0], D=size[1], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        # self.rho = nn.Sequential(*[nn.Linear(size[0], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        self.rho = nn.Sequential(*[nn.Linear(size[0], size[2]), nn.ReLU()])

        # ---- Classifier (same sizing; n_heads kept for API compatibility) ----
        self.classifier = nn.Sequential(nn.Linear(size[2]*n_heads, num_classes))

        self.activation = nn.ReLU()

    def forward(self, x, *args):
        x_path = self.projection(x)              # [B?, N, C] -> [B?, N, 512] or [N, 512]
        # be robust to [1, N, C] and [N, C] without blindly removing dims
        if x_path.dim() == 3:
            if x_path.size(0) != 1:
                raise ValueError(f"Expected batch size 1 bag, got {x_path.size(0)}")
            x_path = x_path.squeeze(0)          # [N, 512]

        A, h_path = self.attention_net(x_path)  # A: [N,1], h_path: [N,512] or [N,L]
        A = torch.transpose(A, 1, 0)            # [1, N]
        A = F.softmax(A, dim=1)
        h_path = torch.mm(A, h_path)            # [1, L]
        h_path = self.rho(h_path)    # [rho_dim]

        latent = h_path
        logits = self.classifier(latent) if self.use_local_head else None  # [1, num_classes] or None
        return latent, logits


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)
        (kept identical to classical)
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a.mul(b))  # [N, n_classes]
        return A, x
