import torch
import torch.nn as nn
import numpy as np
from .nystrom_attention import NystromAttention
from .layers import create_mlp

class TransLayer(nn.Module):
    def __init__(self, norm_layer: nn.Module = nn.LayerNorm, dim: int = 512, num_heads: int = 8):
        """
        Transformer Layer with Nystrom Attention.

        Args:
            norm_layer (nn.Module): Normalization layer, default is nn.LayerNorm.
            dim (int): Dimension for the transformer layer, default is 512.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.norm = norm_layer(dim)
        self.attention = NystromAttention(
            dim=dim,
            dim_head=dim // num_heads,
            heads=num_heads,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the transformer layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention and normalization.
        """
        x = x + self.attention(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim: int = 512):
        """
        Position-wise Projection Embedded Gradient (PPEG) for positional encoding.

        Args:
            dim (int): Dimension for the embedding, default is 512.
        """
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Forward pass for the PPEG layer.

        Args:
            x (torch.Tensor): Input tensor.
            H (int): Height for reshaping.
            W (int): Width for reshaping.

        Returns:
            torch.Tensor: Output tensor with positional encoding applied.
        """
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMILExpert(nn.Module):
    def __init__(self,
                 in_dim: int,
                 embed_dim: int,
                 num_classes: int,
                 num_fc_layers: int = 1,
                 dropout: float = 0.25,
                 num_attention_layers: int = 2,
                 num_heads: int = 4,
                 mode: str = "separate",
                 shared_proj: nn.Module = None,
                 use_local_head: bool = True):
        """
        Expert variant of TransMIL with minimal changes for MoE compatibility:
        - adds `mode` in {"separate", "shared", "shared_adapter"}
        - supports `shared_proj` injection
        - optional `use_local_head`

        Everything else mirrors classical TransMIL.

        Args:
            in_dim (int): Input dimension for the MLP.
            embed_dim (int): Embedding dimension for all layers.
            num_classes (int): Number of output classes for classification.
            num_fc_layers (int): Number of fully connected layers in the MLP (default: 1).
            dropout (float): Dropout rate for MLP (default: 0.25).
            num_attention_layers (int): Number of transformer attention layers (default: 2).
            num_heads (int): Number of attention heads (default: 4).
            mode (str): Expert mode - "separate", "shared", or "shared_adapter".
            shared_proj (nn.Module): Shared projection module (for shared/shared_adapter modes).
            use_local_head (bool): Whether to use local classification head (default: True).
        """
        super().__init__()

        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_local_head = use_local_head
        self.mode = mode

        # ---- Projection/Patch Embedding ----
        if mode == "separate":
            if in_dim is None:
                raise ValueError("in_dim must be provided when mode='separate'.")
            self.patch_embed = create_mlp(
                in_dim=in_dim,
                hid_dims=[embed_dim] * (num_fc_layers - 1),
                dropout=dropout,
                out_dim=embed_dim,
                end_with_fc=False
            )
        elif mode == "shared":
            if shared_proj is None:
                raise ValueError("shared_proj must be provided for mode='shared'.")
            self.patch_embed = shared_proj  # expected to output embed_dim
        elif mode == "shared_adapter":
            if shared_proj is None:
                raise ValueError("shared_proj must be provided for mode='shared_adapter'.")
            # Shared base + expert-specific adapter
            self.patch_embed = nn.Sequential(
                shared_proj,
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ---- Positional Encoding (PPEG) ----
        self.pos_layer = PPEG(dim=embed_dim)

        # ---- CLS Token ----
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # ---- Transformer Blocks ----
        self.blocks = nn.ModuleList(
            [TransLayer(dim=embed_dim, num_heads=num_heads) for _ in range(num_attention_layers)]
        )

        # ---- Normalization ----
        self.norm = nn.LayerNorm(embed_dim)

        # ---- Classifier (conditional based on use_local_head) ----
        if use_local_head:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = None
        self.initialize_weights()

    def forward_features(self, h: torch.Tensor, return_attention: bool = False) -> tuple:
        """
        Get slide-level features from cls token.

        Args:
            h (torch.Tensor): The input tensor of shape (features, dim) or
                              (batch_size, features, dim).
            return_attention (bool): Whether to return attention scores.

        Returns:
            tuple: (wsi_feat, attention_dict)
                - wsi_feat: Slide-level feature of cls token
                - attention_dict: Dictionary containing attention scores if requested
        """
        # Handle both 2D and 3D inputs
        if len(h.shape) == 2:
            h = h.unsqueeze(0)  # [N, D] -> [1, N, D]
        elif len(h.shape) == 3:
            if h.size(0) != 1:
                raise ValueError(f"Expected batch size 1 bag, got {h.size(0)}")

        # Project to embedding dimension
        h = self.patch_embed(h)  # [B, N, embed_dim]

        # Square pad for positional encoding
        h, h_square, w_square = self._square_pad(h)

        # Add CLS token
        h = self._add_cls_token(h)  # [B, N+1, embed_dim]

        # Apply transformer layers with positional encoding
        h, attn_dict = self._apply_trans_layers(h, h_square, w_square, return_attention)

        # Get CLS token as slide-level feature
        wsi_feat = self.norm(h)[:, 0]  # [B, embed_dim]

        return wsi_feat, attn_dict
    
    def initialize_weights(self):
        """
        Initialize the weights of the model with kaiming he for linear layers, and xavier for all others
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _add_cls_token(self, h: torch.Tensor) -> torch.Tensor:
        """
        Add class token to the input tensor.

        Args:
            h (torch.Tensor): Input tensor [B, N, D].

        Returns:
            torch.Tensor: Input tensor with class token added [B, N+1, D].
        """
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        return h

    def _apply_trans_layers(self, h: torch.Tensor, h_square: int, w_square: int, return_attention: bool = False) -> tuple:
        """
        Apply transformer layers to the input.

        Args:
            h (torch.Tensor): Input tensor after adding class token.
            h_square (int): Square height obtained from padding calculation.
            w_square (int): Square width obtained from padding calculation.
            return_attention (bool): Whether to compute attention scores wrt cls token.

        Returns:
            tuple: (transformed_tensor, attention_dict)
        """
        intermed_dict = {}
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == 0:
                if return_attention:
                    # Compute attention scores wrt cls token in first position
                    cls_token = h[:, 0]  # [B, D]
                    feats = h[:, 1:]  # [B, N, D]
                    # Compute dot product similarity between each feat and cls token
                    intermed_dict['attention'] = torch.matmul(feats, cls_token.unsqueeze(-1)).squeeze(-1)
                # Apply positional encoding after first transformer block
                h = self.pos_layer(h, h_square, w_square)

        return h, intermed_dict

    def _square_pad(self, h: torch.Tensor) -> tuple:
        """
        Pad feature tensor to make it square.

        Args:
            h (torch.Tensor): Input tensor.

        Returns:
            tuple: Padded tensor, square height, and square width.
        """
        H = h.shape[1]
        add_length, h_square, w_square = self._get_square_length(H)
        h = torch.cat([h, h[:, :add_length, :]], dim=1)
        return h, h_square, w_square

    def _get_square_length(self, H: int) -> tuple:
        """
        Calculate the required lengths to convert the input into square form.

        Args:
            H (int): Original height (or length) of the input tensor.

        Returns:
            tuple: Additional length needed, and new square dimensions.
        """
        h_square, w_square = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = h_square * w_square - H
        return add_length, h_square, w_square

    def forward_head(self, wsi_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head (if enabled).

        Args:
            wsi_feat (torch.Tensor): Slide-level feature for classification.

        Returns:
            torch.Tensor: Logits for classification, or None if use_local_head=False.
        """
        if self.classifier is not None:
            logits = self.classifier(wsi_feat)
            return logits
        return None

    def forward(self,
                h: torch.Tensor,
                loss_fn: nn.Module = None,
                label: torch.LongTensor = None,
                attn_mask = None,
                return_attention: bool = False,
                return_slide_feats: bool = False) -> tuple:
        """
        Complete forward pass of the model.

        Args:
            h (torch.Tensor): Input feature tensor.
            loss_fn: Unused (kept for API compatibility).
            label: Unused (kept for API compatibility).
            attn_mask: Unused (kept for API compatibility).
            return_attention (bool): Whether to return attention scores.
            return_slide_feats: Unused (kept for API compatibility).

        Returns:
            tuple: (wsi_feats, logits)
                - wsi_feats: Slide-level features (latent representation)
                - logits: Classification logits (or None if use_local_head=False)
        """
        wsi_feats, _ = self.forward_features(h, return_attention=return_attention)
        logits = self.forward_head(wsi_feats) if self.use_local_head else None

        return wsi_feats, logits
