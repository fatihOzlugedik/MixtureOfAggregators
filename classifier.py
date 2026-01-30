import torch.nn as nn
import models as models

class MIL_aggregtor(nn.Module):
    """
    MoA Classifier with simplified architecture:
    - Always uses separate experts (no shared/adapter modes)
    - Always uses single shared classification head (no local heads per expert)
    - Expert diversity initialization (always on)
    - Support for router warmup
    - Enhanced routing controls
    - Optional Straight-Through estimator

    Architecture: Separate experts → latents → router weighting → shared head
    """
    def __init__(self, class_count, arch, expert_arch, embedding_dim=768,
                 router_style="topk", topk=1, save_gates=False, num_expert=1,
                 router_type="linear", base_seed=None, use_st=False):
        super(MIL_aggregtor, self).__init__()

        # Determine which architecture to use
        if use_st and arch == "MixtureOfAggregators":
            # Use ST variant for MoA
            arch_to_use = "MixtureOfAggregators_ST"
        else:
            arch_to_use = arch

        if arch_to_use not in models.__dict__:
            raise ValueError(f"Unknown model architecture '{arch_to_use}'")

        # Simplified architecture with all improvements enabled
        # - Always separate experts (diversity via different seeds)
        # - Always single shared head (no local heads per expert)
        self.model = models.__dict__[arch_to_use](
            input_dim=embedding_dim,
            num_classes=class_count,
            expert_arch=expert_arch,
            router_style=router_style,
            k_active=topk,
            num_experts=num_expert,
            router_type=router_type,
            diversity_init=True,                # Always enabled
            base_seed=base_seed
        )
        self.save_gates = save_gates
        self.use_st = use_st

    def forward(self, embeddings, return_latent=False, return_gates=False, temp=None,
                use_gumbel=None, k=None, epoch=None, force_uniform_epochs=None,
                warmup_epochs=None):
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0).unsqueeze(0)
        elif embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        elif embeddings.dim() != 3:
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

        # Pass optional routing controls if supported by the underlying model
        kwargs = {}
        if temp is not None:
            kwargs['temp'] = temp
        if use_gumbel is not None:
            kwargs['use_gumbel'] = use_gumbel
        if k is not None:
            kwargs['k'] = k
        if epoch is not None:
            kwargs['epoch'] = epoch
        if force_uniform_epochs is not None:
            kwargs['force_uniform_epochs'] = force_uniform_epochs
        if warmup_epochs is not None:
            kwargs['warmup_epochs'] = warmup_epochs

        # No try-except: let it fail if model doesn't support the architecture
        out = self.model(embeddings, **kwargs)
        if len(out) == 3:
            latent, logits, gates = out
        else:
            latent, logits = out
            gates = None

        if return_latent and return_gates:
            return latent, logits, gates
        elif return_latent:
            return latent, logits
        elif return_gates:
            return logits, gates
        return logits

    def __repr__(self):
        return f"MIL_aggregtor(model={self.model})"
