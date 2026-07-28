import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

class ImprovedResNet(nn.Module):
    """
    Pre-norm residual MLP for surrogate regression.
    Designed for stable training and accurate edge behavior.
    """

    def __init__(
        self,
        input_dim: int,
        features: int = 512,
        depth: int = 2,
        num_outputs: int = 1,
        dropout: float = 0.05,
        activation: str = 'silu',  # 'silu' or 'relu'
    ):
        super().__init__()

        # -----------------------------
        # Input projection
        # -----------------------------
        self.first = nn.Linear(input_dim, features)

        # -----------------------------
        # Residual blocks (PRE-NORM)
        # -----------------------------
        self.residuals = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(depth)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(features) for _ in range(depth)]
        )

        # -----------------------------
        # Activation & regularization
        # -----------------------------
        if activation.lower() == 'silu':
            self.activation = nn.SiLU()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout(dropout)

        # -----------------------------
        # Output head (linear)
        # -----------------------------
        self.last = nn.Linear(features, num_outputs)

        # Stable regression initialization
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, x):
        # Input projection
        out = self.first(x)

        # Residual stack
        for lin, norm in zip(self.residuals, self.norms):
            h = norm(out)
            h = lin(h)
            h = self.activation(h)
            h = self.dropout(h)
            out = out + h

        # Linear output (no sigmoid / clipping)
        return self.last(out)
