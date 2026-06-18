import torch
import torch.nn as nn

from .transformer import RenormTransformerLayer


class RenormLM(nn.Module):
    """
    An Autoregressive Language Model constructed entirely from
    self-stabilizing Renorm-Native components.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        depth: int,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Stack our upgraded causal transformer blocks
        self.blocks = nn.ModuleList(
            [RenormTransformerLayer(d_model=d_model, n_heads=n_heads) for _ in range(depth)]
        )

        # Language modeling head projecting hidden states back to vocabulary log-probabilities
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights between embedding and projection head for training efficiency
        self.token_emb.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        device = idx.device

        # Combine word tokens and positional indices
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)

        # Sequentially pass tensors through the causal blocks
        for block in self.blocks:
            x = block(x, causal=True)

        return self.lm_head(x)
