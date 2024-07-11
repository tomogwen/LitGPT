from typing import Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.utilities.data import AttributeDict
from torch import Tensor
from torch.nn import functional as F


class Head(nn.Module):
    """
    One head of self-attention.
    """

    def __init__(self, head_size: int, n_embd: int, dropout: float, block_size: int):
        """
        Args:
            head_size: output dimension of attention head.
            n_embd: dimension of embedding input into head.
            dropout: probability that each parameter will be zeroed out during training.
            block_size: number of tokens in each training sample.
        """
        super().__init__()
        self.key: nn.Linear = nn.Linear(n_embd, head_size, bias=False)
        self.query: nn.Linear = nn.Linear(n_embd, head_size, bias=False)
        self.value: nn.Linear = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )  # non-trainable param
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the attention head."""
        # keys, queries, values are simply different linear models on the same x
        B: int
        T: int
        C: int
        B, T, C = x.shape
        k: Tensor = self.key(x)
        q: Tensor = self.query(x)
        v: Tensor = self.value(x)

        # compute weights matrix wei, holding the attention scores (element-to-element affinities)
        wei: Tensor = q @ k.transpose(-2, -1) * C**-0.5
        wei: Tensor = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # making it a decoder block by masking 'forward' elements
        wei: Tensor = F.softmax(wei, dim=-1)
        wei: Tensor = self.dropout(wei)

        # perform the weighted aggregation of the values
        out: Tensor = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        n_embd: int,
        dropout: float,
        block_size: int,
    ):
        """
        Args:
            num_heads: number of attention heads in parallel.
            head_size: output dimension of attention head.
            n_embd: dimension of embedding input into head.
            dropout: probability that each parameter will be zeroed out during training.
            block_size: number of tokens in each training sample.
        """
        super().__init__()
        self.heads: nn.ModuleList = nn.ModuleList(
            [Head(head_size, n_embd, dropout, block_size) for _ in range(num_heads)]
        )
        self.projection: nn.Linear = nn.Linear(n_embd, n_embd)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the multiheaded attention."""
        x: Tensor = torch.cat([h(x) for h in self.heads], dim=-1)
        x: Tensor = self.dropout(self.projection(x))
        return x


class FeedForward(nn.Module):
    """
    Linear layer + non-linearity.
    """

    def __init__(self, n_embd: int, dropout: float):
        """
        Args:
            n_embd: dimension of embedding input into head.
            dropout: probability that each parameter will be zeroed out during training.
        """
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(
                4 * n_embd, n_embd
            ),  # projection layer? 'going back into the residual pathway'?
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the feed forward layer."""
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication then computation"""

    def __init__(self, n_embd: int, n_heads: int, dropout: float, block_size: int):
        """
        Args:
            n_embd: dimension of embedding input into head.
            n_heads: number of attention heads in parallel.
            dropout: probability that each parameter will be zeroed out during training.
            block_size: number of tokens in each training sample.
        """
        super().__init__()

        head_size: int = n_embd // n_heads

        self.sa: MultiHeadAttention = MultiHeadAttention(
            n_heads, head_size, n_embd, dropout, block_size
        )
        self.ffwd: FeedForward = FeedForward(n_embd, dropout)
        self.ln1: nn.LayerNorm = nn.LayerNorm(n_embd)
        self.ln2: nn.LayerNorm = nn.LayerNorm(n_embd)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the transformer block."""
        # layernorm added before transformations now (different to original paper)
        # called pre-norm formulation of a transformer
        x: Tensor = x + self.sa(self.ln1(x))  # communicate
        x: Tensor = x + self.ffwd(self.ln2(x))  # compute
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder PyTorch module.
    """

    def __init__(self, hparams: AttributeDict):
        """
        Args:
            hparams: hyperparameters AttributeDict passed from LitMinGPT.
        """
        super().__init__()
        self.hparams: AttributeDict = hparams

        self.token_embedding_table: nn.Embedding = nn.Embedding(
            self.hparams.vocab_size, self.hparams.n_embd
        )
        self.position_embedding_table: nn.Embedding = nn.Embedding(
            self.hparams.block_size, self.hparams.n_embd
        )

        self.blocks: nn.Sequential = nn.Sequential(
            *[
                Block(
                    self.hparams.n_embd,
                    self.hparams.n_heads,
                    self.hparams.dropout,
                    self.hparams.block_size,
                )
                for _ in range(self.hparams.num_blocks)
            ]
        )
        self.ln_f: nn.LayerNorm = nn.LayerNorm(self.hparams.n_embd)
        self.lm_head: nn.Linear = nn.Linear(
            self.hparams.n_embd, self.hparams.vocab_size
        )

    def forward(self, idx, targets=None) -> Tuple[Tensor, Optional[float]]:
        """Forward pass through the Transformer decoder."""
        B: int
        T: int
        B, T = idx.shape

        token_embeddings: Tensor = self.token_embedding_table(idx)
        T_arange: Tensor = torch.arange(T)
        T_arange: Tensor = T_arange.type_as(idx)
        position_embeddings: Tensor = self.position_embedding_table(T_arange)

        x: Tensor = token_embeddings + position_embeddings
        x: Tensor = self.blocks(x)
        x: Tensor = self.ln_f(x)
        logits: Tensor = self.lm_head(x)

        if targets is None:
            loss: Optional[float] = None
        else:
            B: int
            T: int
            C: int
            B, T, C = logits.shape
            logits: Tensor = logits.view(B * T, C)
            targets: Tensor = targets.view(B * T)
            loss: Optional[float] = F.cross_entropy(logits, targets)
            return logits, loss

        return logits, loss

    def generate(self, idx, max_new_tokens) -> Tensor:
        """Generates up to max_new_tokens from the prompt idx."""
        for _ in range(max_new_tokens):
            idx_cond: Tensor = idx[:, -self.hparams.block_size :]

            logits: Tensor
            logits, _ = self(idx_cond)  # logits are (B, T, C)
            logits: Tensor = logits[:, -1, :]  # logits becomes (B, C)
            probs: Tensor = F.softmax(logits, dim=-1)
            idx_next: Tensor = torch.multinomial(
                probs, num_samples=1
            )  # (B, T+1) - samples from across C for each batch
            idx: Tensor = torch.concat(
                (idx, idx_next), dim=1
            )  # becomes (B, T+1) -> adds on next loop
        return idx


class LitMinGPT(L.LightningModule):
    """
    LightningModule for an implementation of Kaparthy's MinGPT.
    """

    def __init__(
        self,
        vocab_size: int = 65,
        n_embd: int = 384,
        n_heads: int = 6,
        num_blocks: int = 3,
        batch_size: int = 64,
        block_size: int = 256,
        dropout: float = 0.2,
        lr: float = 3e-4,
    ):
        """
        Args:
            vocab_size: number of possible tokens.
            n_embd:  dimension of token embeddings.
            n_heads: number of self-attention heads.
            num_blocks: number of transformer blocks.
            batch_size: number of independent sequences processed in paralell.
            block_size: maximum context length for the transformer.
            dropout: probability that each parameter will be zeroed out during training.
            lr: learning rate.
        """
        super().__init__()
        self.save_hyperparameters()
        self.decoder: TransformerDecoder = TransformerDecoder(self.hparams)

    def forward(
        self, inputs: Tensor, target: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[float]]:
        """Forward pass through LightningModule MinGPT."""
        return self.decoder(inputs, target)

    def training_step(self, batch: Tensor, batch_idx: int) -> float:
        """Run one training step. Lightning built-in."""
        x: Tensor
        y: Tensor
        x, y = batch
        loss: float
        _, loss = self(x, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> float:
        """Run one validation step. Lightning built-in."""
        x: Tensor
        y: Tensor
        x, y = batch
        loss: float
        _, loss = self(x, y)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self) -> torch.optim.AdamW:
        """Configure optimisers. Lightning built-in."""
        optimizer: torch.optim.AdamW = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr
        )
        return optimizer
