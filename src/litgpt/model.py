# Minimal GPT model, includes:
#  - Torch implementation of Kaparthy's minGPT
#  - Lightning wrapper for the model

import lightning as L
import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )  # non-trainable param
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # keys, queries, values are simply different linear models on the same x
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # compute weights matrix wei, holding the attention scores (element-to-element affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # making it a decoder block by masking 'forward' elements
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, dropout, block_size) for _ in range(num_heads)]
        )
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.dropout(self.projection(x))
        return x


class FeedForward(nn.Module):
    """linear layer + non-linearity"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(
                4 * n_embd, n_embd
            ),  # projection layer? 'going back into the residual pathway'?
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication then computation"""

    def __init__(self, n_embd, n_heads, dropout, block_size):
        super().__init__()

        head_size = n_embd // n_heads

        self.sa = MultiHeadAttention(n_heads, head_size, n_embd, dropout, block_size)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # layernorm added before transformations now (different to original paper)
        # called pre-norm formulation of a transformer
        x = x + self.sa(self.ln1(x))  # communicate
        x = x + self.ffwd(self.ln2(x))  # compute
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.token_embedding_table = nn.Embedding(
            self.hparams.vocab_size, self.hparams.n_embd
        )
        self.position_embedding_table = nn.Embedding(
            self.hparams.block_size, self.hparams.n_embd
        )

        self.blocks = nn.Sequential(
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
        self.ln_f = nn.LayerNorm(self.hparams.n_embd)
        self.lm_head = nn.Linear(self.hparams.n_embd, self.hparams.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)
        T_arange = torch.arange(T)
        T_arange = T_arange.type_as(idx)
        position_embeddings = self.position_embedding_table(T_arange)

        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.hparams.block_size :]

            logits, _ = self(idx_cond)  # logits are (B, T, C)
            logits = logits[:, -1, :]  # logits becomes (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(
                probs, num_samples=1
            )  # (B, T+1) - samples from across C for each batch
            idx = torch.concat(
                (idx, idx_next), dim=1
            )  # becomes (B, T+1) -> adds on next loop
        return idx


class LitMinGPT(L.LightningModule):
    def __init__(
        self,
        vocab_size: int = 65,
        n_embd: int = 384,  # dimension of token embeddings
        n_heads: int = 6,  # number of self-attention heads
        num_blocks: int = 3,  # number of transformer blocks
        batch_size: int = 64,  # how many independent sequences processed in paralell?
        block_size: int = 256,  # maximum context length for the transformer (max T)
        dropout: float = 0.2,  # propo of dropout
        lr: float = 3e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.decoder = TransformerDecoder(self.hparams)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.decoder(x, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.decoder(x, y)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
