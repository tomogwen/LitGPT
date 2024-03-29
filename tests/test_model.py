# Tests for src/litgpt/model.py

from dataclasses import dataclass

import pytest
import torch

from litgpt.model import LitMinGPT, TransformerDecoder


@dataclass
class TestHparams:
    vocab_size: int = 65
    n_embd: int = 384
    n_heads: int = 6
    num_blocks: int = 3
    batch_size: int = 2  # smaller size for testing
    block_size: int = 256
    dropout: float = 0.2
    lr: float = 3e-4


@pytest.fixture
def sample_input():
    hparams = TestHparams()
    return torch.randint(
        low=0, high=hparams.vocab_size, size=(hparams.batch_size, hparams.block_size)
    )


@pytest.fixture
def transformer_decoder():
    hparams = TestHparams()
    return TransformerDecoder(hparams)


@pytest.fixture
def lit_mingpt(transformer_decoder):
    return LitMinGPT()


# Tests for TransformerDecoder
def test_transformer_decoder_forward(transformer_decoder, sample_input):
    # Test the forward pass
    hparams = TestHparams()
    logits, loss = transformer_decoder(sample_input)
    assert logits is not None, "Logits should not be None"
    assert logits.shape == (
        hparams.batch_size,
        hparams.block_size,
        hparams.vocab_size,
    ), f"Logits should have shape {(hparams.batch_size, hparams.block_size, hparams.vocab_size)}"


# Tests for LitMinGPT
def test_lit_mingpt_training_step(lit_mingpt, sample_input):
    # Test the training step
    y = sample_input  # Dummy target for testing
    batch = (sample_input, y)
    loss = lit_mingpt.training_step(batch, batch_idx=0)
    assert loss is not None, "Training loss should not be None"


def test_lit_mingpt_validation_step(lit_mingpt, sample_input):
    # Test the validation step
    y = sample_input  # Dummy target for testing
    batch = (sample_input, y)
    loss = lit_mingpt.validation_step(batch, batch_idx=0)
    assert loss is not None, "Validation loss should not be None"
