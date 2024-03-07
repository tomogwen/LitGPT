# Tests for src/litgpt/model.py

from dataclasses import dataclass

import pytest
import torch

from litgpt.model import LitMinGPT, TransformerDecoder


@dataclass
class TestHParams:
    VOCAB_SIZE: int = 65
    N_EMBD: int = 384
    N_HEADS: int = 6
    NUM_BLOCKS: int = 3
    BATCH_SIZE: int = 2  # smaller size for testing
    BLOCK_SIZE: int = 256
    DROPOUT: float = 0.2
    lr: float = 3e-4


@pytest.fixture
def sample_input():
    hparams = TestHParams()
    return torch.randint(
        low=0, high=hparams.VOCAB_SIZE, size=(hparams.BATCH_SIZE, hparams.BLOCK_SIZE)
    )


@pytest.fixture
def transformer_decoder():
    hparams = TestHParams()
    return TransformerDecoder(hparams)


@pytest.fixture
def lit_mingpt(transformer_decoder):
    return LitMinGPT()


# Tests for TransformerDecoder
def test_transformer_decoder_forward(transformer_decoder, sample_input):
    # Test the forward pass
    hparams = TestHParams()
    logits, loss = transformer_decoder(sample_input)
    assert logits is not None, "Logits should not be None"
    assert logits.shape == (
        hparams.BATCH_SIZE,
        hparams.BLOCK_SIZE,
        hparams.VOCAB_SIZE,
    ), f"Logits should have shape {(hparams.BATCH_SIZE, hparams.BLOCK_SIZE, hparams.VOCAB_SIZE)}"


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
