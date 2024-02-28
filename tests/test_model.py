# Tests for src/litgpt/model.py

import pytest
import torch

from litgpt.model import LitMinGPT, TransformerDecoder

# Constants based on your model's hyperparameters
VOCAB_SIZE = 65
BATCH_SIZE = 2  # Smaller size for testing
BLOCK_SIZE = 256


@pytest.fixture
def sample_input():
    return torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, BLOCK_SIZE))


@pytest.fixture
def transformer_decoder():
    return TransformerDecoder()


@pytest.fixture
def lit_mingpt(transformer_decoder):
    return LitMinGPT(transformer_decoder=transformer_decoder)


# trigger tests
print("hi")


# Tests for TransformerDecoder
def test_transformer_decoder_forward(transformer_decoder, sample_input):
    # Test the forward pass
    logits, loss = transformer_decoder(sample_input)
    assert logits is not None, "Logits should not be None"
    assert logits.shape == (
        BATCH_SIZE,
        BLOCK_SIZE,
        VOCAB_SIZE,
    ), f"Logits should have shape {(BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE)}"


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
