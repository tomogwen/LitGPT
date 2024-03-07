# Tests for src/litgpt/data.py

import os

import pytest
import torch

from litgpt.data import TinyShakespeareDataModule, TinyShakespeareDataSet


# Tests for the DataSet
def test_tiny_shakespeare_dataset():
    # Simulate tokenised data
    tokenised_text = torch.arange(1000)
    block_size = 20
    dataset = TinyShakespeareDataSet(tokenised_text, block_size=block_size)

    # Check the length of the dataset
    assert len(dataset) == 980  # 1000 - block_size

    # Check the first item
    first_x, first_y = dataset[0]
    assert torch.equal(first_x, tokenised_text[:block_size])
    assert torch.equal(first_y, tokenised_text[1 : block_size + 1])

    # Check type
    assert isinstance(first_x, torch.Tensor)
    assert isinstance(first_y, torch.Tensor)


# Tests for the DataModule
@pytest.fixture
def sample_data_module():
    dummy_dataset_path = os.path.abspath("tests/sample_data/sample.txt")

    data_module = TinyShakespeareDataModule(
        dataset_path=dummy_dataset_path,
        batch_size=4,
        block_size=20,
        train_test_split=0.8,
    )
    return data_module


def test_data_module_setup(sample_data_module):
    # Run the setup
    sample_data_module.prepare_data()
    sample_data_module.setup(stage="fit")

    # Validate setup
    assert hasattr(sample_data_module, "train_data")
    assert hasattr(sample_data_module, "val_data")
    assert len(sample_data_module.train_data) > 0
    assert len(sample_data_module.val_data) > 0


def test_data_module_dataloaders(sample_data_module):
    sample_data_module.prepare_data()
    sample_data_module.setup(stage="fit")

    train_dataloader = sample_data_module.train_dataloader()
    val_dataloader = sample_data_module.val_dataloader()

    # Check if dataloaders are iterable
    assert iter(train_dataloader)
    assert iter(val_dataloader)

    # Check the batch
    for batch in train_dataloader:
        x, y = batch
        assert x.shape == (4, 20)  # Assuming batch size of 4 and block size of 20
        assert y.shape == (4, 20)
        break
