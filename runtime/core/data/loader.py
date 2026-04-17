from aeon_fenix_delta.data.loader import get_loader, AEONDataset
import torch

def test_aeon_dataset_shape():
    dataset = AEONDataset(num_samples=10, input_dim=16)
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (16,)

def test_get_loader_batching():
    loader = get_loader(batch_size=4, input_dim=16, num_samples=20, shuffle=False)
    batch = next(iter(loader))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (4, 16)
