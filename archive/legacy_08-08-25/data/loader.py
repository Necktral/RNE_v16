from .data_loader import AEONDataset
import torch

def test_aeon_dataset_shape():
    dataset = AEONDataset(num_samples=10, input_dim=16)
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (16,)

def get_loader(batch_size=4, input_dim=16, num_samples=20, shuffle=False):
    dataset = AEONDataset(num_samples=num_samples, input_dim=input_dim)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def test_get_loader_batching():
    loader = get_loader(batch_size=4, input_dim=16, num_samples=20, shuffle=False)
    batch = next(iter(loader))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (4, 16)
