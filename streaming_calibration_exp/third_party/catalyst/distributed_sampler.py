"""DistributedSamplerWrapper.
Adopted verbatim from catalyst v21.5 (https://github.com/catalyst-team/catalyst/blob/v21.05/catalyst/data/sampler.py)
Licensed under Apache-2.0 License.
"""
from operator import itemgetter

from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler


class _DatasetFromSampler(Dataset):
    """Wrap a Sampler so it can be consumed as a Dataset by DistributedSampler."""

    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """Wrap a non-distributed Sampler so it works under DDP."""

    def __init__(self, sampler, num_replicas=None, rank=None, shuffle: bool = True):
        super().__init__(
            _DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        self.dataset = _DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        return iter(itemgetter(*indexes_of_indexes)(self.dataset))
