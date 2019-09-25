from torch.utils.data.sampler import Sampler

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

# import gpustat
# import numpy as np
def bestGPU():
    # https://github.com/bamos/setGPU

    return 0