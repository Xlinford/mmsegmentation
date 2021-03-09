from itertools import cycle
from types import MethodType

class ZipDataset(object):
    def __init__(self, data_loaders):
        self.iter_loader = [iter(zip(cycle(data_loaders[0]), data_loaders[1]))]
        self.epoch_max_iters = len(data_loaders[1])

    def __len__(self):
        return self.epoch_max_iters

    def length(self):
        return self.epoch_max_iters

    def zip_dataset_length(self, dataset):
        dataset.__len__ = MethodType(self.length, dataset)
        return dataset


