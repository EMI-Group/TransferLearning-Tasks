import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from . import build_dataset


def make_data_loader(cfg, is_train=True, distributed=False, max_iter=None, start_iter=0):
    shuffle = is_train or distributed
    data_loaders = []
    dataset = build_dataset()
    for dataset in datasets:
        if distributed:
            sampler = samplers.DistributedSampler(dataset, shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)

        batch_size = cfg.SOLVER.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
        if max_iter is not None:
            batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=max_iter, start_iter=start_iter)

        data_loader = DataLoader(dataset, num_workers=cfg.DATA_LOADER.NUM_WORKERS, batch_sampler=batch_sampler, pin_memory=cfg.DATA_LOADER.PIN_MEMORY, collate_fn=BatchCollator(is_train))
        data_loaders.append(data_loader)

    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
