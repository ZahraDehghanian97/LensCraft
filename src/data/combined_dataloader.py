class CombinedDataLoader:
    """Yield paired batches for one epoch, restarting the shorter loader.

    Both loaders must have batches to form a pair. If either loader is empty,
    the combined loader is empty as well.
    """

    def __init__(self, sim_loader, ccdm_loader):
        self.sim_loader = sim_loader
        self.ccdm_loader = ccdm_loader
        lengths = (len(sim_loader), len(ccdm_loader))
        self.length = max(lengths) if all(lengths) else 0

    def __iter__(self):
        if self.length == 0:
            return
        loaders = {"simulation": self.sim_loader, "ccdm": self.ccdm_loader}
        iterators = {key: iter(loader) for key, loader in loaders.items()}
        for _ in range(self.length):
            batch = {}
            for key, loader in loaders.items():
                try:
                    batch[key] = next(iterators[key])
                except StopIteration:
                    iterators[key] = iter(loader)
                    batch[key] = next(iterators[key])
            yield batch

    def __len__(self):
        return self.length
