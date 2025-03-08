class CombinedDataLoader:
    def __init__(self, sim_loader, ccdm_loader):
        self.sim_loader = sim_loader
        self.ccdm_loader = ccdm_loader
        self.length = max(len(sim_loader), len(ccdm_loader))
    
    def __iter__(self):
        self.sim_iter = iter(self.sim_loader)
        self.ccdm_iter = iter(self.ccdm_loader)
        return self
    
    def __next__(self):
        try:
            sim_batch = next(self.sim_iter)
        except StopIteration:
            self.sim_iter = iter(self.sim_loader)
            sim_batch = next(self.sim_iter)
        
        try:
            ccdm_batch = next(self.ccdm_iter)
        except StopIteration:
            self.ccdm_iter = iter(self.ccdm_loader)
            ccdm_batch = next(self.ccdm_iter)
        
        return {'simulation': sim_batch, 'ccdm': ccdm_batch}
    
    def __len__(self):
        return self.length