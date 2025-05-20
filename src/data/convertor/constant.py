from data.ccdm.convertor import CCDMConvertor
from data.simulation.convertor import SIMConvertor
from data.et.convertor import ETConvertor
from data.ccdm.dataset import CCDMDataset
from data.et.dataset import ETDataset
from data.simulation.dataset import SimulationDataset

default_convertors = {
    "ccdm": CCDMConvertor(),
    "et": ETConvertor(),
    "simulation": SIMConvertor(),
}

default_normalizers = {
    "ccdm": CCDMDataset.normalize_item,
    "et": ETDataset.normalize_item,
    "simulation": SimulationDataset.normalize_item
}
