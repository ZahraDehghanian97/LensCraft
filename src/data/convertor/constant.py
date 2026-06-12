from data.ccdm.convertor import CCDMConvertor
from data.simulation.convertor import SIMConvertor
from data.et.convertor import ETConvertor
from data.gendop.convertor import GenDoPConvertor
from data.ccdm.dataset import CCDMDataset
from data.et.dataset import ETDataset
from data.simulation.dataset import SimulationDataset
from data.gendop.dataset import GenDoPDataset

default_convertors = {
    "ccdm": CCDMConvertor(),
    "et": ETConvertor(),
    "simulation": SIMConvertor(),
    "gendop": GenDoPConvertor(),
}

default_normalizers = {
    "ccdm": CCDMDataset.normalize_item,
    "et": ETDataset.normalize_item,
    "simulation": SimulationDataset.normalize_item,
    "gendop": GenDoPDataset.normalize_item,
}
