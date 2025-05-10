from data.ccdm.convertor import CCDMConvertor
from data.simulation.convertor import SIMConvertor
from data.et.convertor import ETConvertor

default_convertors = {
    "ccdm": CCDMConvertor(),
    "et": ETConvertor(),
    "simulation": SIMConvertor(),
}
