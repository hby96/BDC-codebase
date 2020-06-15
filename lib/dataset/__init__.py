from .loader import Base
from .loader_by_order import LoadByOrder
from .loader_by_feature import LoadByFeature
from .loader_by_our_feature import LoadByOurFeature


def build_loader(mode, cfg):
    loader = eval(cfg.DATASET.LOADER.TYPE)(mode, cfg)
    return loader
