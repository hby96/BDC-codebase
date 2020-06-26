from .loader import Base
from .loader_by_order import LoadByOrder
from .loader_by_feature import LoadByFeature
from .loader_by_our_feature import LoadByOurFeature
from .loader_by_gps_feature import LoadByGpsFeature
from .load_by_multi_time_feature import LoadByMultiTimeFeature
from .load_by_res_feature import LoadByResFeature


def build_loader(mode, cfg):
    loader = eval(cfg.DATASET.LOADER.TYPE)(mode, cfg)
    return loader
