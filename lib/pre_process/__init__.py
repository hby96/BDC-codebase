from .base import Base
from .our_base import OurBase
from .each_point import EachPoint
from .identity import Identity


def build_pre_process(data, mode, cfg):
    pre_process = eval(cfg.PRE_PROCESS.TYPE)(data, mode)
    return pre_process
