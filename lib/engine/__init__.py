from .trainer import Base
from .trainer_multi_time import MultiTime
from .trainer_res import Res
from .trainer_res_xgboost import ResXgboost
from .trainer_res_ensemble import ResEnsemble


def build_trainer(meta, cfg):
    trainer = eval(cfg.TRAIN.TRAINER.TYPE)(meta, cfg)
    return trainer

