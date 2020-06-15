from .trainer import Base


def build_trainer(meta, cfg):
    trainer = eval(cfg.TRAIN.TRAINER.TYPE)(meta)
    return trainer

