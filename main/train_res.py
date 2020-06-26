import _init_paths
import sys

import argparse

from config import get_defaults_cfg, setup_cfg
from dataset import build_loader
from pre_process import build_pre_process
from engine import build_trainer

from utils import save_to_csv


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # init args
    cfg = get_defaults_cfg()
    cfg = setup_cfg(cfg, args.cfg, args.opts)

    # load data
    train_data = build_loader(mode='train', cfg=cfg).get_data()
    test_data = build_loader(mode='test', cfg=cfg).get_data()

    # pre process
    train = build_pre_process(data=train_data, mode='train', cfg=cfg).get_feature()
    test = build_pre_process(data=test_data, mode='test', cfg=cfg).get_feature()
    features = [c for c in train.columns if c not in ['loadingOrder', 'label', 'mmin', 'mmax', 'count', 'TRANSPORT_TRACE']]
    # features = [c for c in test.columns if c not in ['loadingOrder', 'label', 'mmin', 'mmax', 'count']]

    # training
    meta = {
        'train': train,
        'test': test,
        'pred': features,
        'label': 'label',
        # 'seed': 1080,
        'seed': 147+2080,
        'is_shuffle': True,
    }
    trainer = build_trainer(meta=meta, cfg=cfg)

    result = trainer.do_train()

    save_to_csv(result, test_data, cfg)
