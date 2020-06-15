from yacs.config import CfgNode


def get_defaults_cfg():
    """
    Construct the default configuration tree.

    Returns:
        cfg (CfgNode): the default configuration tree.
    """
    cfg = CfgNode()

    cfg.CSV_SAVE_PATH = ""

    cfg["DATASET"] = CfgNode()
    cfg.DATASET.TRAIN_GPS_PATH = ""
    cfg.DATASET.VALID_GPS_PATH = ""
    cfg.DATASET.TEST_DATA_PATH = ""
    cfg.DATASET.LOADER = CfgNode()
    cfg.DATASET.LOADER.TYPE = "Base"
    cfg.DATASET.LOADER.NROWS = 0

    cfg["PRE_PROCESS"] = CfgNode()
    cfg.PRE_PROCESS.TYPE = "Base"

    cfg["MODEL"] = CfgNode()
    cfg.MODEL.TYPE = "Base"

    cfg["TRAIN"] = CfgNode()
    cfg.TRAIN.TRAINER = CfgNode()
    cfg.TRAIN.TRAINER.TYPE = "Base"
    cfg.TRAIN.TESTER = CfgNode()
    cfg.TRAIN.TESTER.TYPE = "Base"

    cfg["EVALUATE"] = CfgNode()
    cfg.EVALUATE.TYPE = 'Base'

    return cfg


def setup_cfg(cfg, cfg_file, cfg_opts):
    """
    Load a yaml config file and merge it this CfgNode.

    Args:
        cfg (CfgNode): the configuration tree with default structure.
        cfg_file (str): the path for yaml config file which is matched with the CfgNode.
        cfg_opts (list, optional): config (keys, values) in a list (e.g., from command line) into this CfgNode.

    Returns:
        cfg (CfgNode): the configuration tree with settings in the config file.
    """
    cfg.merge_from_file(cfg_file)
    cfg.merge_from_list(cfg_opts)
    cfg.freeze()

    return cfg