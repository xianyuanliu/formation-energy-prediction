import os
from datetime import datetime
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.BASE_DATA_DIR = "data"
_C.DATASET.DATA_PATH = "data/split_both_hhi"
_C.DATASET.CIF_PATH = "data/cifs"
_C.DATASET.TASK = "regression"
_C.DATASET.XRD = True
_C.DATASET.TEXT = True
_C.DATASET.TRAIN_FILE = "train.csv"
_C.DATASET.TEST_FILE = "test.csv"
_C.DATASET.VAL_RATIO = 0.1
_C.DATASET.TEST_RATIO = 0.1

# -----------------------------------------------------------------------------
# Solver (Optimizer, Learning Rate, etc.)
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.OPTIM = "SGD"
_C.SOLVER.LR = 0.01
_C.SOLVER.LR_MILESTONES = [100]
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.BATCH_SIZE = 256
_C.SOLVER.EPOCHS = 30
_C.SOLVER.PRINT_FREQ = 10
_C.SOLVER.WORKERS = 0

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.GRAPH_TYPE = "cgcnn"
_C.MODEL.ATOM_FEA_LEN = 64
_C.MODEL.H_FEA_LEN = 128
_C.MODEL.N_CONV = 3
_C.MODEL.N_H = 1

# -----------------------------------------------------------------------------
# Misc (WandB, Results)
# -----------------------------------------------------------------------------
_C.MISC = CN()
_C.MISC.USE_WANDB = True
_C.MISC.WANDB_PROJECT = "formatin-energy-prediction"
_C.MISC.WANDB_GROUP = datetime.now().strftime("%m%d")
_C.MISC.WANDB_NAME = ""
_C.MISC.RESULT_DIR = "outputs"
_C.MISC.RESULT_FILES = ["checkpoint.pth.tar", "model_best.pth.tar", "test_results.csv", "test.n*"]
_C.MISC.DISABLE_CUDA = False
_C.MISC.RESUME = ""

def finalize_config(cfg):
    """Post-process config to generate dynamic fields like WANDB_NAME and RESULT_DIR."""
    # 1. Auto-generate WANDB_NAME if empty
    if not cfg.MISC.WANDB_NAME:
        gt = cfg.MODEL.GRAPH_TYPE
        dp = os.path.basename(cfg.DATASET.DATA_PATH)
        tf = "default"
        if cfg.DATASET.TEST_FILE:
            tf_path = cfg.DATASET.TEST_FILE[0] if isinstance(cfg.DATASET.TEST_FILE, list) else cfg.DATASET.TEST_FILE
            tf = os.path.splitext(os.path.basename(tf_path))[0]
        cfg.MISC.WANDB_NAME = f"{gt}_{dp}_{tf}"
    
    # 2. Apply hierarchical path: outputs/{group}/{name}
    if cfg.MISC.RESULT_DIR == "outputs":
        cfg.MISC.RESULT_DIR = os.path.join("outputs", cfg.MISC.WANDB_GROUP, cfg.MISC.WANDB_NAME)
        
    return cfg

def load_and_apply_config(args):
    """Dynamically merge YAML into args. Avoids manual mapping lists."""
    cfg = get_cfg_defaults()
    if args.config:
        cfg.merge_from_file(args.config)
    cfg = finalize_config(cfg)
    
    # Automatically map all nested cfg values to args namespace
    # This avoids redundant mapping lists and keeps code clean.
    for section_name in cfg.keys():
        section = cfg[section_name]
        if isinstance(section, CN):
            for key, value in section.items():
                # Map to lowercase args (e.g., DATA_PATH -> args.data_path)
                attr_name = key.lower()
                if hasattr(args, attr_name):
                    setattr(args, attr_name, value)

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    return _C.clone()
