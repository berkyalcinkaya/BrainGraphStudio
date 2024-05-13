import random
import os
import torch
import numpy as np
import logging
from sklearn.model_selection import train_test_split
logger = logging.getLogger(__name__)

TRAIN_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def seed_everything(seed):
    logging.info(f"seed for seed_everything(): {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # set random seed for numpy
    torch.manual_seed(seed)  # set random seed for CPU
    torch.cuda.manual_seed_all(seed)  # set random seed for all GPUs

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU {device} found")
    else:
        device = torch.device("cpu")
        logger.info("Using with CPU")
    return device


