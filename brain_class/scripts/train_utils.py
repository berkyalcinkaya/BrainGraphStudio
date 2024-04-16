import random
import os
import sys
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
import logging
from skimage.io import imread
import json
from brain_class.data import apply_transforms, convert_raw_to_datas, density_threshold
logger = logging.getLogger(__name__)


class Data():
    def __init__(self, path):
        self.path = path
        self.x = imread( os.path.join(path, "x.npy"))
        self.y = imread( os.path.join(path, "y.npy"))

        self.file_args = json.load(os.path.join(path,"data.json"))
        for key, value in self.file_args.items():
            setattr(self, key, value)
        
        self.model_args = json.load(os.path.join(path, "model.json"))
        self.use_brain_gnn = self.model_args["use_brain_gnn"]
        if self.use_brain_gnn:
            self.check_brain_gnn_args()
        self.data = self.data_parser()
    
    def check_brain_gnn_args(self):
        if self.threshold != 10:
            print(f"BrainGNN recommends threshold value of 10. Proceeding with top {self.threshold} of edges")
        
    def data_parser(self):
        thresh = self.threshold
        if thresh == 0:
            self.x = np.zeros_like(self.x)
        elif thresh < 100:
            self.x = density_threshold(self.x,thresh)
        data_list = convert_raw_to_datas(self.x, self.y)
        if self.use_brain_gnn:
            data_list = (apply_transforms(data_list, self.model_args["node_features"]))
        return np.array(data_list)

def seed_everything(seed):
    logging.info(f"seed for seed_everything(): {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # set random seed for numpy
    torch.manual_seed(seed)  # set random seed for CPU
    torch.cuda.manual_seed_all(seed)  # set random seed for all GPUs


