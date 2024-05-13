import os
import numpy as np
from BrainGraphStudio.utils import merge_nested_dicts
import logging
import json
import nni
from BrainGraphStudio.data import apply_transforms, convert_raw_to_datas, density_threshold
logger = logging.getLogger(__name__)

class ParamArgs():
    def __init__(self, path):
        
        self.path = path
        self.x_train_val = np.load(os.path.join(path, "x_train.npy"))
        self.y_train_val = np.load(os.path.join(path, "y_train.npy"))
        self.x_test = np.load(os.path.join(path, "x_test.npy"))
        self.y_test = np.load(os.path.join(path, "y_test.npy"))

        with open(os.path.join(path,"data.json"), "r") as file:
            self.file_args = json.load(file)
        for key, value in self.file_args.items():
            setattr(self, key, value)
        
        with open(os.path.join(path, "model.json"), "r") as file:
            self.model_args = json.load(file)

        self.use_brain_gnn = self.model_args["use_brain_gnn"]
        self.data_train_val, self.data_test = self.data_parser()
        self.update_data_features()
        self.get_model_name()

        if self.use_brain_gnn:
            self.check_brain_gnn_args()

        with open(os.path.join(path, "params.json"), "r") as file:
            self.param_args = json.load(file)
        self.process_param_args()

        self.nni_params = None
    
    def check_brain_gnn_args(self):
        if self.threshold != 10:
            logger.info(f"BrainGNN recommends threshold value of 10. Proceeding with top {self.threshold} of edges")
        
    def data_parser(self):
        thresh = self.threshold
        if thresh == 0:
            self.x_train_val = np.zeros_like(self.x)
            self.x_test = np.zeros_like(self.x_train_val)
        elif thresh < 100:
            self.x_train_val = density_threshold(self.x_train_val,thresh)
            self.x_test = density_threshold(self.x_test,thresh)
        data_list = convert_raw_to_datas(self.x_train_val, self.y_train_val)
        data_list_test = convert_raw_to_datas(self.x_test, self.y_test)
        if not self.use_brain_gnn:
            data_list = (apply_transforms(data_list, self.model_args["node_features"]))
            data_list_test = (apply_transforms(data_list_test, self.model_args["node_features"]))
        return data_list, data_list_test
    
    def update_data_features(self):
        self.num_features = self.data_train_val[0].x.shape[1]
        self.num_nodes = self.data_train_val[0].num_nodes

    def get_model_name(self):
        self.gcn_mp_type, self.gat_mp_type = None, None
        if self.use_brain_gnn:
            self.model_name = "brainGNN"
        elif self.model_args["message_passing"] != "":
            self.model_name = "gcn"
            self.gcn_mp_type = self.model_args["message_passing"]
        elif self.model_args["message_passing_w_attn"] != "":
            self.model_name = "gat"
            self.gat_mp_type = self.model_args["message_passing_w_attn"]
        
        if not self.use_brain_gnn:
            self.pooling = self.model_args["pooling"]
    def process_param_args(self):
        self.use_nni = self.param_args["nni"]["optimization_algorithm"] != "None"
        self.param_args = merge_nested_dicts(self.param_args)
        for key, value in self.param_args.items():
            setattr(self, key, value)
    
    def add_nni_args(self,nni_parameters):
        if self.use_nni:
            logger.info("Logging NNI args")
            if not isinstance(nni.typehint.Parameters):
                raise ValueError()
            else:
                self.nni_params = nni_parameters
                for key in nni_parameters:
                    value = nni_parameters[key]
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    setattr(self, key, value)

