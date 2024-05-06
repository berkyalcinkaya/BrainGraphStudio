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
        self.x = np.load(os.path.join(path, "x.npy"))
        self.y = np.load(os.path.join(path, "y.npy"))

        self.file_args = json.load(os.path.join(path,"data.json"))
        for key, value in self.file_args.items():
            setattr(self, key, value)
        
        self.model_args = json.load(os.path.join(path, "model.json"))
        self.use_brain_gnn = self.model_args["use_brain_gnn"]
        self.data = self.data_parser()
        self.update_data_features()
        self.get_model_name()

        if self.use_brain_gnn:
            self.check_brain_gnn_args()
    
        self.param_args = json.load(os.path.join(path, "params.json"))
        self.process_param_args()
    
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
        if not self.use_brain_gnn:
            data_list = (apply_transforms(data_list, self.model_args["node_features"]))
        return np.array(data_list)
    
    def update_data_features(self):
        self.num_features = self.data[0].x.shape[1]
        self.num_nodes = self.data[0].num_nodes

    def get_model_name(self):
        self.gcn_mp_type, self.gat_mp_type = None, None
        if self.use_brain_gnn:
            self.model_name = "brainGNN"
        if len(self.model_args["message_passing"]) == 0:
            self.model_name = "gcn"
            self.gcn_mp_type = self.model_args["message_passing"][0]
        elif len(self.model_args["message_passing with attention"]) > 0:
            self.model_name = "gat"
            self.gat_mp_type = self.model_args["message_passing with attention"][0]
        
        if not self.use_brain_gnn:
            self.pooling = self.model_args["pooling"][0]
    
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


