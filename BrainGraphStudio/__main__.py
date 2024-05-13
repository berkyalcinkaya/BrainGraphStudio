from BrainGraphStudio.utils import BGS_DIR
from BrainGraphStudio.train.train_utils import TRAIN_SCRIPT_DIR
import subprocess
import sys
from PyQt5.QtWidgets import QApplication, QWizard
from PyQt5.QtGui import QPixmap
from BrainGraphStudio.gui.pages import FilePage, ModelPage, HyperParamDialog
import os
import logging
import json
from BrainGraphStudio.utils import write_dict_to_json
import numpy as np
from BrainGraphStudio.nni import configure_nni
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(TRAIN_SCRIPT_DIR))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream = sys.stdout)

class CustomWizard(QWizard):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Build your Graph Neural Network")

        #logo = QPixmap("logo.png").scaled(100,70)
        banner = QPixmap("BrainGraphStudio/gui/ims/banner.png").scaled(125,550)
        #self.setPixmap(QWizard.LogoPixmap, logo)
        self.setPixmap(QWizard.WatermarkPixmap, banner)
        self.setWizardStyle(QWizard.ClassicStyle)

        # Add pages
        self.filePage = FilePage()
        self.modelPage = ModelPage()
        self.hyperParamDialogPage = HyperParamDialog()
        self.pages = [self.filePage, self.modelPage, self.hyperParamDialogPage]

        self.addPage(self.filePage)
        self.addPage(self.modelPage)
        self.addPage(self.hyperParamDialogPage)
    
    def get_file_data(self):
        return self.filePage.get_data()
    
    def get_model_data(self):
        return self.modelPage.get_data()
    
    def get_param_data(self):
        return self.hyperParamDialogPage.get_data()


def write_file_data_to_disk(path, file_data, test_split, seed):
    x = file_data["data"]
    y = file_data["labels"]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = test_split, random_state = seed, stratify = y)


    x_train_path = os.path.join(path,"x_train.npy")
    y_train_path = os.path.join(path,"y_train.npy")
    x_test_path = os.path.join(path,"x_test.npy")
    y_test_path = os.path.join(path,"y_test.npy")
    
    np.save(x_train_path, x_train)
    np.save(y_train_path, y_train)
    logging.info(f"training data saved to {x_train_path}")
    logging.info(f"training labels to saved to {y_train_path}")

    np.save(x_test_path, x_test)
    np.save(y_test_path, y_test)
    logging.info(f"training data saved to {x_test_path}")
    logging.info(f"training labels to saved to {y_test_path}")

    del file_data["data"]
    del file_data["labels"]

    file_data_path = os.path.join(path, "data.json")
    write_dict_to_json(file_data, file_data_path)
    logging.info(f"file data saved to {file_data_path}")

def make_project_dir(project_dir, project_name):
    potential_path = os.path.join(project_dir, project_name)
    counter = 1

    if os.path.exists(potential_path):
        unique_path = potential_path
        while os.path.exists(unique_path):
            # Append a numeric suffix to create a unique directory name
            unique_path = f"{potential_path}_{counter}"
            counter += 1
        logging.info(f"{potential_path} exists. Utilizing {unique_path} instead")
        os.mkdir(unique_path)
        return unique_path
    else:
        os.mkdir(potential_path) 
        logging.info(f"{potential_path} initialized as project directory")
        return potential_path

def main():
    app = QApplication(sys.argv)
    wizard = CustomWizard()
    wizard.show()
    if wizard.exec_() == QWizard.Accepted and wizard.filePage.isComplete():
        file_data = wizard.get_file_data()
        model_data = wizard.get_model_data()
        param_data = wizard.get_param_data()

        seed = param_data["data"]["random_seed"] 
        if seed == 0:
            seed = param_data["data"]["random_seed"] = None
            logging.info("No Random Seed Specified")

 
        python_path = file_data["python_path"]
        if not os.path.exists(python_path):
            python_path = sys.executable
        
        project_path = make_project_dir(file_data["project_dir"], file_data["project_name"])

        write_file_data_to_disk(project_path, file_data, param_data["data"]["test_split"], seed)
        write_dict_to_json(model_data, os.path.join(project_path, "model.json"))

        use_nni = param_data["nni"]["optimization_algorithm"] != "None"
        if use_nni:
            param_data["nni"]["search_space"] = json.loads(param_data["nni"]["search_space"])
        write_dict_to_json(param_data, os.path.join(project_path, "params.json"))

        use_brain_gnn = model_data["use_brain_gnn"]

        if use_nni:
            logging.info("Utilizing An NNI Experiment to Train Models with Hyperparameter Optimization")
            experiment = configure_nni(param_data["nni"], project_path, python_path, brainGNN=use_brain_gnn )
            experiment.run(8080)
            # some sort of model testing here
        else:
            run_dir = os.path.join(BGS_DIR, "train")
            if use_brain_gnn:
                train_file = "BrainGraphStudio.train.train_brain_gnn"
            else:
                train_file = "BrainGraphStudio.train.train_brain_gb"
            
            #train_file = os.path.join(run_dir, train_file)
            logging.info("No Hyperparameter Optimization In Use")
            logging.info(f"Running {python_path} {train_file} {project_path}")
            subprocess.run([python_path, "-m", train_file, project_path])

if __name__ == "__main__":
    main()