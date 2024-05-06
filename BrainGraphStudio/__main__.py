import sys
from PyQt5.QtWidgets import QApplication, QWizard
from PyQt5.QtGui import QPixmap
from BrainGraphStudio.gui.pages import FilePage, ModelPage, HyperParamDialog
import os
import logging
from BrainGraphStudio.utils import write_dict_to_json
import numpy as np
from BrainGraphStudio.nni import configure_nni

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream = sys.stdout)

class CustomWizard(QWizard):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Build your Graph Neural Network")

        #logo = QPixmap("logo.png").scaled(100,70)
        banner = QPixmap("brain_class/gui/ims/banner.png").scaled(125,550)
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


def write_file_data_to_disk(path, file_data):
    x = file_data["data"]
    y = file_data["labels"]
    x_path = os.path.join(path,"x.npy")
    y_path = os.path.join(path,"y.npy")
    
    np.save(x_path, x)
    np.save(y_path, y)
    logging.info(f"training data saved to {x_path}")
    logging.info(f"training labels to saved to {y_path}")

    del file_data["data"]
    del file_data["labels"]

    file_data_path = os.path.join(path, "data.json")

    write_dict_to_json(file_data, file_data_path)
    logging.info(f"file data saved to {file_data_path}")

def make_project_dir(project_dir, project_name):
    new_path = os.path.join(project_dir, project_name)
    if os.path.exists(new_path):
        old_path = new_path
        new_path = new_path+"-2"
        logging.warning(f"{old_path} already exists")
    else:
        os.mkdir(new_path) 
    logging.info(f"{new_path} initialized as project directory")
    return new_path

def main():
    app = QApplication(sys.argv)
    wizard = CustomWizard()
    wizard.show()
    if wizard.exec_() == QWizard.Accepted and wizard.filePage.isComplete():
        file_data = wizard.get_file_data()
        model_data = wizard.get_model_data()
        param_data = wizard.get_param_data()

        python_path = file_data["python_path"]
        if not os.path.exists(python_path):
            python_path = sys.executable
        
        project_path = make_project_dir(file_data["project_dir"], file_data["project_name"])

        write_file_data_to_disk(project_path, file_data)
        write_dict_to_json(model_data, os.path.join(project_path, "model.json"))
        write_dict_to_json(param_data)

        use_nni = param_data["nni"]["optimization_algorithm"] != "None"
        if use_nni:
            experiment = configure_nni(param_data["nni"], project_path, python_path)

        
    sys.exit() 

if __name__ == "__main__":
    main()