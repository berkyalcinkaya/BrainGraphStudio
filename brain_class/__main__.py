import sys
from PyQt5.QtWidgets import QApplication, QWizard
from PyQt5.QtGui import QPixmap
from brain_class.gui.pages import FilePage, ModelPage, HyperParamDialog
from brain_class.data import apply_transforms, convert_raw_to_datas, density_threshold
from brain_class.models.model import build_model
import numpy as np


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
    
def data_parser(file_args, model_args):
    x,y = file_args["data"],file_args["labels"]
    thresh = file_args["threshold"] 
    
    if thresh == 0:
        x = np.zeros_like(x)
    elif thresh < 100:
        x = density_threshold(x,thresh)
    
    data_list = convert_raw_to_datas(x, y)
    data_array = np.array(apply_transforms(data_list, model_args["node_features"]))
    return data_array


def main():
    app = QApplication(sys.argv)
    wizard = CustomWizard()
    wizard.show()
    if wizard.exec_() == QWizard.Accepted and wizard.filePage.isComplete():
        file_data = wizard.get_file_data()
        model_data = wizard.get_model_data()
        param_data = wizard.get_param_data()

        
        data = data_parser(file_data, model_data)

    


    sys.exit() 

if __name__ == "__main__":
    main()