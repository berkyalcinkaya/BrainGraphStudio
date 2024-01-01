import sys
from PyQt5.QtWidgets import QApplication, QWizard
from PyQt5.QtGui import QPixmap
from brain_class.gui.pages import FilePage, ModelPage, HyperParamDialog


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
        self.hyperParamDialog = HyperParamDialog()

        self.addPage(FilePage())
        self.addPage(ModelPage())
        self.addPage(HyperParamDialog())
    
    def collectData(wizard):
        pass

def main():
    app = QApplication(sys.argv)
    wizard = CustomWizard()
    wizard.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()