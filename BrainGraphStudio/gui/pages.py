import os
from PyQt5.QtWidgets import (QSizePolicy, QComboBox, QGroupBox, QFormLayout, QGridLayout, QApplication, QWizard, 
                             QWizardPage, QVBoxLayout, QPushButton, QFileDialog, QCheckBox, QSpinBox, 
                             QTextEdit, QRadioButton, QLabel, QLineEdit, QGroupBox, QRadioButton, 
                             QDialogButtonBox, QButtonGroup)
import numpy as np
from scipy.io import loadmat
from PyQt5 import QtCore, QtGui
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from .utils import (is_binary, custom_json_dump, is_flist, is_mat, 
                    is_mat_flist, is_npy, is_npy_flist, load_npy_flist, process_label_file)
from BrainGraphStudio.models.model import BrainGB, BrainGNN
import sys


class FilePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Load and Configure Training Data")
        self.setLayout(QGridLayout())
        self.setMinimumHeight(700)

        self.projectNameLabel = QLabel("Project Name")
        self.projectName = QLineEdit()
        self.projectName.setText("test")
        self.projectName.setEnabled(True)
        self.layout().addWidget(self.projectNameLabel, 0,0,1,2)
        self.layout().addWidget(self.projectName, 0,2,1,2)

        self.openFileButton = QPushButton("Select Train Data")
        self.openFileButton.setEnabled(True)
        self.layout().addWidget(self.openFileButton, 1,0,1,4)
        self.openFileButton.clicked.connect(self.openDataFileDialog)

        self.labelButton = QPushButton("Open Label CSV")
        self.labelButton.setToolTip("Select a CSV File containing the labels for the training data")
        self.labelButton.setEnabled(True)
        self.labelButton.clicked.connect(self.openLabelFileDialog)
        self.layout().addWidget(self.labelButton, 2,0,1,4)

        self.openDirButton = QPushButton("Choose Project Location")
        self.openDirButton.setToolTip("Choose a directory where all data/model info will be written. If none, defaults to directory of training data")
        self.openDirButton.setEnabled(True)
        self.layout().addWidget(self.openDirButton, 3,0,1,4)
        self.openDirButton.clicked.connect(self.openDirDialog)

        # self.augmentedCheckbox = QCheckBox("Aug Data      |")
        # self.layout().addWidget(self.augmentedCheckbox, 2,0,1,2)

        # spinBoxLabel = QLabel("Aug Factor")
        # self.augmentationFactor = QSpinBox()
        # self.augmentationFactor.setDisabled(True)
        # self.layout().addWidget(spinBoxLabel,2,2,1,1)
        # self.layout().addWidget(self.augmentationFactor, 2,3,1,1)
        # self.augmentedCheckbox.toggled.connect(self.augmentationFactor.setEnabled)

        chooseLabel = QLabel("Choose variable key")
        self.labelChoose = QComboBox()
        self.labelChoose.setEnabled(False)
        self.labelChoose.currentIndexChanged.connect(self.labelChooseChange)
        self.layout().addWidget(chooseLabel, 4,0,1,2)
        self.layout().addWidget(self.labelChoose,4,2,1,2)

        spinBoxLabel = QLabel("Threshold percentile")
        self.thresholdLevel = QSpinBox()
        self.thresholdLevel.setMaximum(100)
        self.thresholdLevel.setMinimum(0)
        self.thresholdLevel.setToolTip("Edge weights below this percentile will be discarded")
        self.thresholdLevel.setEnabled(False)
        self.layout().addWidget(spinBoxLabel, 5,0,1,2)
        self.layout().addWidget(self.thresholdLevel, 5,2,1,1)

        pythonPathLabel = QLabel("Python Path:")
        self.pythonPathEdit = QLineEdit()
        self.pythonPathEdit.setText(sys.executable)
        self.layout().addWidget(pythonPathLabel, 6,0,1,2)
        self.layout().addWidget(self.pythonPathEdit,7,0,1,6)

        self.textBox = QLabel()
        font = QtGui.QFont()
        font.setBold(True)
        self.textBox.setFont(font)
        self.layout().addWidget(self.textBox,8,0,1,4)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout().addWidget(self.canvas, 9, 0, 4, 4)
        self.canvas.setVisible(False)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        #sizePolicy.setRetainSizeWhenHidden(True)
        self.canvas.setSizePolicy(sizePolicy)

        self.extensions = ["mat", "npy"]

        self.data = []
        self.labels = []
        self.num_examples = 0
        self.num_labels = 0
        self.num_classes = 0
        self.shape = ()
        self.dtype = None
        self.is_binary = False
        self.dataLoaded = False
        self.labelsLoaded = False
        self.overrideComplete = True
        self.validInputs = True
        self.is_mat = False
        self.mat_var_chosen = False
        self.selected_directory = None
        self.filepath = None
        self.data_dir = None
    
    def openDirDialog(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory For Project Files")
        if directory:  # If a directory was selected
            self.selected_directory = directory

    def openDataFileDialog(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Text, Npy, or Mat File", "", "Text Files (*.txt);; Flist files (*.flist);; Mat file (*.mat);; Npy files (*.npy)", options=options)
        if filePath:
            self.filepath = filePath
            self.data_dir = os.path.dirname(self.filepath)
            self.overrideComplete = False

            if is_mat_flist(self.filepath) or is_mat(self.filepath):
                self.is_mat = True
                
                if is_mat_flist(self.filepath):
                    with open(self.filepath, "r") as f:
                        self.files = [file.strip() for file in f.readlines()]
                    self.allowMatVarChoose(loadmat(self.files[0]).keys())           
                self.dataLoaded = True
                self.resetTextBox()

            elif is_npy(self.filepath) or is_npy_flist(self.filepath):
                if is_npy(self.filepath):
                    self.data = np.load(filePath)
                elif is_npy_flist(self.filepath):
                    self.data = load_npy_flist(self.filepath)
                self.getDataAttributes()
                self.checkBinary()
                self.dataLoaded = True
                self.labelChoose.setEnabled(False)
                self.resetTextBox()
            else:
                self.setTextBox("Cannot Read Data\nEnsure Files are in the correct format.")
        self.completeChanged.emit()
    
    def setTextBox(self, text):
        self.textBox.clear()
        self.textBox.setText(text)
    
    def addToTextBox(self, text):
        curr_text = self.textBox.text()
        curr_text+= text
        self.textBox.setText(curr_text)
    
    def openLabelFileDialog(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open CSV or Text File Containing Training Data Labels", "", "Csv files (*.csv);; Text Files (*.txt)")
        if filePath:
            self.overrideComplete = False
            is_valid, data = process_label_file(filePath)
            
            if is_valid:
                self.labels = np.array(data)
                self.labelsLoaded = True
                self.getLabelAttributes()
                self.resetTextBox()
                self.drawLabelHistogram()
            else:
                self.dataLoaded = False
                self.labels = []
        else:
            if not self.dataLoaded:
                self.overrideComplete = True
            else:
                self.overrideComplete = False
        self.completeChanged.emit()
    
    def drawLabelHistogram(self):
        if self.labels is not None and len(self.labels) > 0:
            self.canvas.setVisible(True)
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.hist(self.labels, edgecolor='black')
            unique_labels = np.unique(self.labels)
            ax.set_xticks(unique_labels)
            ax.set_xlabel('Label')
            ax.set_ylabel('Frequency')
            ax.set_title('Label Distribution')
            self.figure.tight_layout()
            self.canvas.draw()
            #self.figure.set_size_inches(2,3)
        else:
            self.canvas.setVisible(False)

    def isComplete(self):
        return (self.overrideComplete or 
            (self.dataLoaded and self.labelsLoaded and self.validInputs))

    def getLabelAttributes(self):
        labels = self.labels
        self.num_classes = len(np.unique(labels))
        self.num_labels = len(labels)
        if self.dataLoaded:
            self.checkValidInput()

    def getDataAttributes(self):
        data = self.data
        self.num_examples = len(data)
        self.shape = data.shape
        self.num_features = self.shape[-1]
        self.is_binary = is_binary(data)
        self.dtype = data.dtype
        if self.labelsLoaded:
            self.checkValidInput()
            
    def checkBinary(self):
        if not self.is_binary:
            self.thresholdLevel.setEnabled(True)
            self.thresholdLevel.setValue(100)
        else:
            self.thresholdLevel.setEnabled(False)

    def allowMatVarChoose(self, matKeys):
        self.labelChoose.setEnabled(True)
        self.labelChoose.clear()
        self.labelChoose.blockSignals(True)
        self.labelChoose.addItems(matKeys)
        self.labelChoose.setCurrentIndex(-1)
        self.labelChoose.blockSignals(False)

    
    def augmentationCheckBoxClicked(self):
        if self.augmentedCheckbox.isChecked():
            self.augmentationFactor.setEnabled(True)
            self.augmentationFactor.setValue(15)
        else:
            self.augmentationFactor.setEnabled(False)
            self.augmentationFactor.setValue(True)

    def labelChooseChange(self):
        if self.labelChoose.currentIndex() == -1:
            self.mat_var_chosen = False
            return
        self.mat_var_chosen = True
        key = self.labelChoose.currentText()
        self.data = self.getMatData(key)
        self.getDataAttributes()
        self.checkBinary()
        self.resetTextBox()
    
    def getMatData(self, key):
        return np.array([loadmat(file)[key] for file in self.files])

    def resetTextBox(self, custom_data_text=None, custom_label_text=None):
        self.textBox.clear()

        if self.labelsLoaded:
            if custom_label_text is not None:
                self.setTextBox(custom_label_text)
            else:
                self.addLabelAttributesToText()
        if self.dataLoaded:
            if self.labelsLoaded:
                self.addToTextBox("\n\n")
            
            if custom_data_text is not None:
                self.addToTextBox(custom_data_text)
            else:
                self.addDataAttributesToText()
        
        if self.dataLoaded and self.labelsLoaded:
            self.checkValidInput()

    def addDataAttributesToText(self):
        if not self.dataLoaded:
            return
        if self.is_mat and not self.mat_var_chosen:
            self.addToTextBox(f'''{self.filepath}\nnum_examples: {len(self.files)}\n\nSelect the Mat file variable to continue''')
        else:
            text = (f"num_examples: {self.num_examples}\nshape: {self.shape}\ndtype: {self.dtype}\nis_binary: {self.is_binary}")
            if self.is_binary:
                text+= "\n\nWeighted Networks are Recommended"
            self.addToTextBox(text)
        
    def getLabelAttributeAsText(self):
        return f"num_labels: {self.num_labels}\nnum_classes: {self.num_classes}"

    def addLabelAttributesToText(self):
        if not self.labelsLoaded:
            return
        label_text = self.getLabelAttributeAsText()
        text = self.textBox.text()
        text = text + "\n" + label_text
        self.textBox.setText(text)
    
    def checkValidInput(self, add_text = True):
        if self.num_examples!= self.num_labels:
            self.addToTextBox("\n\nERROR: Number of labels and examples do not match")
        if self.num_examples<10:
            self.addToTextBox("\n\n ERROR: Less than 10 examples present")
        self.validInputs = self.num_examples == self.num_labels and self.num_labels>=10
        if self.is_mat and not self.mat_var_chosen:
            self.validInputs = False
        self.completeChanged.emit()

    def get_data(self):
        if self.selected_directory is not None:
            project_dir = self.selected_directory
        else:
            project_dir = self.data_dir
        
        data_dict = {
            "project_name": self.projectName.text(),
            "project_dir": project_dir,
            "num_classes": self.num_classes,
            "num_features": self.num_features,
            "data": self.data,
            "labels": self.labels,
            "shape": self.shape,
            "type": str(self.dtype),
            "is_binary": bool(self.is_binary),
            # "augmentation": self.augmentedCheckbox.isChecked(),
            # "aug_factor": self.augmentationFactor.value(),
            "threshold": self.thresholdLevel.value(),
            "python_path": self.pythonPathEdit.text()
        }

        return data_dict


class ModelPage(QWizardPage):
    def __init__(self):
        super().__init__()

        self.setTitle("Build your GNN")
        self.setSubTitle("Select from preimplemented models or customize componentry")

        myFont=QtGui.QFont()
        myFont.setBold(True)

        self.setWindowTitle("Graph Neural Network Customization")
        self.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout()
        
        boldLabel1 = QLabel("Use Preimplemented Model")
        boldLabel1.setFont(myFont)
        self.layout.addWidget(boldLabel1)
        self.models_bg = QButtonGroup()
        self.models_bg.setExclusive(False)
        #self.use_brain_cnn = QCheckBox("BrainNetCNN")
        self.use_brain_gnn = QCheckBox("BrainGNN")
        #self.models_bg.addButton(self.use_brain_cnn,1)
        self.models_bg.addButton(self.use_brain_gnn,2)
        self.models_bg.buttonClicked.connect(self.use_preimpl_model)
        #self.layout.addWidget(self.use_brain_cnn)
        self.layout.addWidget(self.use_brain_gnn)

        self.groupboxes = []

        label = QLabel("Or Customize GNN")
        label.setFont(myFont)
        self.layout.addWidget(label)

        self.node_features_group = QGroupBox("Node Features")
        self.node_features_layout = QFormLayout()
        self.node_features_layout.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.node_features_group.setLayout(self.node_features_layout)
        self.groupboxes.append(self.node_features_group)

        self.graph_conv_group = QGroupBox("Graph Convolution Layer Type")
        self.graph_conv_attention_checkbox = QCheckBox("USE ATTENTION")
        self.graph_conv_layout = QFormLayout()
        self.graph_conv_layout.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.graph_conv_group.setLayout(self.graph_conv_layout)
        self.groupboxes.append(self.graph_conv_group)

        self.pooling_group = QGroupBox("Pooling Strategies")
        self.pooling_layout = QFormLayout()
        self.pooling_layout.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.pooling_group.setLayout(self.pooling_layout)
        self.groupboxes.append(self.pooling_group)

        self.layout.addWidget(self.node_features_group)
        self.layout.addWidget(self.graph_conv_group)
        self.layout.addWidget(self.pooling_group)

        self.setLayout(self.layout)

        # Node Features
        self.node_features_options = ['identity', 'degree', 'degree_bin', 'LDP', 'node2vec', 'adj', 'diff_matrix',
                                 'eigenvector', 'eigen_norm']
        self.node_features_radios = []
        for option in self.node_features_options:
            radio = QRadioButton(option)
            self.node_features_layout.addWidget(radio, )
            self.node_features_radios.append(radio)
        self.node_features_radios[0].setChecked(True)

        # Graph Convolution Layer Type
        self.graph_conv_options_mp = ["weighted_sum", "bin_concate", "edge_weight_concate", "node_edge_concate", "node_concate"]
        self.graph_conv_options_ma = ["attention_weighted", "attention_edge_weighted", "sum_attention_edge", "edge_node_concate", "node_concate"]
        self.graph_conv_radios = []

        self.graph_conv_layout.addWidget(self.graph_conv_attention_checkbox)

        for option in self.graph_conv_options_mp:
            radio = QRadioButton(option)
            self.graph_conv_layout.addRow(radio)
            self.graph_conv_radios.append(radio)
        self.graph_conv_radios[0].setChecked(True)

        # Pooling Strategies
        self.pooling_options = ["mean", "sum", "concat"]
        self.pooling_radios = []
        for option in self.pooling_options:
            radio = QRadioButton(option)
            self.pooling_layout.addWidget(radio)
            self.pooling_radios.append(radio)
        self.pooling_radios[0].setChecked(True)


        self.graph_conv_attention_checkbox.stateChanged.connect(self.update_graph_conv_options)
    
    def use_preimpl_model(self):
        self.toggle_gbs(self.models_bg.checkedButton() is None)
        # if self.models_bg.checkedId() ==2:
        #     self.use_brain_cnn.setChecked(False)
        if self.models_bg.checkedId()==1:
            self.use_brain_gnn.setChecked(False)

    def toggle_gbs(self, b):
        for groupbox in self.groupboxes:
                groupbox.setEnabled(b)
        
    def update_graph_conv_options(self):
        use_attention = self.graph_conv_attention_checkbox.isChecked()
        options_to_use = self.graph_conv_options_ma if use_attention else self.graph_conv_options_mp

        for radio, new_text in zip(self.graph_conv_radios, options_to_use):
            radio.setText(new_text)

    def get_data(self):
        data = {
            "use_brain_gnn": bool(self.use_brain_gnn.isChecked()),
            "node_features": str([radio.text() for radio in self.node_features_radios if radio.isChecked()][0]),
            "message_passing": "",
            "message_passing_w_attn": "",
            "pooling": str([radio.text() for radio in self.pooling_radios if radio.isChecked()][0])
        }
    
        if self.graph_conv_attention_checkbox.isChecked():
            mp_key = "message_passing_w_attn"
        else:
            mp_key = "message_passing"

        data[mp_key] = str([radio.text() for radio in self.graph_conv_radios if radio.isChecked()][0])

        return data

class HyperParamDialog(QWizardPage):
    def __init__(self):
        super().__init__()

        self.setTitle("Select Hyperparameters")
        self.setSubTitle("Choose hyperparameters, or define parameter search spaces")
        self.setFixedHeight(800)

        self.caption = '''Define hyperparameter searchspace in json using <a href="https://nni.readthedocs.io/en/stable/hpo/search_space.html">NNI specs</a>:
        <br>A parameter is optimizable if it has an entry in the json. Delete the parameter's <br> entry to remove it as a search space dimension. 
        <br> Ensure search space type is compatible with chosen optimization algorithm'''
        
        self.bold_font = QtGui.QFont()
        self.bold_font.setBold(True)

        self.setWindowTitle("Select Parameters")
        self.make_layout()

        # Store the original size of the window
        self.original_size = self.size()
        self.param_data = {}

    def make_layout(self):
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Create the caption label for search space with HTML formatting
        self.caption_label = QLabel(self.caption)
        self.caption_label.setTextFormat(QtCore.Qt.RichText)  # Set text format to RichText for HTML support
        self.caption_label.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        self.caption_label.setOpenExternalLinks(True)
        self.caption_label.setVisible(False)

        # Create the QTextEdit for search space but keep it hidden initially
        self.search_space_text = QTextEdit()
        self.search_space_text.setVisible(False)

        # Add the caption label and QTextEdit to the layout
        self.layout.addWidget(self.caption_label, 0, 2, 3,1)
        self.layout.addWidget(self.search_space_text, 3, 2, 20, 1)

    def initializePage(self):
        for i in reversed(range(self.layout.count())): 
            widget = self.layout.itemAt(i).widget()
            if widget is not None and widget not in [self.search_space_text, self.caption_label]:
                widget.setParent(None)

        # Example model selection logic
        modelType = self.wizard().page(1).use_brain_gnn.isChecked() #or self.wizard().page(1).use_brain_cnn.isChecked()
        if modelType:
            model = BrainGNN  # Replace with your actual model class
        else:
            model = BrainGB  # Replace with your actual model class

        self.params = []
        self.row = -1
        self.main_column_span = 2  # Main widgets span two columns

        for key, value in model.params.items():
            subdata = {}
            self.param_data[key] = subdata
            self.row += 1
            self.layout.addWidget(self.make_bold_label(key), self.row, 0, 1, self.main_column_span)
            for param in value:
                subdata[param.name] = param
                self.params.append(param)
                self.row += 1
                widget = param.get_widget()
                if type(widget) is tuple:
                    label, widget = widget
                    self.layout.addWidget(label, self.row, 0)
                    self.layout.addWidget(widget, self.row, 1)
                else:
                    self.layout.addWidget(widget, self.row, 0, 1, self.main_column_span)
        self.configure_nni_dropdown()

    def configure_nni_dropdown(self):
        nni_params = {"search_space":self.search_space_text.toPlainText}
        self.param_data["nni"] = nni_params
        
        self.row += 1
        self.layout.addWidget(self.make_bold_label("hyperparameter search"), self.row, 0)
        self.row+=1

        self.nni_dropdown = QComboBox()
        self.nni_dropdown.addItems(["None", "random", "GridSearch", "TPE", "Evolution", "Anneal", "Evolution", 
                                    "Hyperband", "SMAC", "Batch", "Hyperband", "Metis", "BOHB", "GP", "PBT", "DNGO"])
        self.nni_dropdown.setToolTip("Choose a hyperparameter optimization algorithm to enable intelligent hyperparameter search. See NNI documentation for details on each algorithm.")
        nni_params["optimization_algorithm"] = self.nni_dropdown.currentText
        self.layout.addWidget(QLabel("optimization"), self.row, 0, 1, self.main_column_span)
        self.layout.addWidget(self.nni_dropdown, self.row, 1)
        self.nni_dropdown.currentIndexChanged.connect(self.nni_dropdown_change)
        self.row += 1

        self.assesor_dropdown = QComboBox()
        self.assesor_dropdown.addItems(["None", "Medianstop", "Curvefitting"])
        self.assesor_dropdown.setToolTip("Assessors dictate early stopping protocols. See NNI documentation for more details")
        self.layout.addWidget(QLabel("assessors"), self.row, 0,1,self.main_column_span)
        self.layout.addWidget(self.assesor_dropdown, self.row, 1)
        nni_params["assessor_algorithm"] = self.assesor_dropdown.currentText
        self.row += 1

        self.num_trials_spin = QSpinBox()
        self.num_trials_spin.setValue(10)
        self.num_trials_spin.setMaximum(1000)
        self.layout.addWidget(QLabel("max trials"), self.row, 0)
        self.layout.addWidget(self.num_trials_spin, self.row, 1)
        nni_params["n_trials"] = self.num_trials_spin.value
        self.row += 1

        self.max_time_edit = QLineEdit()
        self.max_time_edit.setText("24hr")
        self.layout.addWidget(QLabel("max time"), self.row, 0)
        self.layout.addWidget(self.max_time_edit, self.row, 1)
        nni_params["max_time"] = self.max_time_edit.text
        self.row += 1

    def nni_dropdown_change(self):
        selected = self.nni_dropdown.currentText()
        if selected and selected != "None":
            self.make_search_space_json()
            self.caption_label.setVisible(True)
            self.search_space_text.setVisible(True)
            self.num_trials_spin.setEnabled(True)
            self.max_time_edit.setEnabled(True)

            # Increase window width by 150%
            self.wizard().resize(int(self.original_size.width() * 1.5), self.original_size.height())
        else:
            self.search_space_text.clear()
            self.search_space_text.setVisible(False)
            self.caption_label.setVisible(False)
            self.num_trials_spin.setEnabled(False)
            self.max_time_edit.setEnabled(False)

            # Reset window to original size
            self.wizard().resize(self.original_size)

    def make_search_space_json(self):
        self.search_space = {}
        for param in self.params:
            if param.optimizable:
                self.add_param_to_search_space(param)
        self.search_space_text.clear()
        self.search_space_text.setText(custom_json_dump(self.search_space))

    def add_param_to_search_space(self, param):
        name = param.name
        search_type = param.default_search_type
        space = param.default_search_space
        self.search_space[name] = {"_type": search_type, "_value": space}

    def make_bold_label(self, text):
        label = QLabel(text)
        label.setFont(self.bold_font)
        return label
    
    def cleanupPage(self):
        # Hide the search space QTextEdit and caption label
        self.search_space_text.setVisible(False)
        self.caption_label.setVisible(False)

        # Reset the dropdown to "None" or to its default state
        self.nni_dropdown.setCurrentIndex(self.nni_dropdown.findText("None"))

        # Disable and reset other widgets as needed
        self.num_trials_spin.setEnabled(False)
        self.max_time_edit.setEnabled(False)

        # Reset window to original size
        self.wizard().resize(self.original_size)

        # Call the base class cleanup
        super().cleanupPage()
    
    def extract_nni_data(self):
        nni_out_data = {}
        for key, value in self.param_data["nni"].items():
            if key == "search_space":
                selected = self.nni_dropdown.currentText()
                if selected and selected != "None":
                    val = value()
                else:
                    val = ""
            else:
                val = value()
            nni_out_data[key] = val
        return nni_out_data

    def get_data(self):
        out_data = {}
        out_data["nni"] = self.extract_nni_data()
        for key,value in self.param_data.items():
            if key != "nni":
                sub_dict = {key2:val2.get_value() for key2,val2 in value.items()}
                out_data[key] = sub_dict
        return out_data
