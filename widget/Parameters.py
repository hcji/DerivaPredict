# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:50:51 2024

@author: DELL
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow

from uic.parameters import Ui_Dialog

class ParametersUI(QtWidgets.QDialog, Ui_Dialog):
    
    def __init__(self, parent=None):
        super(ParametersUI, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Setting")
        
        self.comboBox_der_type.addItems(['Biochemical', 'Chemical', 'Metabolic'])
        self.comboBox_dta_model.addItems(['CNN-CNN', 'MPNN-CNN', 'Morgan-CNN', 'baseTransformer-CNN'])
        self.add_comboBox_der_model()
        self.comboBox_der_type.currentTextChanged.connect(self.add_comboBox_der_model)
        self.spinBox_n_loop.setMaximum(3)
        self.spinBox_n_loop.setProperty("value", 1)
        
    def add_comboBox_der_model(self):
        if self.comboBox_der_type.currentText() == 'Chemical':
            self.comboBox_der_model.clear()
            self.comboBox_der_model.addItems(['Chemical-Template-based'])
            
        if self.comboBox_der_type.currentText() == 'Biochemical':
            self.comboBox_der_model.clear()
            self.comboBox_der_model.addItems(['Biochemical-Template-based'])
        
        if self.comboBox_der_type.currentText() == 'Metabolic':
            self.comboBox_der_model.clear()
            self.comboBox_der_model.addItems(['BioTransformer-EC-based', 
                                              'BioTransformer-CYP450', 
                                              'BioTransformer-Phase II', 
                                              'BioTransformer-Human gut microbial', 
                                              'BioTransformer-All Human', 
                                              'BioTransformer-Environmental microbial'])
        
        

if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    ui = ParametersUI()
    ui.show()
    sys.exit(app.exec_())
