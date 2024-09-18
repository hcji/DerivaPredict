# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:54:21 2024

@author: DELL
"""
#  -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install torchtext==0.6.0 同core.utils
#pip install configargparse
#pip install sacrebleu
#pip install pytorch_lightning==1.2.3
import os
import sys
if sys.platform.startswith('win'):## need to add environ aug here ##windows版Graphviz要求。
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

import shutil
import string
import random

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
from graphviz import Digraph

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QVariant, QThread
from PyQt5.QtWidgets import QApplication, QMainWindow

from uic.main333ui import Ui_MainWindow
from core import utils
from widget.PubchemSketcher import PubchemSketcherUI
from widget.ChemicalImage import ChemicalImageUI
from widget.Parameters import ParametersUI
from datetime import datetime
import time

class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data, showAllColumn=False):
        QtCore.QAbstractTableModel.__init__(self)
        self.showAllColumn = showAllColumn
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self,col,orientation,role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if type(self._data.columns[col]) == tuple:
                return self._data.columns[col][-1]
            else:
                return self._data.columns[col]
        elif orientation == Qt.Vertical and role == Qt.DisplayRole:
            return (self._data.axes[0][col])
        return None


class Thread_PredictDerivative(QThread): 
    _r = QtCore.pyqtSignal(pd.DataFrame)
    
    def __init__(self, smiles_list, n_loop, n_branch, sim_filter, method):
        super().__init__()
        self.smiles_list = smiles_list
        self.n_loop = n_loop
        self.n_branch = n_branch
        self.sim_filter = sim_filter
        self.method = method
        
    def run(self):
        if self.method == 'Chemical-Template-based':
            derivative_list = utils.predict_compound_derivative_chemical_templete(self.smiles_list, n_loop = self.n_loop, n_branch = self.n_branch, sim_filter = self.sim_filter)
        elif self.method == 'BioTransformer-EC-based':
            derivative_list = utils.predict_compound_derivative_biotransformer(self.smiles_list, n_loop = self.n_loop, sim_filter = self.sim_filter, method = 'ecbased')
        elif self.method == 'BioTransformer-CYP450':
            derivative_list = utils.predict_compound_derivative_biotransformer(self.smiles_list, n_loop = self.n_loop, sim_filter = self.sim_filter, method = 'cyp450')
        elif self.method == 'BioTransformer-Phase II':
            derivative_list = utils.predict_compound_derivative_biotransformer(self.smiles_list, n_loop = self.n_loop, sim_filter = self.sim_filter, method = 'phaseII')
        elif self.method == 'BioTransformer-Human gut microbial':
            derivative_list = utils.predict_compound_derivative_biotransformer(self.smiles_list, n_loop = self.n_loop, sim_filter = self.sim_filter, method = 'hgut')
        elif self.method == 'BioTransformer-All Human':
            derivative_list = utils.predict_compound_derivative_biotransformer(self.smiles_list, n_loop = self.n_loop, sim_filter = self.sim_filter, method = 'allHuman')
        elif self.method == 'BioTransformer-Environmental microbial':
            derivative_list = utils.predict_compound_derivative_biotransformer(self.smiles_list, n_loop = self.n_loop, sim_filter = self.sim_filter, method = 'env')
        else:
            derivative_list = None      
        self._r.emit(derivative_list)
        
    
class Thread_PredictDTI(QThread): 
    _r = QtCore.pyqtSignal(pd.DataFrame)
    
    def __init__(self, smiles_list, target_list, affinity_model_type):
        super().__init__()
        self.smiles_list = smiles_list
        self.target_list = target_list
        self.affinity_model_type = affinity_model_type
        
    def run(self):
        dti_list = utils.predict_compound_target_affinity(smiles_list=self.smiles_list, target_list=self.target_list, affinity_model_type=self.affinity_model_type)
        self._r.emit(dti_list)


class Thread_PredictADMET(QThread): 
    _r = QtCore.pyqtSignal(pd.DataFrame)
    
    def __init__(self, smiles_list):
        super().__init__()
        self.smiles_list = smiles_list
        
    def run(self):
        admet_list = utils.predict_compound_ADMET_property(smiles_list=self.smiles_list)
        self._r.emit(admet_list)

    
class NPDS_App(QMainWindow, Ui_MainWindow):
    
    def __init__(self, parent=None):
        super(NPDS_App, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("NP Derivative Screening")
        self.progressBar.setValue(100)
        self.progressBar.setFormat('Ready')
        
        try:
            shutil.rmtree('temp')  
            os.mkdir('temp') 
        except:
            pass
        
        self.precursor = None
        self.target_list = []
        
        self.PubchemSketcherUI = PubchemSketcherUI()
        self.ChemicalImageUI = ChemicalImageUI()
        self.ParametersUI = ParametersUI()
        
        self.pushButton_chem_add.clicked.connect(self.add_precursor_to_list)
        self.pushButton_chem_upload.clicked.connect(self.upload_precursor_to_list)
        self.pushButton_chem_draw.clicked.connect(self.PubchemSketcherUI.show)
        self.pushButton_chem_clear.clicked.connect(self.listWidget_chem_list.clear)
        self.pushButton_chem_show.clicked.connect(self.show_chemical_image)
        self.pushButton_prot_search.clicked.connect(self.retrieve_targets)
        self.pushButton_prot_add.clicked.connect(self.add_target_to_list)
        self.pushButton_tar_clear.clicked.connect(self.clear_target_list) ## copy this add below clear_prot_search_result_list
        self.pushButton_prot_clear.clicked.connect(self.clear_prot_search_result_list) # new add here
        self.pushButton_tar_predict.clicked.connect(self.predict_compound_derivative)
        self.pushButton_setting.clicked.connect(self.ParametersUI.open)
        self.pushButton_save.clicked.connect(self.save_as_file) # new add here
        self.Thread_PredictDerivative = None
        self.Thread_PredictDTI = None
        self.Thread_ADMET = None
        
        self.derivative_list = None
        self.DTI_list = None
        self.ADMET_list = None
        
        #self.tableWidget_dta_out.cellClicked.connect(self.fill_AMDET_table)
        #
        self.tableWidget_dta_out.setSortingEnabled(True)
        #### 跳转到 show_synthesis_path
        self.tableWidget_dta_out.cellClicked.connect(self.show_synthesis_path)

    def save_as_file(self): #self.derivative_list 可能为none 可能为
        if isinstance(self.derivative_list, pd.DataFrame):
            if not self.derivative_list.empty:      # self.derivative_list不为None且不为空
                n_loop = self.ParametersUI.spinBox_n_loop.value()
                n_branch = self.ParametersUI.spinBox_n_branch.value()
                method = self.ParametersUI.comboBox_der_model.currentText()
                print(n_loop, n_branch, method)
                time = datetime.now().strftime("%m%d_%H%M%S")
                file_name = f'derivative_list_{method}_{n_branch}xx{n_loop}_{time}'
                destpath, filetype = QtWidgets.QFileDialog.getSaveFileName(self, "Select the save path", f"{file_name}", "csv Files (*.csv)")

                if destpath:
                    folder_path = destpath.rsplit('/', 1)[0]
                    print(folder_path, filetype)
                    self.derivative_list.to_csv(destpath)
                    self.DTI_list.to_csv(f'{folder_path}/DTI_list_{method}_{n_branch}xx{n_loop}_{time}.csv')
                    # self.ADMET_list.to_csv(f'{folder_path}/ADMET_list_{method}_{n_branch}xx{n_loop}_{time}.csv')

                self.InforMsg('Finished')
            else:
                self.WarnMsg('No result to save')
        else:
            self.WarnMsg('No result to save')


    def WarnMsg(self, Text):
        msg = QtWidgets.QMessageBox()
        msg.resize(550, 200)
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(Text)
        msg.setWindowTitle("Warning")
        msg.exec_()    
    
    
    def ErrorMsg(self, Text):
        msg = QtWidgets.QMessageBox()
        msg.resize(550, 200)
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(Text)
        msg.setWindowTitle("Error")
        msg.exec_()


    def InforMsg(self, Text):
        msg = QtWidgets.QMessageBox()
        msg.resize(550, 200)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(Text)
        msg.setWindowTitle("Information")
        msg.exec_()
    
    
    def _set_table_widget(self, widget, data):
        widget.setRowCount(0)
        widget.setRowCount(data.shape[0])
        widget.setColumnCount(data.shape[1])
        widget.setHorizontalHeaderLabels(data.columns)
        widget.setVerticalHeaderLabels(data.index.astype(str))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if type(data.iloc[i,j]) == np.float64:
                    item = QtWidgets.QTableWidgetItem()
                    item.setData(Qt.EditRole, QVariant(float(data.iloc[i,j])))
                else:
                    item = QtWidgets.QTableWidgetItem(str(data.iloc[i,j]))
                widget.setItem(i, j, item)

    
    def _set_derivative_list(self, msg):
        self.derivative_list = msg
        
    
    def _set_DTI_list(self, msg):
        self.DTI_list = msg
        
        
    def _set_AMDET_list(self, msg):
        self.ADMET_list = msg
    
    
    def _clear_all(self):
        self.derivative_list = None
        self.DTI_list = None
        self.ADMET_list = None
        self.tableWidget_dta_out.clear()
        self.pushButton_chem_add.setDisabled(True)
        self.pushButton_chem_upload.setDisabled(True)
        self.pushButton_chem_draw.setDisabled(True)
        self.pushButton_chem_clear.setDisabled(True)
        self.pushButton_chem_show.setDisabled(True)
        self.pushButton_prot_search.setDisabled(True)
        self.pushButton_prot_add.setDisabled(True)
        self.pushButton_tar_clear.setDisabled(True)
        self.pushButton_tar_predict.setDisabled(True)
        self.pushButton_setting.setDisabled(True)

        
    def _set_finished(self):
        self.pushButton_chem_add.setDisabled(False)
        self.pushButton_chem_upload.setDisabled(False)
        self.pushButton_chem_draw.setDisabled(False)
        self.pushButton_chem_clear.setDisabled(False)
        self.pushButton_chem_show.setDisabled(False)
        self.pushButton_prot_search.setDisabled(False)
        self.pushButton_prot_add.setDisabled(False)
        self.pushButton_tar_clear.setDisabled(False)
        self.pushButton_tar_predict.setDisabled(False)
        self.pushButton_setting.setDisabled(False)
    
    
    def retrieve_targets(self):
        gene_input = self.plainTextEdit_prot_inp.toPlainText()
        if gene_input == '':
            self.ErrorMsg('No valid targets were retrieved')
            return            
        gene_list = utils.retrieve_gene_from_name(gene_input)
        if gene_list is None:
            self.ErrorMsg('No valid targets were retrieved')
            return
        self._set_table_widget(self.tableWidget_prot_inp, gene_list)
        self.tableWidget_prot_inp.setCurrentCell(0, 0)
    
    
    def add_target_to_list(self):
        try:
            index = self.tableWidget_prot_inp.currentRow()
            gene_symbol = self.tableWidget_prot_inp.item(index, 1).text()
            uniprot_id = self.tableWidget_prot_inp.item(index, 0).text()
            sequence = utils.retrieve_protein_sequence(uniprot_id)
            if sequence is None:
                self.ErrorMsg('No valid sequence is retrieved')
                return
            self.target_list.append([uniprot_id, gene_symbol, sequence])
            target_list_model = TableModel(pd.DataFrame(self.target_list, columns = ['uniprot_id', 'gene_symbol', 'sequence']))
            self.tableView_prot_list.setModel(target_list_model)
            self.InforMsg('Finished')
        except:
            self.ErrorMsg('No target was selected')
        
    
    def clear_target_list(self):
        self.target_list = []
        self.tableView_prot_list.setModel(None)

    def clear_prot_search_result_list(self):
        #self.tableWidget_prot_inp.clear()
        self.tableWidget_prot_inp.setRowCount(0)
        self.tableWidget_prot_inp.setColumnCount(0)

    def add_precursor_to_list(self):
        precursor_smi = self.plainTextEdit_chem_inp.toPlainText()
        if precursor_smi == '':
            self.ErrorMsg('Invalid input molecule')
            return            
        precursor_mol = Chem.MolFromSmiles(precursor_smi)
        if precursor_mol is None:
            self.ErrorMsg('Invalid input molecule')
            return
        self.listWidget_chem_list.addItem(Chem.MolToSmiles(precursor_mol))
        self.InforMsg('Finished')
        
        
    def upload_precursor_to_list(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load", "","Smiles Files (*.smi);;SDF Files (*.sdf)", options=options)
        if fileName:
            if fileName.split('.')[-1] == 'smi':
                with open(fileName) as f:
                    smiles_list = f.readlines()
                mol_list = [Chem.MolFromSmiles(s) for s in smiles_list]
                mol_list = [m for m in mol_list if m is not None]
                smiles_list = [Chem.MolToSmiles(m) for m in mol_list]
            elif fileName.split('.')[-1] == 'sdf':
                mol_list = Chem.SDMolSupplier(fileName)
                smiles_list = [Chem.MolToSmiles(m) for m in mol_list]
            else:
                self.ErrorMsg("Invalid format")
                return None
            for smi in smiles_list:
                self.listWidget_chem_list.addItem(smi)
        self.InforMsg('Finished')
        
        
    def show_chemical_image(self):
        precursor_smi = self.listWidget_chem_list.currentItem()
        if not precursor_smi:
            self.WarnMsg('Please select a compound')
            return
        precursor_smi = precursor_smi.text()
        self.ChemicalImageUI.show()
        precursor_mol = Chem.MolFromSmiles(precursor_smi)
        file_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        Draw.MolToFile(precursor_mol, 'temp/{}.png'.format(file_name))
        self.ChemicalImageUI.label_chem_image.setPixmap(QPixmap('temp/{}.png'.format(file_name)))
        
        
    def predict_compound_derivative(self):
        precursor_smiles_list = [self.listWidget_chem_list.item(x).text() for x in range(self.listWidget_chem_list.count())]
        if (len(precursor_smiles_list) == 0) or (len(self.target_list) == 0):
            self.ErrorMsg('Precursor List or Target List is empty')
            return
        n_loop = self.ParametersUI.spinBox_n_loop.value()
        n_branch = self.ParametersUI.spinBox_n_branch.value()
        sim_filter = self.ParametersUI.doubleSpinBox_sim_filter.value()
        method = self.ParametersUI.comboBox_der_model.currentText()

        self._clear_all()
        self.progressBar.setValue(30)
        self.progressBar.setFormat('Predicting Derivative')
        self.Thread_PredictDerivative = Thread_PredictDerivative(precursor_smiles_list, n_loop, n_branch, sim_filter, method)
        self.Thread_PredictDerivative._r.connect(self._set_derivative_list)
        self.Thread_PredictDerivative.start()
        self.Thread_PredictDerivative.finished.connect(self.predict_DTI)

    
    def predict_DTI(self):
        if self.derivative_list.empty:
            self.ErrorMsg('No valid prediction for current input precursor compound')
            self.progressBar.setValue(100)
            self.progressBar.setFormat('Ready')
            self._set_finished()
        else:
            smiles_list = list(set(list(self.derivative_list['precursor']) + list(self.derivative_list['derivant'])))
            target_list = pd.DataFrame(self.target_list, columns = ['uniprot_id', 'gene_symbol', 'sequence'])
            target_list = list(target_list.loc[:,'sequence'].values)
            affinity_model_type = self.ParametersUI.comboBox_dta_model.currentText()

            self.progressBar.setValue(50)
            self.progressBar.setFormat('Predicting DTI')
            self.Thread_PredictDTI = Thread_PredictDTI(smiles_list, target_list, affinity_model_type)
            self.Thread_PredictDTI._r.connect(self._set_DTI_list)
            self.Thread_PredictDTI.start()
            self.Thread_PredictDTI.finished.connect(self.fill_DTI_table)


    def fill_DTI_table(self):
        target_data = pd.DataFrame(self.target_list, columns = ['uniprot_id', 'gene_symbol', 'sequence'])
        DTI_table = self.DTI_list.pivot(index='SMILES', columns='Target Sequence', values='Predicted Value').reset_index()
        gene_maping = {target_data.loc[i,'sequence']: target_data.loc[i,'gene_symbol'] for i in target_data.index}
        gene_maping['Target Sequence SMILES'] = 'SMILES'
        DTI_table = DTI_table.rename(columns=gene_maping)
        self._set_table_widget(self.tableWidget_dta_out, DTI_table)
        self.tableWidget_dta_out.setCurrentCell(0, 0)
        #self.predict_ADMET()
        # #change
        self.progressBar.setValue(100)
        self.progressBar.setFormat('Ready')
        self._set_finished()
        # # 关闭功能
        # self.predict_ADMET()
        
        
    def predict_ADMET(self):
        smiles_list = list(self.derivative_list['precursor']) + list(self.derivative_list['derivant'])
        
        self.progressBar.setValue(70)
        self.progressBar.setFormat('Predicting AMDET')
        self.Thread_PredictAMDET = Thread_PredictADMET(smiles_list)  
        self.Thread_PredictAMDET._r.connect(self._set_AMDET_list)
        self.Thread_PredictAMDET.start()
        self.Thread_PredictAMDET.finished.connect(self.fill_AMDET_table)


    def fill_AMDET_table(self):
        index = self.tableWidget_dta_out.currentRow()
        current_smiles = self.tableWidget_dta_out.item(index, 0).text()
        
        Physicochemical = utils.refine_compound_ADMET_property(self.ADMET_list, current_smiles, property_class = 'Physicochemical')
        Absorption = utils.refine_compound_ADMET_property(self.ADMET_list, current_smiles, property_class = 'Absorption')
        Distribution = utils.refine_compound_ADMET_property(self.ADMET_list, current_smiles, property_class = 'Distribution')
        Metabolism = utils.refine_compound_ADMET_property(self.ADMET_list, current_smiles, property_class = 'Metabolism')
        Excretion = utils.refine_compound_ADMET_property(self.ADMET_list, current_smiles, property_class = 'Excretion')
        Toxicity = utils.refine_compound_ADMET_property(self.ADMET_list, current_smiles, property_class = 'Toxicity')

        self.tableView_prop_1.setModel(TableModel(Physicochemical))
        self.tableView_prop_2.setModel(TableModel(Absorption))
        self.tableView_prop_3.setModel(TableModel(Distribution))
        self.tableView_prop_4.setModel(TableModel(Metabolism))
        self.tableView_prop_5.setModel(TableModel(Excretion))
        self.tableView_prop_6.setModel(TableModel(Toxicity))
        self.progressBar.setValue(100)
        self.progressBar.setFormat('Ready')
        self._set_finished()

    #### new add here
    def show_synthesis_path(self):
        df = self.derivative_list
        index = self.tableWidget_dta_out.currentRow()
        current_smiles = self.tableWidget_dta_out.item(index, 0).text()
        print(f'get_synthesis_path for current_SMILES:{current_smiles}')
        # 获取当前分子的合成路径
        search_line = df[df['derivant'].isin([current_smiles])].to_dict('records')
        smi_lst = []
        smi_lst.append(current_smiles)
        while len(search_line) != 0:  # 有对应前体。读取前体
            search_result = search_line[0]['precursor']
            smi_lst.insert(0, search_result)
            get_smiles = search_result
            search_line = df[df['derivant'].isin([get_smiles])].to_dict('records')
        # 生成分子图，并生成路径图
        G = Digraph('G', filename='temp/return_synthesis_path')
        G.attr('node', shape='box')
        G.format = 'png'
        G_save_path = 'temp/return_synthesis_path.png'

        current_path = os.getcwd()
        for i in range(len(smi_lst)):
            print(f'synthesis_path:{i + 1}/{len(smi_lst)}', '\n', smi_lst[i])
            try:
                mol = Chem.MolFromSmiles(smi_lst[i])
            except:
                print('invalid molecular SMILES string')
            else:
                save_path = f'{current_path}/temp/smi{i}-mol-graph.png'
                Draw.MolToFile(mol, save_path, size=(400, 400))
                ## G.node 需要读取绝对路径，不然可能报错。
                G.node(name=smi_lst[i], image=save_path, label='', labelloc='top')  # label=smi_lst[i] f'mol{i + 1}'
                if i >= 1:
                    G.edge(smi_lst[i - 1], smi_lst[i])  # 可加label # , label=f"no-level-expansion"
        G.render()
        load_image = QPixmap(G_save_path)
        width = load_image.width()
        height = load_image.height()  ##获取图片高度
        print('original pic size', width, height)
        print('label size', self.label_7.width(), self.label_7.height())
        ratio = (self.label_7.width() / width)
        new_width = int(width * ratio * 0.9)  ##定义新图片的宽和高
        new_height = int(height * ratio * 0.9)
        print('resized pic size', new_width, new_height)
        pic2show = QtGui.QPixmap(G_save_path).scaled(new_width, new_height, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.label_7.setPixmap(pic2show)
        self.predict_ADMET()
        

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    ui = NPDS_App()
    ui.show()
    sys.exit(app.exec_())