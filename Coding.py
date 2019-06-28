from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout,QDesktopWidget, QWidget,QTableWidget,QTableView,QTableWidgetItem,QHeaderView,QGraphicsScene,QGraphicsPixmapItem,QFileDialog
from Design import Ui_MainWindow

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from xlrd import open_workbook
from openpyxl.reader.excel import load_workbook
from shutil import copyfile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from skimage import data, img_as_float,io
from skimage.measure import compare_ssim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB

class MainWindow(QWidget,Ui_MainWindow):

    dataset_file_path = ""
    select_classes_index=0
    
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)

        self.model = QtGui.QStandardItemModel(self)
        self.model_x_train = QtGui.QStandardItemModel(self)
        self.btn_dataset_import.clicked.connect(self.button_Dataset_Load_Other)
        self.btn_PCA.clicked.connect(self.button_PCA)
        self.btn_KFOLD.clicked.connect(self.button_KFOLD)
        self.btn_apply.clicked.connect(self.btn_Apply)
        self.cmb_KFOLD_classes.activated[str].connect(self.onActivated_Classes)
        self.table_details.cellClicked.connect(self.onSelected)
    
    def onSelected(self,row,column):
        self.get_Details(column-1)
       
    x_index = []
    y_index = -1
    def onSelected_Load(self,item):
        self.x_index = []
        self.y_index = item.column()

        for i in range(len(self.dataset.iloc[0])):
            if(i != self.y_index):
                self.x_index.append(i)
        
        print("x: ",self.x_index)
        print("y: ",self.y_index)
        
        self.X = self.dataset.iloc[:, self.x_index].values
        self.y = self.dataset.iloc[:, self.y_index].values
        
        print(self.X)
        print(self.y)
    
    def get_Details(self,_column): 
        self.list_x_train.clear()
        self.list_x_test.clear()
        self.list_y_train.clear()
        self.list_y_test.clear()
        
        self.list_x_train.setColumnCount(len(self.classes_X_Train[_column][0]))
        self.list_x_train.setRowCount(len(self.classes_X_Train[_column]))
        for i,row in enumerate(self.classes_X_Train[_column]):
           for j,cell in enumerate(row):
               self.list_x_train.setItem(i,j, QTableWidgetItem(str(cell)))
        self.list_x_train.horizontalHeader().setStretchLastSection(True)
        self.list_x_train.resizeColumnsToContents()
        #self.lbl_x_train_count.setText(str(len(self.classes_X_Train[_row])))
                
        print("x_test--->",self.classes_X_Test[_column][0])
        self.list_x_test.setColumnCount(len(self.classes_X_Test[_column][0]))
        self.list_x_test.setRowCount(len(self.classes_X_Test[_column]))
        for i,row in enumerate(self.classes_X_Test[_column]):
           for j,cell in enumerate(row):
               self.list_x_test.setItem(i,j, QTableWidgetItem(str(cell)))
        self.list_x_test.horizontalHeader().setStretchLastSection(True)
        self.list_x_test.resizeColumnsToContents()
        #self.lbl_x_test_count.setText(str(len(self.classes_X_Test[_row])))


        self.list_y_train.setColumnCount(1)
        self.list_y_train.setRowCount(len(self.classes_Y_Train[_column]))
        for i,row in enumerate(self.classes_Y_Train[_column]):
            self.list_y_train.setItem(i,0, QTableWidgetItem(str(row)))
        self.list_y_train.horizontalHeader().setStretchLastSection(True)
        self.list_y_train.resizeColumnsToContents()
        #self.lbl_y_train_count.setText(str(len(self.classes_Y_Train[_row])))
        
        self.list_y_test.setColumnCount(1)
        self.list_y_test.setRowCount(len(self.classes_Y_Test[_column]))
        for i,row in enumerate(self.classes_Y_Test[_column]):
            self.list_y_test.setItem(i,0, QTableWidgetItem(str(row)))
        self.list_y_test.horizontalHeader().setStretchLastSection(True)
        self.list_y_test.resizeColumnsToContents()
    
    def onActivated_Classes(self, text):
        self.select_classes_index=self.cmb_KFOLD_classes.currentIndex()
        
        print("x train :",str(self.cmb_KFOLD_classes.currentIndex()),str(len(self.classes_X_Train[self.cmb_KFOLD_classes.currentIndex()])))
        print("x test :",str(self.cmb_KFOLD_classes.currentIndex()),str(len(self.classes_X_Test[self.cmb_KFOLD_classes.currentIndex()])))
        print("y train :",str(self.cmb_KFOLD_classes.currentIndex()),str(len(self.classes_Y_Train[self.cmb_KFOLD_classes.currentIndex()])))
        print("y test :",str(self.cmb_KFOLD_classes.currentIndex()),str(len(self.classes_Y_Test[self.cmb_KFOLD_classes.currentIndex()])))
        
        self.list_x_train.clear()
        self.list_x_train.setColumnCount(len(self.classes_X_Train[self.cmb_KFOLD_classes.currentIndex()][0]))
        self.list_x_train.setRowCount(len(self.classes_X_Train[self.cmb_KFOLD_classes.currentIndex()]))
        for i,row in enumerate(self.classes_X_Train[self.cmb_KFOLD_classes.currentIndex()]):
           for j,cell in enumerate(row):
               self.list_x_train.setItem(i,j, QTableWidgetItem(str(cell)))
        self.list_x_train.horizontalHeader().setStretchLastSection(True)
        self.list_x_train.resizeColumnsToContents()
        #self.lbl_x_train_count.setText(str(len(self.classes_X_Train[self.cmb_KFOLD_classes.currentIndex()])))
        
        self.list_x_test.clear()
        self.list_x_test.setColumnCount(len(self.classes_X_Test[self.cmb_KFOLD_classes.currentIndex()][0]))
        self.list_x_test.setRowCount(len(self.classes_X_Test[self.cmb_KFOLD_classes.currentIndex()]))
        for i,row in enumerate(self.classes_X_Test[self.cmb_KFOLD_classes.currentIndex()]):
           for j,cell in enumerate(row):
               self.list_x_test.setItem(i,j, QTableWidgetItem(str(cell)))
        self.list_x_test.horizontalHeader().setStretchLastSection(True)
        self.list_x_test.resizeColumnsToContents()
        #self.lbl_x_test_count.setText(str(len(self.classes_X_Test[self.cmb_KFOLD_classes.currentIndex()])))
        
        self.list_y_train.clear()
        self.list_y_train.setColumnCount(1)
        self.list_y_train.setRowCount(len(self.classes_Y_Train[self.cmb_KFOLD_classes.currentIndex()]))
        for i,row in enumerate(self.classes_Y_Train[self.cmb_KFOLD_classes.currentIndex()]):
            self.list_y_train.setItem(i,0, QTableWidgetItem(str(row)))
        self.list_y_train.horizontalHeader().setStretchLastSection(True)
        self.list_y_train.resizeColumnsToContents()
        #self.lbl_y_train_count.setText(str(len(self.classes_Y_Train[self.cmb_KFOLD_classes.currentIndex()])))
        
        self.list_y_test.clear()
        self.list_y_test.setColumnCount(1)
        self.list_y_test.setRowCount(len(self.classes_Y_Test[self.cmb_KFOLD_classes.currentIndex()]))
        for i,row in enumerate(self.classes_Y_Test[self.cmb_KFOLD_classes.currentIndex()]):
            self.list_y_test.setItem(i,0, QTableWidgetItem(str(row)))
        self.list_y_test.horizontalHeader().setStretchLastSection(True)
        self.list_y_test.resizeColumnsToContents()
        #self.lbl_y_test_count.setText(str(len(self.classes_Y_Test[self.cmb_KFOLD_classes.currentIndex()])))
        
        #y için seçilen kolon verisi label dizisine uyarlanmamıştır. bu alan için seçilen y güncellenmeli.
        #self.list_x_train.setHorizontalHeaderLabels(self.header_labels[0:6])
        #self.list_x_test.setHorizontalHeaderLabels(self.header_labels[0:6])
        #self.list_y_train.setHorizontalHeaderLabels(self.header_labels[6:7])
        #self.list_y_test.setHorizontalHeaderLabels(self.header_labels[6:7])
      
    dataset = []
    X,Y=[],[]
    def button_Dataset_Load_Other(self):
        file,_ = QFileDialog.getOpenFileName(self, 'Open file', './',"CSV files (*.csv)")
        #copyfile(file, "./"+self.dataset_file_path)
        self.dataset_file_path = file
        print(self.dataset_file_path)
        #self.dataset = pd.read_csv(self.dataset_file_path, engine='python')  
        #self.read_CSV(self.dataset_file_path)
        self.dataset = pd.read_csv(self.dataset_file_path, engine='python')
        self.dataset = self.dataset.values
        
        print("yükleme--->",len(self.dataset[0]))
        
        #print(len(self.dataset))
        self.table_dataset_load.clear()
        self.table_dataset_load.setColumnCount(len(self.dataset[0]))
        self.table_dataset_load.setRowCount(len(self.dataset))
        for i,row in enumerate(self.dataset):
           for j,cell in enumerate(row):
               self.table_dataset_load.setItem(i,j, QTableWidgetItem(str(cell)))
        self.table_dataset_load.horizontalHeader().setStretchLastSection(True)
        self.table_dataset_load.resizeColumnsToContents()
        
        self.X = self.dataset[:, 0:(len(self.dataset[0])-1)]
        self.Y = self.dataset[:, len(self.dataset[0])-1]
    
    
    def read_CSV(self,file):
        with open(file, 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                lines=[]
                for value in row:
                    lines.append(str(round(float(value),3)))
                    
                self.dataset.append(lines)
        csvFile.close()
    
    def button_PCA(self):       
        X = self.dataset[:, 0:(len(self.dataset[0])-1)]
        pca = PCA(n_components=5)
        pca.fit(X)
        
        features = pca.transform(X)
        print(type(features))
        
        self.table_dataset_load.clear()
        self.table_dataset_load.setColumnCount(len(features[0]))
        self.table_dataset_load.setRowCount(len(features))
        for i,row in enumerate(features):
           for j,cell in enumerate(row):
               self.table_dataset_load.setItem(i,j, QTableWidgetItem(str(cell)))
        self.table_dataset_load.horizontalHeader().setStretchLastSection(True)
        self.table_dataset_load.resizeColumnsToContents()
        

        self.X = features
        self.Y = self.dataset[:, len(self.dataset[0])-1]
        
        print(len(self.dataset))
        self.dataset = features

    
    classes_X_Train,classes_X_Test=[],[]
    classes_Y_Train,classes_Y_Test=[],[]        
    X_train, X_test, y_train, y_test=0,0,0,0
    split = 0
    def button_KFOLD(self):
        
        X_train, X_test, y_train, y_test=0,0,0,0
        self.classes_X_Train,self.classes_X_Test=[],[]
        self.classes_Y_Train,self.classes_Y_Test=[],[]
        
        self.cmb_KFOLD_classes.clear()
        random_state = 12883823
        classes_index = 0
        
        self.split = int(self.btn_KFOLD_split.text())
        rkf = KFold(n_splits=self.split, random_state=random_state, shuffle=True)
        for train, test in rkf.split(self.dataset):
            classes_index += 1
            
            self.cmb_KFOLD_classes.addItem(str(classes_index)+". Classes")
            #print("kFold--->",self.X)
            #print("kFold--->",self.Y)
            X_train, X_test = self.X[train], self.X[test]
            y_train, y_test = self.Y[train], self.Y[test]

            self.classes_X_Train.append(X_train)
            self.classes_X_Test.append(X_test)
            self.classes_Y_Train.append(y_train)
            self.classes_Y_Test.append(y_test)


    def btn_Apply(self):
        rate = self.split
        column = rate+2
        self.count_detail=0
        self.table_details.clear();
        self.table_details.setColumnCount(column)
        self.table_details.setRowCount(8)
        
        ortalama=[]
        for i in range(0,rate):
            temp_ortalama = self.ALGORITHM_RBF(self.classes_X_Train[i],self.classes_X_Test[i],self.classes_Y_Train[i],self.classes_Y_Test[i])
            ortalama.append(temp_ortalama)
        self.Detail_Add("RDB",ortalama,column)
            
        ortalama=[]
        for i in range(0,rate):
            temp_ortalama = self.ALGORITHM_LOGIC(self.classes_X_Train[i],self.classes_X_Test[i],self.classes_Y_Train[i],self.classes_Y_Test[i])
            ortalama.append(temp_ortalama) 
        self.Detail_Add("LOGIC",ortalama,column)
        
        ortalama=[]
        for i in range(0,rate):
            temp_ortalama = self.ALGORITHM_LINEAR(self.classes_X_Train[i],self.classes_X_Test[i],self.classes_Y_Train[i],self.classes_Y_Test[i])
            ortalama.append(temp_ortalama)
        self.Detail_Add("LINEAR",ortalama,column)
        
        ortalama=[]
        for i in range(0,rate):
            temp_ortalama = self.ALGORITHM_POLY(self.classes_X_Train[i],self.classes_X_Test[i],self.classes_Y_Train[i],self.classes_Y_Test[i])
            ortalama.append(temp_ortalama)
        self.Detail_Add("POLY",ortalama,column)
        
        ortalama=[]
        for i in range(0,rate):
            temp_ortalama = self.ALGORITHM_SIGMOID(self.classes_X_Train[i],self.classes_X_Test[i],self.classes_Y_Train[i],self.classes_Y_Test[i])
            ortalama.append(temp_ortalama)
        self.Detail_Add("SIGMOID",ortalama,column)
        
        ortalama=[]
        for i in range(0,rate):
            temp_ortalama = self.ALGORITHM_RANDOM_FOREST(self.classes_X_Train[i],self.classes_X_Test[i],self.classes_Y_Train[i],self.classes_Y_Test[i])
            ortalama.append(temp_ortalama)
        self.Detail_Add("RANDOM FOREST",ortalama,column)
        
        ortalama=[]
        for i in range(0,rate):
            temp_ortalama = self.ALGORITHM_KNEIGHBORS(self.classes_X_Train[i],self.classes_X_Test[i],self.classes_Y_Train[i],self.classes_Y_Test[i])
            ortalama.append(temp_ortalama)
        self.Detail_Add("KNEIGHBORS",ortalama,column)
        
        ortalama=[]
        for i in range(0,rate):
            temp_ortalama = self.ALGORITHM_GAUSSIAN(self.classes_X_Train[i],self.classes_X_Test[i],self.classes_Y_Train[i],self.classes_Y_Test[i])
            ortalama.append(temp_ortalama)
        self.Detail_Add("GAUSSIAN",ortalama,column)
        
        self.table_details.horizontalHeader().setStretchLastSection(True)
        self.table_details.resizeColumnsToContents()
        self.table_details.setHorizontalHeaderLabels(self.header_labels_details)
    
    count_detail = 0
    header_labels_details = []
    def Detail_Add(self,algoritma,ortalamalar,rate):
        self.table_details.setItem(self.count_detail,0, QTableWidgetItem(str(algoritma)))
        self.header_labels_details.clear()
        self.header_labels_details.append('Algorithm')
        count_detail_column=1
        ortalama_toplam = 0
        for i,value in enumerate(ortalamalar):
            ortalama_toplam += float(value)
            self.table_details.setItem(self.count_detail,count_detail_column, QTableWidgetItem(str(value)))
            self.header_labels_details.append(str(i+1)+'. Ortalama')
            count_detail_column+=1
        
        self.table_details.setItem(self.count_detail,count_detail_column, QTableWidgetItem(str(ortalama_toplam/(rate-2))))
        self.header_labels_details.append('Genel Ortalama')
        self.count_detail+=1

    #algoritmalar
    def ALGORITHM_RBF(self,_x_train,_x_test,_y_train,_y_test):
        sc = StandardScaler()
        X_train = sc.fit_transform(_x_train)
        X_test = sc.transform(_x_test)
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(X_train, _y_train)
        y_pred = classifier.predict(X_test)
        cm  = confusion_matrix(_y_test, y_pred)
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = _y_train, cv = self.split)
        return str(round(accuracies.mean()*100,2))
        
    def ALGORITHM_LOGIC(self,_x_train,_x_test,_y_train,_y_test):
        from sklearn.linear_model import LogisticRegression
        LogReg = LogisticRegression()
        LogReg.fit(_x_train, _y_train)
        y_pred = LogReg.predict(_x_test)
        cm  = confusion_matrix(_y_test, y_pred)
        accuracy = accuracy_score(_y_test, y_pred)
        return str(round(accuracy*100,2))
        
    def ALGORITHM_POLY(self,_x_train,_x_test,_y_train,_y_test):
        sc = StandardScaler()
        X_train = sc.fit_transform(_x_train)
        X_test = sc.transform(_x_test)
        classifier = SVC(kernel = 'poly', random_state = 0)
        classifier.fit(X_train, _y_train)
        y_pred = classifier.predict(X_test)
        cm  = confusion_matrix(_y_test, y_pred)
        #self.lbl_kFold_mean.setText(str(cm))
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = _y_train, cv = self.split)
        return str(round(accuracies.mean()*100,2))
    
    def ALGORITHM_LINEAR(self,_x_train,_x_test,_y_train,_y_test):
        sc = StandardScaler()
        X_train = sc.fit_transform(_x_train)
        X_test = sc.transform(_x_test)
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(X_train, _y_train)
        y_pred = classifier.predict(X_test)
        cm  = confusion_matrix(_y_test, y_pred)
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = _y_train, cv = self.split)
        return str(round(accuracies.mean()*100,2))
    
    def ALGORITHM_SIGMOID(self,_x_train,_x_test,_y_train,_y_test):
        sc = StandardScaler()
        X_train = sc.fit_transform(_x_train)
        X_test = sc.transform(_x_test)
        classifier = SVC(kernel = 'sigmoid', random_state = 0)
        classifier.fit(X_train, _y_train)
        y_pred = classifier.predict(X_test)
        cm  = confusion_matrix(_y_test, y_pred)
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = _y_train, cv = self.split)
        return str(round(accuracies.mean()*100,2))

    def ALGORITHM_RANDOM_FOREST(self,_x_train,_x_test,_y_train,_y_test):
        model = RandomForestClassifier()
        model.fit(_x_train,_y_train)
        y_pred = model.predict(_x_test)
        accuracy = accuracy_score(_y_test,y_pred)
        return str(round(accuracy*100,2))
    
    def ALGORITHM_KNEIGHBORS(self,_x_train,_x_test,_y_train,_y_test):
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(_x_train,_y_train)
        y_pred = model.predict(_x_test)
        accuracy = accuracy_score(_y_test,y_pred)
        return str(round(accuracy*100,2))
    
    def ALGORITHM_GAUSSIAN(self,_x_train,_x_test,_y_train,_y_test):
        model = GaussianNB()
        model.fit(_x_train,_y_train)
        y_pred = model.predict(_x_test)
        accuracy = accuracy_score(_y_test,y_pred)
        return str(round(accuracy*100,2))
    
    #metodlar
    def get_CSV_READ(self,filename):
        with open(filename, "r") as fileInput:
            for row in csv.reader(fileInput):    
                items = [
                    QtGui.QStandardItem(field)
                    for field in row
                ]
                self.model.appendRow(items)
                
                
    def get_LIST_READ(self,table):
        for row in table:
            items = [
                QtGui.QStandardItem(field)
                for field in row
            ]
            self.model_x_train.appendRow(items)
    
    def set_sum(self,a,b):        
        return np.concatenate((a, b))
        
    
    def showdialog(self,window_title,title,content):
       msg = QtWidgets.QMessageBox()
       msg.setIcon(QtWidgets.QMessageBox.Information)
    
       msg.setText(title)
       msg.setInformativeText(content)
       msg.setWindowTitle(window_title)
       msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
       msg.exec_()