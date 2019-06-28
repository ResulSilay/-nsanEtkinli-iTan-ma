# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design2.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 655)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.table_details = QtWidgets.QTableWidget(self.centralwidget)
        self.table_details.setGeometry(QtCore.QRect(30, 263, 741, 101))
        self.table_details.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.table_details.setObjectName("table_details")
        self.table_details.setColumnCount(0)
        self.table_details.setRowCount(0)
        self.label_31 = QtWidgets.QLabel(self.centralwidget)
        self.label_31.setGeometry(QtCore.QRect(30, 240, 151, 16))
        self.label_31.setStyleSheet("color:red;")
        self.label_31.setObjectName("label_31")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 60, 61, 16))
        self.label_2.setObjectName("label_2")
        self.table_dataset_load = QtWidgets.QTableWidget(self.centralwidget)
        self.table_dataset_load.setGeometry(QtCore.QRect(30, 90, 741, 81))
        self.table_dataset_load.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.table_dataset_load.setObjectName("table_dataset_load")
        self.table_dataset_load.setColumnCount(0)
        self.table_dataset_load.setRowCount(0)
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(30, 410, 47, 13))
        self.label_12.setObjectName("label_12")
        self.list_x_test = QtWidgets.QTableWidget(self.centralwidget)
        self.list_x_test.setGeometry(QtCore.QRect(30, 520, 471, 71))
        self.list_x_test.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list_x_test.setObjectName("list_x_test")
        self.list_x_test.setColumnCount(0)
        self.list_x_test.setRowCount(0)
        self.list_x_train = QtWidgets.QTableWidget(self.centralwidget)
        self.list_x_train.setGeometry(QtCore.QRect(30, 430, 471, 61))
        self.list_x_train.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list_x_train.setObjectName("list_x_train")
        self.list_x_train.setColumnCount(0)
        self.list_x_train.setRowCount(0)
        self.label_33 = QtWidgets.QLabel(self.centralwidget)
        self.label_33.setGeometry(QtCore.QRect(30, 383, 151, 16))
        self.label_33.setStyleSheet("color:red;")
        self.label_33.setObjectName("label_33")
        self.list_y_test = QtWidgets.QTableWidget(self.centralwidget)
        self.list_y_test.setGeometry(QtCore.QRect(530, 520, 241, 71))
        self.list_y_test.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list_y_test.setObjectName("list_y_test")
        self.list_y_test.setColumnCount(0)
        self.list_y_test.setRowCount(0)
        self.list_y_train = QtWidgets.QTableWidget(self.centralwidget)
        self.list_y_train.setGeometry(QtCore.QRect(530, 430, 241, 61))
        self.list_y_train.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list_y_train.setObjectName("list_y_train")
        self.list_y_train.setColumnCount(0)
        self.list_y_train.setRowCount(0)
        self.label_26 = QtWidgets.QLabel(self.centralwidget)
        self.label_26.setGeometry(QtCore.QRect(530, 410, 47, 13))
        self.label_26.setObjectName("label_26")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(30, 500, 47, 13))
        self.label_10.setObjectName("label_10")
        self.label_23 = QtWidgets.QLabel(self.centralwidget)
        self.label_23.setGeometry(QtCore.QRect(530, 500, 47, 13))
        self.label_23.setObjectName("label_23")
        self.btn_PCA = QtWidgets.QPushButton(self.centralwidget)
        self.btn_PCA.setGeometry(QtCore.QRect(440, 180, 61, 31))
        self.btn_PCA.setStyleSheet("QPushButton\n"
"{\n"
"border: 1px solid red;\n"
"color:red;\n"
"\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"   background-color:rgb(250, 250, 250);\n"
"}")
        self.btn_PCA.setObjectName("btn_PCA")
        self.btn_KFOLD = QtWidgets.QPushButton(self.centralwidget)
        self.btn_KFOLD.setGeometry(QtCore.QRect(560, 180, 81, 31))
        self.btn_KFOLD.setStyleSheet("QPushButton\n"
"{\n"
"border: 1px solid red;\n"
"color:red;\n"
"\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"   background-color:rgb(250, 250, 250);\n"
"}")
        self.btn_KFOLD.setObjectName("btn_KFOLD")
        self.label_36 = QtWidgets.QLabel(self.centralwidget)
        self.label_36.setGeometry(QtCore.QRect(530, 610, 81, 31))
        self.label_36.setObjectName("label_36")
        self.cmb_KFOLD_classes = QtWidgets.QComboBox(self.centralwidget)
        self.cmb_KFOLD_classes.setGeometry(QtCore.QRect(630, 610, 141, 31))
        self.cmb_KFOLD_classes.setStyleSheet("border:1px solid black;")
        self.cmb_KFOLD_classes.setObjectName("cmb_KFOLD_classes")
        self.btn_KFOLD_split = QtWidgets.QLineEdit(self.centralwidget)
        self.btn_KFOLD_split.setGeometry(QtCore.QRect(520, 180, 41, 31))
        self.btn_KFOLD_split.setStyleSheet("padding-left:3px;\n"
"border:1px solid red;")
        self.btn_KFOLD_split.setObjectName("btn_KFOLD_split")
        self.btn_apply = QtWidgets.QPushButton(self.centralwidget)
        self.btn_apply.setGeometry(QtCore.QRect(660, 180, 111, 31))
        self.btn_apply.setStyleSheet("QPushButton\n"
"{\n"
"border: 1px solid red;\n"
"color:red;\n"
"\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"   background-color:rgb(250, 250, 250);\n"
"}")
        self.btn_apply.setObjectName("btn_apply")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(30, 20, 371, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.btn_dataset_import = QtWidgets.QPushButton(self.centralwidget)
        self.btn_dataset_import.setGeometry(QtCore.QRect(30, 180, 81, 31))
        self.btn_dataset_import.setStyleSheet("QPushButton\n"
"{\n"
"border: 1px solid black;\n"
"color:black;\n"
"\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"   background-color:rgb(250, 250, 250);\n"
"}")
        self.btn_dataset_import.setObjectName("btn_dataset_import")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_31.setText(_translate("MainWindow", "D e t a i l s"))
        self.label_2.setText(_translate("MainWindow", "D a t a s e t"))
        self.label_12.setText(_translate("MainWindow", "X  T r a i n"))
        self.label_33.setText(_translate("MainWindow", "D a t a  S e t"))
        self.label_26.setText(_translate("MainWindow", "Y  T r a i n"))
        self.label_10.setText(_translate("MainWindow", "X  T e s t"))
        self.label_23.setText(_translate("MainWindow", "Y  T e s t"))
        self.btn_PCA.setText(_translate("MainWindow", "P C A"))
        self.btn_KFOLD.setText(_translate("MainWindow", "K F O L D"))
        self.label_36.setText(_translate("MainWindow", "Split Datasets :"))
        self.btn_KFOLD_split.setText(_translate("MainWindow", "10"))
        self.btn_apply.setText(_translate("MainWindow", "A P P L Y"))
        self.label_3.setText(_translate("MainWindow", "İ N S A N  E T K İ N L İ Ğ İ  T A N I M A"))
        self.btn_dataset_import.setText(_translate("MainWindow", "I M P O R T"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

