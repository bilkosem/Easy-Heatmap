# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\git\easy_heatmap_creater\gui_template.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(653, 865)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.hm_label = QtWidgets.QLabel(self.centralwidget)
        self.hm_label.setGeometry(QtCore.QRect(20, 240, 47, 13))
        self.hm_label.setObjectName("hm_label")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(20, 20, 621, 201))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_field = QtWidgets.QWidget()
        self.tab_field.setObjectName("tab_field")
        self.btn_save_field = QtWidgets.QPushButton(self.tab_field)
        self.btn_save_field.setGeometry(QtCore.QRect(40, 90, 221, 31))
        self.btn_save_field.setObjectName("btn_save_field")
        self.field_name_input = QtWidgets.QTextEdit(self.tab_field)
        self.field_name_input.setGeometry(QtCore.QRect(110, 60, 151, 21))
        self.field_name_input.setObjectName("field_name_input")
        self.label_2 = QtWidgets.QLabel(self.tab_field)
        self.label_2.setGeometry(QtCore.QRect(40, 60, 71, 21))
        self.label_2.setObjectName("label_2")
        self.cmb_field = QtWidgets.QComboBox(self.tab_field)
        self.cmb_field.setGeometry(QtCore.QRect(420, 20, 151, 21))
        self.cmb_field.setObjectName("cmb_field")
        self.btn_del_sel_field = QtWidgets.QPushButton(self.tab_field)
        self.btn_del_sel_field.setGeometry(QtCore.QRect(350, 50, 221, 31))
        self.btn_del_sel_field.setObjectName("btn_del_sel_field")
        self.label_3 = QtWidgets.QLabel(self.tab_field)
        self.label_3.setGeometry(QtCore.QRect(350, 20, 71, 21))
        self.label_3.setObjectName("label_3")
        self.line = QtWidgets.QFrame(self.tab_field)
        self.line.setGeometry(QtCore.QRect(300, 10, 20, 151))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.btn_del_all_field = QtWidgets.QPushButton(self.tab_field)
        self.btn_del_all_field.setGeometry(QtCore.QRect(350, 90, 221, 31))
        self.btn_del_all_field.setObjectName("btn_del_all_field")
        self.btn_upload_image = QtWidgets.QPushButton(self.tab_field)
        self.btn_upload_image.setGeometry(QtCore.QRect(40, 20, 221, 31))
        self.btn_upload_image.setObjectName("btn_upload_image")
        self.tabWidget.addTab(self.tab_field, "")
        self.tab_point = QtWidgets.QWidget()
        self.tab_point.setObjectName("tab_point")
        self.label_4 = QtWidgets.QLabel(self.tab_point)
        self.label_4.setGeometry(QtCore.QRect(40, 90, 71, 16))
        self.label_4.setObjectName("label_4")
        self.cmb_sel_point = QtWidgets.QComboBox(self.tab_point)
        self.cmb_sel_point.setGeometry(QtCore.QRect(130, 90, 131, 21))
        self.cmb_sel_point.setObjectName("cmb_sel_point")
        self.line_2 = QtWidgets.QFrame(self.tab_point)
        self.line_2.setGeometry(QtCore.QRect(300, 10, 20, 151))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.cmb_del_point = QtWidgets.QComboBox(self.tab_point)
        self.cmb_del_point.setGeometry(QtCore.QRect(430, 20, 141, 21))
        self.cmb_del_point.setObjectName("cmb_del_point")
        self.label_5 = QtWidgets.QLabel(self.tab_point)
        self.label_5.setGeometry(QtCore.QRect(350, 20, 81, 21))
        self.label_5.setObjectName("label_5")
        self.btn_del_sel_p = QtWidgets.QPushButton(self.tab_point)
        self.btn_del_sel_p.setGeometry(QtCore.QRect(350, 50, 221, 31))
        self.btn_del_sel_p.setObjectName("btn_del_sel_p")
        self.btn_del_all_p = QtWidgets.QPushButton(self.tab_point)
        self.btn_del_all_p.setGeometry(QtCore.QRect(350, 90, 221, 31))
        self.btn_del_all_p.setObjectName("btn_del_all_p")
        self.btn_upload_data = QtWidgets.QPushButton(self.tab_point)
        self.btn_upload_data.setGeometry(QtCore.QRect(40, 20, 221, 31))
        self.btn_upload_data.setObjectName("btn_upload_data")
        self.tabWidget.addTab(self.tab_point, "")
        self.tab_heatmap = QtWidgets.QWidget()
        self.tab_heatmap.setObjectName("tab_heatmap")
        self.slider = QtWidgets.QSlider(self.tab_heatmap)
        self.slider.setGeometry(QtCore.QRect(40, 130, 241, 21))
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setObjectName("slider")
        self.cmb_paint_field = QtWidgets.QComboBox(self.tab_heatmap)
        self.cmb_paint_field.setGeometry(QtCore.QRect(130, 30, 151, 22))
        self.cmb_paint_field.setObjectName("cmb_paint_field")
        self.btn_save_hm = QtWidgets.QPushButton(self.tab_heatmap)
        self.btn_save_hm.setGeometry(QtCore.QRect(330, 30, 121, 31))
        self.btn_save_hm.setObjectName("btn_save_hm")
        self.label_6 = QtWidgets.QLabel(self.tab_heatmap)
        self.label_6.setGeometry(QtCore.QRect(40, 30, 61, 21))
        self.label_6.setObjectName("label_6")
        self.btn_clr_hm = QtWidgets.QPushButton(self.tab_heatmap)
        self.btn_clr_hm.setGeometry(QtCore.QRect(460, 30, 121, 31))
        self.btn_clr_hm.setObjectName("btn_clr_hm")
        self.line_3 = QtWidgets.QFrame(self.tab_heatmap)
        self.line_3.setGeometry(QtCore.QRect(300, 10, 20, 151))
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.label_7 = QtWidgets.QLabel(self.tab_heatmap)
        self.label_7.setGeometry(QtCore.QRect(40, 110, 91, 16))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.tab_heatmap)
        self.label_8.setGeometry(QtCore.QRect(40, 70, 91, 16))
        self.label_8.setObjectName("label_8")
        self.cmb_sel_cm = QtWidgets.QComboBox(self.tab_heatmap)
        self.cmb_sel_cm.setGeometry(QtCore.QRect(130, 70, 151, 21))
        self.cmb_sel_cm.setObjectName("cmb_sel_cm")
        self.btn_save_img = QtWidgets.QPushButton(self.tab_heatmap)
        self.btn_save_img.setGeometry(QtCore.QRect(330, 90, 251, 51))
        self.btn_save_img.setObjectName("btn_save_img")
        self.tabWidget.addTab(self.tab_heatmap, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 653, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.hm_label.setText(_translate("MainWindow", "hm_label"))
        self.btn_save_field.setText(_translate("MainWindow", "Save Field"))
        self.label_2.setText(_translate("MainWindow", "Field Name:"))
        self.btn_del_sel_field.setText(_translate("MainWindow", "Delete Selected Field"))
        self.label_3.setText(_translate("MainWindow", "Fields:"))
        self.btn_del_all_field.setText(_translate("MainWindow", "Delete All Fields"))
        self.btn_upload_image.setText(_translate("MainWindow", "Upload Image File"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_field), _translate("MainWindow", "Create Field"))
        self.label_4.setText(_translate("MainWindow", "Choose Point:"))
        self.label_5.setText(_translate("MainWindow", "Points On Map:"))
        self.btn_del_sel_p.setText(_translate("MainWindow", "Delete Selected Point"))
        self.btn_del_all_p.setText(_translate("MainWindow", "Delete All Points"))
        self.btn_upload_data.setText(_translate("MainWindow", "Upload Data File"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_point), _translate("MainWindow", "Create Point"))
        self.btn_save_hm.setText(_translate("MainWindow", "Save Heatmap Field"))
        self.label_6.setText(_translate("MainWindow", "Select Field:"))
        self.btn_clr_hm.setText(_translate("MainWindow", "Clear Field"))
        self.label_7.setText(_translate("MainWindow", "Adjust Radius:"))
        self.label_8.setText(_translate("MainWindow", "Select Colormap:"))
        self.btn_save_img.setText(_translate("MainWindow", "Save Image"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_heatmap), _translate("MainWindow", "Paint Heatmap"))

import res_file_rc

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

