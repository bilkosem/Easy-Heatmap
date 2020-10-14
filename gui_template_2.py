
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import pandas as pd

variable=0

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
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        #self.hm_label.setText(_translate("MainWindow", "hm_label"))
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
        self.configure_gui(MainWindow)
       
    def btn_ui_cliked(self):
        filepath = QtWidgets.QFileDialog.getOpenFileName()[0]
        #filepath="D:/borders.jpg"
        print(filepath)
        self.image = cv2.imread(filepath)
        self.kmeans = kmeans_color_quantization(self.image, clusters=3)
        mypixmap = self.cv2_to_pix(self.image)
        #MainWindow.resize(240+mypixmap.height(), mypixmap.width())
        print(mypixmap.height())
        MainWindow.setGeometry(100,100,47+mypixmap.width(),240+mypixmap.height(),)
        self.display_pix(mypixmap)
        
    def btn_ud_cliked(self):
        filepath = QtWidgets.QFileDialog.getOpenFileName()[0]
        print(filepath)
        df = pd.read_excel(filepath)
        self.p_labels = df['label'].values.tolist()
        self.cmb_sel_point.addItems(self.p_labels)
        
    def configure_gui(self,MainWindow):
        self.hm_label.mousePressEvent = self.getPixel
        self.btn_upload_image.clicked.connect(self.btn_ui_cliked)
        self.btn_upload_data.clicked.connect(self.btn_ud_cliked)
        self.btn_save_field.clicked.connect(self.btn_sf_clicked)
        
        self.field_list=[]

        
    def getPixel(self, event):
        x = event.pos().x()
        y = event.pos().y()

        print(x,y)
        seed_point = (x, y)
        cv2.floodFill(self.kmeans, None, seedPoint=seed_point, newVal=(36, 255, 12), loDiff=(0, 0, 0, 0), upDiff=(0, 0, 0, 0))
        pixmap = self.cv2_to_pix(self.kmeans)
        self.display_pix(pixmap)
        #return x, y, c_rgb

    def cv2_to_pix(self,cv2_obj):
        height, width, channel = cv2_obj.shape
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(cv2_obj.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(QtGui.QPixmap.fromImage(qImg))
        return pixmap

    def display_pix(self,pix):
        self.hm_label.setFixedWidth(pix.width())
        self.hm_label.setFixedHeight(pix.height())
        self.hm_label.setPixmap(pix)
        
    def btn_sf_clicked(self):
        
        hsv = cv2.cvtColor(self.kmeans, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255)) 
        #print(mask)
        mask = self.kmeans[np.all(self.kmeans == (36, 255, 12), axis=-1)]#cv2.inRange(hsv, (36, 25, 25), (70, 255,255)) 
        self.field_list.append(mask)
        #print(mask)
        global variable
        variable = self.field_list
        #self.kmeans[mask] = (255,255,0)
        #self.kmeans[np.all(self.kmeans == (36, 255, 12), axis=-1)] = (169,169,169)
        #pixmap = self.cv2_to_pix(self.kmeans)
        #self.display_pix(pixmap)
        #self.kmeans[green_mask]=[0,0,0]
        #print(mask1)
        #for f in self.field_list:
        self.kmeans[np.all(self.kmeans == (36, 255, 12), axis=-1)] = (169,169,169)
        #res = cv2.bitwise_and(self.kmeans,self.kmeans, mask= mask)
        pixmap = self.cv2_to_pix(self.kmeans)
        self.display_pix(pixmap) 


def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))
    
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

