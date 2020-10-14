# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\git\easy_heatmap_creater\gui_template.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import cv2
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(40, 40, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(140, 40, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(250, 40, 75, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(350, 40, 75, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 230, 200, 200))
        self.label.setObjectName("label")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(480, 10, 256, 192))
        self.graphicsView.setObjectName("graphicsView")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1045, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)


        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "PushButton"))
        self.pushButton_2.setText(_translate("MainWindow", "PushButton"))
        self.pushButton_3.setText(_translate("MainWindow", "PushButton"))
        self.pushButton_4.setText(_translate("MainWindow", "PushButton"))
        self.pushButton.clicked.connect(self.btn1_clicked)

        
        self.label.mousePressEvent = self.getPixel
        image = cv2.imread(r'D:\borders.jpg')
        self.kmeans = kmeans_color_quantization(image, clusters=2)
        mypixmap = self.cv2_to_pix(image)
        self.display_pix(mypixmap)
        

    def btn1_clicked(self):
        print("btnnn")
        return 1
    def getPixel(self, event):
        x = event.pos().x()
        y = event.pos().y()
        #c = self.img.pixel(x,y)  # color code (integer): 3235912
        # depending on what kind of value you like (arbitary examples)
        #c_qobj = QtGui.QColor(c)  # color object
        #c_rgb = QtGui.QColor(c).getRgb()  # 8bit RGBA: (255, 23, 0, 255)
        #c_rgbf = QtGui.QColor(c).getRgbf()  # RGBA float: (1.0, 0.3123, 0.0, 1.0)
        print(x,y)
        #result = self.kmeans.copy()
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
        self.label.setFixedWidth(pix.width())
        self.label.setFixedHeight(pix.height())
        self.label.setPixmap(pix)
        return 1

    

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

