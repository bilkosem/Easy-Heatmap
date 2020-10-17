
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage import transform
from scipy import ndimage
from matplotlib import cm

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
        self.tabWidget.setCurrentIndex(0)
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

    def configure_gui(self,MainWindow):
        self.slider.setRange(0,200)
        self.slider.setValue(5)
        
        self.hm_label.mousePressEvent = self.getPixel
        self.tabWidget.currentChanged.connect(self.on_tab_change)
        
        self.btn_upload_image.clicked.connect(self.btn_upload_image_clicked)
        self.btn_upload_data.clicked.connect(self.btn_upload_data_cliked)
        self.btn_save_field.clicked.connect(self.btn_save_field_clicked)
        self.btn_del_sel_field.clicked.connect(self.btn_del_sel_field_clicked)
        self.btn_del_all_field.clicked.connect(self.btn_del_all_field_clicked)
        self.btn_del_sel_p.clicked.connect(self.btn_del_sel_p_clicked)
        self.btn_del_all_p.clicked.connect(self.btn_del_all_p_clicked)
        
        self.slider.valueChanged.connect(self.slider_value_changed)
        self.btn_save_hm.clicked.connect(self.btn_save_hm_clicked)
        self.cmb_sel_cm.addItems(plt.colormaps())
        self.field_list=[]
        

        
    def on_tab_change(self,tab_index):
        if tab_index==2:
            self.cmb_paint_field.addItems([f['label'] for f in self.field_list])
            self.kmeans = self.kmeans_base.copy()
            pixmap = self.cv2_to_pix(self.kmeans)
            self.display_pix(pixmap) 

    def slider_value_changed(self,tab_index):
        
        #self.create_heatmap()
        pass
    def btn_upload_image_clicked(self):
        #filepath = QtWidgets.QFileDialog.getOpenFileName()[0]
        filepath="D:/borders.jpg"
        print(filepath)
        self.image = cv2.imread(filepath)
        self.kmeans_base = kmeans_color_quantization(self.image, clusters=3)
        self.kmeans = self.kmeans_base.copy()
        mypixmap = self.cv2_to_pix(self.image)
        #MainWindow.resize(240+mypixmap.height(), mypixmap.width())
        print(mypixmap.height())
        MainWindow.setGeometry(100,100,47+mypixmap.width(),240+mypixmap.height(),)
        self.btn_del_all_field_clicked() #Clear Fields and Map Data
        
    def btn_upload_data_cliked(self):
        self.btn_del_all_p_clicked(False)

        #filepath = QtWidgets.QFileDialog.getOpenFileName()[0]
        filepath="C:/Users/bilko/Desktop/hm.xlsx"
        print(filepath)
        self.df = pd.read_excel(filepath)
        self.p_labels = self.df['label'].values.tolist()
        self.cmb_sel_point.addItems(self.p_labels)
        print(len(self.df))
        
    def btn_save_field_clicked(self):
        
        hsv = cv2.cvtColor(self.kmeans, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255)) 
        #field = self.kmeans[np.all(self.kmeans == (36, 255, 12), axis=-1)]#cv2.inRange(hsv, (36, 25, 25), (70, 255,255)) 
        field_label = str(self.field_name_input.toPlainText())
        self.field_list.append({'label':field_label, 'mask':mask, 'points':[]})
        self.cmb_field.addItem(field_label)

        #Clear image and print fields by masks
        self.display_fields()        

    def btn_del_sel_field_clicked(self):
        self.field_list.pop(self.cmb_field.findText(self.cmb_field.currentText()))
        self.cmb_field.removeItem(self.cmb_field.findText(self.cmb_field.currentText()))
        self.display_fields()

    def btn_del_all_field_clicked(self):
        self.field_list.clear()
        self.cmb_field.clear()
        self.display_fields()

    def btn_del_all_p_clicked(self,reupload_labels=True):   
        for f_idx,points_of_field in enumerate([f['points'] for f in self.field_list]):
            points_of_field.clear()
        self.cmb_del_point.clear() #Delete all points from comboBox
        self.cmb_sel_point.clear()
        if reupload_labels:
            self.cmb_sel_point.addItems(self.p_labels) #Upload All points to comboBox
        self.display_fields(True)

    def btn_del_sel_p_clicked(self):
        print("------------DELETİNG POİNT")
        label2del = str(self.cmb_del_point.currentText())
        print(label2del)
        for f_idx,points_of_field in enumerate([f['points'] for f in self.field_list]):
            for p_idx,point in enumerate(points_of_field):
                if label2del in point.keys():
                    print(p_idx)
                    self.field_list[f_idx]['points'].pop(p_idx)
        self.cmb_del_point.removeItem(self.cmb_del_point.findText(label2del))
        self.cmb_sel_point.addItem(label2del)
        self.display_fields(True)
        self.display_points(False)     
        
    def getPixel(self, event):
        x = event.pos().x()
        y = event.pos().y()
     
        print(x,y)
        if self.tabWidget.currentIndex() == 0: # FIELD TAB
            for mask in [x['mask'] for x in self.field_list]:
                if mask[y,x]==255:
                    return
                
            seed_point = (x, y)
            cv2.floodFill(self.kmeans, None, seedPoint=seed_point, newVal=(36, 255, 12), loDiff=(0, 0, 0, 0), upDiff=(0, 0, 0, 0))
            pixmap = self.cv2_to_pix(self.kmeans)
            self.display_pix(pixmap)
        elif self.tabWidget.currentIndex() == 1: # POINT TAB
            
            if self.cmb_sel_point.count()>0:
                for idx,mask in enumerate([x['mask'] for x in self.field_list]):
                    if mask[y,x]==255:
                        print("IN FIELD")
                        self.field_list[idx]['points'].append({str(self.cmb_sel_point.currentText()):str(self.cmb_sel_point.currentText()),
                                                               'coord':(x,y),
                                                               'value':float(self.df[self.df['label']==str(self.cmb_sel_point.currentText())]['value'])})
                        self.cmb_del_point.addItem(str(self.cmb_sel_point.currentText()))
                        self.cmb_sel_point.removeItem(self.cmb_sel_point.findText(self.cmb_sel_point.currentText()))
                        self.display_points(False)
                        return
                    else:
                        print("OUT FIELD")
        return 
###############################################################################
        # TOOLS
    def update_cmb_field(self):
        self.cmb_field.addItems([f['label'] for f in self.field_list])
        return 1    

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

    def display_fields(self,clear=True):
        print(self.field_list)
        if clear:
            self.kmeans = self.kmeans_base.copy()
        for mask in [f['mask'] for f in self.field_list]:
            self.kmeans[mask==255]=[169,169,169]
        pixmap = self.cv2_to_pix(self.kmeans)
        self.display_pix(pixmap) 

    def display_points(self,clear=True):
        if clear:
            self.kmeans = self.kmeans_base.copy()
        #print([f['points'] for f in self.field_list])
        
        for points_of_field in [f['points'] for f in self.field_list]:
            for point in points_of_field:
                #cv2.circle(self.kmeans, (400,400), 10,(255,0,0))
                self.kmeans = cv2.circle(self.kmeans,point['coord'], 10, (255,0,0), -1)
        
        pixmap = self.cv2_to_pix(self.kmeans)
        self.display_pix(pixmap)         

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
###############################################################################
### Ploting Tools
    def btn_save_hm_clicked(self):
        self.create_heatmap()

    def create_heatmap(self):
                
        # Create Heatmap Parameters
        mask = [f['mask'] for f in self.field_list if str(self.cmb_paint_field.currentText())==f['label']][0]
        colormap = str(self.cmb_sel_cm.currentText())
        point_list = [f['points'] for f in self.field_list if str(self.cmb_paint_field.currentText())==f['label']][0]
        values = [p['value'] for p in point_list]
        slider_value = self.slider.value()
        
        # Create Heatmap Surface
        x = np.ones((ui.kmeans.shape[0],ui.kmeans.shape[1]))*min(values) #(570, 900)
        for p in point_list:
            x[p['coord'][1],p['coord'][0]]=p['value']
        
        # Interpolate Heatmap
        gaussian_map = ndimage.filters.gaussian_filter(x, sigma=slider_value)

        max_value = np.max(gaussian_map)
        min_value = np.min(gaussian_map)        
        normalized_heat_map = (gaussian_map - min_value) / (max_value-min_value)        
        
        # Apply Mask to Heatmap
        #cmap = plt.get_cmap('jet')
        im_new = Image.fromarray(np.uint8(cm.jet(normalized_heat_map)*255))
        opencvImage = cv2.cvtColor(np.array(im_new), cv2.COLOR_RGB2BGR)
        opencvImage_MATPLOT = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2RGB)
        self.kmeans[mask==255]=opencvImage_MATPLOT[mask==255]
        
        # Display Mask to Heatmap
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

