from PyQt5 import QtWidgets,QtGui,QtCore
from PyQt5.QtCore import Qt
from skimage.util import random_noise

import numpy as np
import sys
import matplotlib
import cv2
import torch


class MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__()

        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.confidence_threshold = 0.4
        self.all_colors = [[int(_c*255) for _c in cc] 
                           for cc in matplotlib.colors.BASE_COLORS.values()]
        self.check_clear = False
        self.main_screenshot = None
        self.noisy_screenshot = None
        self.noise_type = 'no_noise'



        self.resize(QtWidgets.QApplication.screens()[0].size())
        self.setWindowTitle('Yolo and Noise')
        
        pixmap01 = QtGui.QPixmap(self.size())
        pixmap01.fill(Qt.transparent)
        self.label_img = MyLabel(self) 
        self.label_img.setPixmap(pixmap01)
        self.label_img.adjustSize()
        self.label_img.clicked.connect(self.on_click_clear)
        self.prev_pixmap = self.label_img.pixmap().copy()
        
        self.btn_new_screenshot = QtWidgets.QPushButton('Apply Yolo')
        self.btn_new_screenshot.setIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, 'SP_DialogApplyButton')))        
        self.btn_new_screenshot.setFixedSize(90, 30)
        self.btn_new_screenshot.clicked.connect(self.apply_yolo)
        
        self.btn_close = QtWidgets.QPushButton('Close App') 
        self.btn_close.setIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, 'SP_DialogCloseButton')))        
        self.btn_close.setFixedSize(90, 30)
        self.btn_close.clicked.connect(self.close_app)
        
        self.clear_button = QtWidgets.QPushButton('Clear')
        self.clear_button.setIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, 'SP_DialogResetButton')))
        self.clear_button.setFixedSize(90,30)
        self.clear_button.clicked.connect(self.on_click_clear)        


        self.thr_lbl = QtWidgets.QLabel('Confidence:0.40')
        self.sl = QtWidgets.QSlider(Qt.Horizontal)
        self.sl.setMinimum(10)
        self.sl.setMaximum(80)
        self.sl.setValue(int(self.confidence_threshold*100))
        self.sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sl.setSingleStep(5)
        self.sl.setTickInterval(5)
        self.sl.valueChanged.connect(self.slider_value_change)


        self.bk_lbl = QtWidgets.QLabel(self)
        aw = 100
        ah = QtWidgets.QApplication.screens()[0].size().width()
        self.bk_lbl.setGeometry(0, 0, aw, ah)
        self.bk_lbl.setStyleSheet('background-color: yellow')

  

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 0, 0, 0)       
        layout.addWidget(self.thr_lbl,0, Qt.AlignLeft)        
        layout.addWidget(self.sl,0, Qt.AlignLeft)        
        layout.addWidget(QtWidgets.QLabel('0.10<----->0.80'), 0, Qt.AlignLeft)
        layout.addWidget(self.btn_new_screenshot,0, Qt.AlignLeft )
        
        layout.addWidget(self.clear_button,0, Qt.AlignLeft)        
        layout.addWidget(self.btn_close,0, Qt.AlignLeft)        
        layout.addWidget(QtWidgets.QLabel('Noise Type:'), 0, Qt.AlignLeft)  
        
        radiobutton = QtWidgets.QRadioButton('No noise')
        radiobutton.setChecked(True)
        radiobutton.noise = 'no_noise'
        radiobutton.toggled.connect(self.on_click_radioButton)
        layout.addWidget(radiobutton, 0, Qt.AlignLeft)
        self.radiobutton_no_noise = radiobutton
        all_noise = ['Salt','Pepper','Salt+pepper',
                     # 'Poisson',
                     'Speckle','Gaussian','GaussianBlur',
                     'MedianBlur','Occlusion','Periodic']
        for _n in all_noise:
            radiobutton = QtWidgets.QRadioButton(_n)
            radiobutton.noise = _n
            radiobutton.toggled.connect(self.on_click_radioButton)
            layout.addWidget(radiobutton, 0, Qt.AlignLeft)
            
        
        layout.addWidget(QtWidgets.QLabel('Noise Level:'), 0, Qt.AlignLeft)
        self.sln = QtWidgets.QSlider(Qt.Horizontal)
        self.sln.setObjectName('sln')        
        self.sln.setMinimum(0)
        self.sln.setMaximum(2)
        self.sln.setValue(0)
        self.sln.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sln.setTickInterval(1)
        self.sln.valueChanged.connect(self.noise_slider_value_change)
        layout.addWidget(self.sln,0, Qt.AlignLeft)
        layout.addWidget(QtWidgets.QLabel('Min<------->Max'), 0, Qt.AlignLeft)  
        self.alarm_lbl = QtWidgets.QLabel('Object not found.')
        layout.addWidget(self.alarm_lbl, 0, Qt.AlignLeft)  
        self.alarm_lbl.setStyleSheet('color: red')
        self.alarm_lbl.hide()
        layout.addStretch(1)
        
        self.setLayout(layout)
        self.setWindowFlags(Qt.FramelessWindowHint| Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground,True)




    def on_click_radioButton(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.clear_noise()
               
            self.noise_type = radioButton.noise            
            if self.noise_type!='no_noise':
                if self.main_screenshot is None:
                    preview_screen = QtWidgets.QApplication.primaryScreen().grabWindow(0,0,0,self.width(),self.height())
                    rgb_img = self.QScreenToArray(preview_screen)[:,:,:3]
                    self.main_screenshot = rgb_img[:,:,::-1]
                
                self.noisy_screenshot = self.apply_noise(self.main_screenshot,
                                                         self.noise_type,self.sln.value())
                height, width, channels = self.noisy_screenshot.shape
                bytesPerLine = channels * width
                qImg = QtGui.QImage(self.noisy_screenshot.copy(), width, height, bytesPerLine,
                                    QtGui.QImage.Format_RGB888)
                pixmap01 = QtGui.QPixmap.fromImage(qImg)           
                self.check_clear = True                        
                self.label_img.setPixmap(pixmap01)        
                self.label_img.update()


        
    def noise_slider_value_change(self):
        self.clear_noise()        
        if self.noise_type!='no_noise':
            if self.main_screenshot is None:
                preview_screen = QtWidgets.QApplication.primaryScreen().grabWindow(0,0,0,self.width(),self.height())
                rgb_img = self.QScreenToArray(preview_screen)[:,:,:3]
                self.main_screenshot = rgb_img[:,:,::-1]
                          
            self.noisy_screenshot = self.apply_noise(self.main_screenshot,
                                                     self.noise_type,self.sln.value())
            height, width, channels = self.noisy_screenshot.shape
            bytesPerLine = channels * width
            qImg = QtGui.QImage(self.noisy_screenshot.copy(), width, height, 
                                bytesPerLine, QtGui.QImage.Format_RGB888)
            pixmap01 = QtGui.QPixmap.fromImage(qImg)
            self.check_clear = True                        
            self.label_img.setPixmap(pixmap01)        
            self.label_img.update()


    def clear_noise(self):
        self.alarm_lbl.hide()        
        pixmap01 = QtGui.QPixmap(self.size())
        pixmap01.fill(Qt.transparent)
        self.label_img.setPixmap(pixmap01)
        self.label_img.update()
            
    def on_click_clear(self):
        self.alarm_lbl.hide()        
        if self.check_clear:           
            pixmap01 = QtGui.QPixmap(self.size())
            pixmap01.fill(Qt.transparent)
            self.check_clear = False
            self.main_screenshot = None
            self.noisy_screenshot = None
            self.noise_type = 'no_noise'
            self.radiobutton_no_noise.setChecked(True)            
            self.label_img.setPixmap(pixmap01)
            self.label_img.update() 
            
            
    def apply_yolo(self):
        if self.noisy_screenshot is None:
            self.hide()
            preview_screen = QtWidgets.QApplication.primaryScreen().grabWindow(0,0,0,self.width(),self.height())
            rgb_img = self.QScreenToArray(preview_screen)[:,:,:3]
            self.main_screenshot = rgb_img[:,:,::-1]
            if self.noise_type!='no_noise':
                tmp_img = self.apply_noise(self.main_screenshot,self.noise_type,self.sln.value())
            else:
                tmp_img = self.main_screenshot
        else:
            tmp_img = self.noisy_screenshot.copy()
            self.noisy_screenshot = None
        
        height, width, channels = tmp_img.shape
        bytesPerLine = channels * width
        qImg = QtGui.QImage(tmp_img.copy(), width, height, bytesPerLine,
                      QtGui.QImage.Format_RGB888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
                
        res = self.yolo_model(tmp_img)
        res = res.pandas().xyxy[0]
        # print(res,'len:',len(res))
        if len(res)>0:
            bbx = res.to_numpy()
            if len(bbx)>0 and max(bbx[:,4])>self.confidence_threshold:
                               
                painter = QtGui.QPainter(pixmap01)
                
                all_classes = np.unique(bbx[:,6])
                
                color_bbx_dict = {}
                color_txt_dict = {}                

                cnt = 0
                for _cls in all_classes:
                    if cnt >= (len(self.all_colors)-1):
                        cnt = 0
                    color_bbx_dict[_cls] = self.all_colors[cnt]
                    color_txt_dict[_cls] = self.all_colors[cnt+1]
                    cnt += 1
                    
                for bb in bbx:
                    if bb[4]>self.confidence_threshold:
                        painter.setPen(QtGui.QPen(QtGui.QColor(*color_bbx_dict[bb[6]]),6))
                        painter.drawRect(bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1])
                        painter.setPen(QtGui.QPen(QtGui.QColor(*color_txt_dict[bb[6]])))
                        painter.setFont(QtGui.QFont('Helvetica', 32))
                        painter.drawText(bb[0],bb[1], bb[6])
                        
                painter.end()
                self.check_clear = True
                self.label_img.setPixmap(pixmap01)        
                self.label_img.update()
            else:
                print('Object not found.')
                self.alarm_lbl.show()
        else:
            self.alarm_lbl.show()
            print('Object not found.')
            
        self.show()


        
    def slider_value_change(self, event):
        self.confidence_threshold = self.sl.value()/100
        tdt = str(self.confidence_threshold)
        if len(tdt)==3:
            tdt+='0'
        self.thr_lbl.setText('Confidence:'+tdt)
        
    def apply_noise(self,main_screenshot,noise,level=0):
        if noise=='no_noise':
            noise_img = main_screenshot.copy()           
        _L = [.1,.3,.5]
        if noise=='Salt':
            noise_img = random_noise(main_screenshot, 
                                     mode='salt',amount=_L[level])
            noise_img = np.array(255*noise_img, dtype = 'uint8')
        if noise=='Pepper':
            noise_img = random_noise(main_screenshot, 
                                     mode='pepper',amount=_L[level])
            noise_img = np.array(255*noise_img, dtype = 'uint8')                
        if noise=='Salt+pepper':
            noise_img = random_noise(main_screenshot, 
                                     mode='s&p',amount=_L[level])
            noise_img = np.array(255*noise_img, dtype = 'uint8')  
            
        if noise=='Poisson':
            noise_img = random_noise(main_screenshot, 
                                     mode='poisson')
            noise_img = np.array(255*noise_img, dtype = 'uint8')
        
        _L = [.01,.08,.16]
        if noise=='Speckle':
            noise_img = random_noise(main_screenshot,
                                     mode='speckle', var=_L[level])
            noise_img = np.array(255*noise_img, dtype = 'uint8')
        
        _L = [.01,0.1,.15]             
        if noise=='Gaussian':
            noise_img = random_noise(main_screenshot, 
                                     mode='gaussian', var=_L[level])
            noise_img = np.array(255*noise_img, dtype = 'uint8')        

        _L = [5,9,13]
        if noise=='MedianBlur':
            noise_img = cv2.medianBlur(main_screenshot, _L[level])
            
        _L = [7,15,21]            
        if noise=='GaussianBlur':
            noise_img = cv2.GaussianBlur(main_screenshot, 
                                        (_L[level], _L[level]),0)   
        _L = [40,80,100]
        if noise=='Occlusion':
            noise_img = main_screenshot.copy()
            num_blk = (140-_L[level])
            for i in range(num_blk):
                _r = np.random.randint(noise_img.shape[0]-max(_L))
                _c = np.random.randint(noise_img.shape[1]-max(_L))
                noise_img[_r:_r+_L[level],
                          _c:_c+_L[level],:] = np.random.randint(0,255,3)
        
        _L = [50,100,200]
        if noise=='Periodic':
            noise_img = main_screenshot.copy()                

            X, Y = np.meshgrid(range(0, noise_img.shape[0]), 
                               range(0, noise_img.shape[1]))
            xs = np.linspace(-2*np.pi, 2*np.pi, noise_img.shape[0])
            ys = np.linspace(-2*np.pi, 2*np.pi, noise_img.shape[1]) 
            X, Y = np.meshgrid(xs, ys)
            periodic_noise = _L[level]*np.sin(X*np.random.randint(4) + Y* (400/_L[level]))            
            # periodic_noise += _L[level]*np.sin(X*-2 + Y* (400/_L[level]))            
            noise_img = noise_img + np.repeat(periodic_noise.T[:,:,None], 3, axis=-1)
            noise_img[noise_img<0]=0
            noise_img[noise_img>255]=255
            noise_img = np.array(noise_img, dtype = 'uint8')
        return noise_img           

    def keyPressEvent(self, qKeyEvent):
        if qKeyEvent.key() == Qt.Key_Return: 
            self.apply_yolo()
        else:
            super().keyPressEvent(qKeyEvent)
            
    def QScreenToArray(self,pixmap):
        channels_count = 4
        image = pixmap.toImage()
        s = image.bits().asstring(pixmap.width() * pixmap.height() * channels_count)
        img = np.frombuffer(s, dtype=np.uint8).reshape((pixmap.height(), pixmap.width(), channels_count))    
        return img
            
    def close_app(self):
        self.close()
            
class MyLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        QtWidgets.QLabel.__init__(self, parent=parent)
    def mousePressEvent(self, event):
        self.clicked.emit()
    
root = QtWidgets.QApplication(sys.argv)
app = MainWindow()
app.showMaximized()
sys.exit(root.exec_())    