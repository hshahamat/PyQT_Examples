from PyQt5 import QtWidgets,QtCore,QtGui
import sys
import numpy as np
import cv2
from scipy.ndimage import morphology



class MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__()
        self.resize(QtWidgets.QApplication.screens()[0].size())
        self.setWindowTitle("Screenshoter")


        self.img_result_lbl = QtWidgets.QLabel(self)
        self.img_result_lbl.setGeometry(self.geometry())        

        self.btn_new_screenshot = QtWidgets.QPushButton("New screenshot")
        self.btn_new_screenshot.setFixedSize(90, 30)        
        self.btn_new_screenshot.clicked.connect(self.start_screenshot)

        self.btn_clear = QtWidgets.QPushButton('Clear')
        self.btn_clear.setIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, 'SP_DialogResetButton')))
        self.btn_clear.setFixedSize(90,30)
        self.btn_clear.clicked.connect(self.clear_label)
        
        self.btn_close = QtWidgets.QPushButton("Close App")
        self.btn_close.setFixedSize(90, 30)
        self.btn_close.clicked.connect(self.close_app)
        
        
        polygon_points = [[585,435],[735,435],
                          [585,620],[735,620]]
        
        band_rect = [200,250,200,150]
        
        self.frames = ResizableFrames(self,polygon_points,band_rect)            
        self.frames.setGeometry(self.geometry())  
        self.frames.mousePressEvent = self.clear_label 
        self.frames.mouseReleaseEvent = self.start_screenshot 
        
        self.bk_lbl = QtWidgets.QLabel(self)
        aw = 100
        ah = self.height()
        self.bk_lbl.setGeometry(0, 0, aw, ah)
        self.bk_lbl.setStyleSheet("background-color: yellow")
        
        self.layout = QtWidgets.QVBoxLayout(self)            
        self.layout.setContentsMargins(5, 0, 0, 5)        
        self.layout.addWidget(self.btn_new_screenshot,0, QtCore.Qt.AlignLeft )
        self.layout.addWidget(self.btn_clear,0, QtCore.Qt.AlignLeft )
        self.layout.addWidget(self.btn_close,0, QtCore.Qt.AlignLeft)
        
        self.layout.addStretch(1)
        self.setLayout(self.layout)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint| QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground,True)
    
            
    def close_app(self):
        self.close()
    def start_screenshot(self,e):
        self.frames.hide()
        QtCore.QTimer.singleShot(3, self.take_screenshot)

    def take_screenshot(self):
        tmp_QAP = QtWidgets.QApplication.primaryScreen()
        preview_screen = tmp_QAP.grabWindow(0,
                                            self.frames.rband1.x(),
                                            self.frames.rband1.y(),
                                            self.frames.rband1.width(),
                                            self.frames.rband1.height())
        tmp_img = self.QScreenToArray(preview_screen)[:,:,:3]
        self.region_screenshot = tmp_img[:,:,::-1]
        self.region_screenshot = cv2.resize(self.region_screenshot, (self.width(),self.height()))        
        rows,cols,ch = self.region_screenshot.shape        
        
        current_points = np.float32([[p.x()+7,p.y()+7] for p in self.frames.Circles])
        (tl, tr, bl, br) = current_points        
        widthA = (tr[0] - tl[0])
        widthB = (br[0] - bl[0])
        heightA = (bl[1] - tl[1])
        heightB = (br[1] - tr[1])
        mW = max(int(widthA), int(widthB))
        mH = max(int(heightA), int(heightB))
        
        resized_screenshot = cv2.resize(self.region_screenshot, (mW,mH))
        src_img_points = np.float32([[0, 0],[mW-1, 0],
                	                 [0, mH-1],[mW-1, mH-1]])

        M = cv2.getPerspectiveTransform(src_img_points, current_points)
        new_screenshot = cv2.warpPerspective(resized_screenshot, M, (cols,rows))

        gray_screenshot = cv2.cvtColor(new_screenshot,cv2.COLOR_RGB2GRAY)
        _,mask = cv2.threshold(gray_screenshot, 0, 255, cv2.THRESH_BINARY)
        mask = morphology.binary_fill_holes(mask/255).astype("uint8")*255

        new_screenshot = cv2.cvtColor(new_screenshot,cv2.COLOR_RGB2RGBA)
        new_screenshot[:,:,3] = mask
        bytesPerLine = 4 * cols
        qImg = QtGui.QImage(new_screenshot,
                            cols, rows, bytesPerLine,
                            QtGui.QImage.Format_RGBA8888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)

        self.img_result_lbl.setPixmap(pixmap01)
        self.img_result_lbl.update()
        self.frames.ploygon_trnasparent = True
        self.frames.show()

    def keyPressEvent(self, qKeyEvent):
        if qKeyEvent.key() == QtCore.Qt.Key_Return: 
            self.start_screenshot(qKeyEvent)
        else:
            super().keyPressEvent(qKeyEvent)
            
    def QScreenToArray(self,pixmap):
        channels_count = 4
        image = pixmap.toImage()
        s = image.bits().asstring(pixmap.width() * pixmap.height() * channels_count)
        img = np.frombuffer(s, dtype=np.uint8).reshape((pixmap.height(), pixmap.width(), channels_count))    
        return img

    def clear_label(self,e):
        if self.frames.ploygon_trnasparent:
            pixmap01 = QtGui.QPixmap(self.size())
            pixmap01.fill(QtCore.Qt.transparent)
            self.img_result_lbl.setPixmap(pixmap01)
            self.img_result_lbl.update()
            self.frames.ploygon_trnasparent = False        
            self.frames.show()
        
'''##################################################'''
'''##################################################'''
class ResizableFrames(QtWidgets.QWidget):
    def __init__(self, parent=None,points=[],rect=[]):       
        super(ResizableFrames, self).__init__(parent)
        self.Circles = []
        self.ploygon_trnasparent = False
        
        if points!=[]:
            for _i,_point in enumerate(points):
                if _i==0:
                    _circle = MoveingCircle(self,color=1)
                else:
                    _circle = MoveingCircle(self,color=2)
    
                _circle.setGeometry(_point[0],_point[1],15,15)        
                _circle.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
                self.Circles.append(_circle)
                
            self.Circles[0].moveEvent = self.moveCircle0
            self.Circle0_old_pos = self.Circles[0].pos()
            
            self.rband1 = ResizableRubberBand(self)            
            self.rband1.setGeometry(*rect)        
            self.rband1.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
            self.rband1_old_pos = self.rband1.pos()

           
    def moveCircle0(self, event):
        moved = self.Circles[0].pos() - self.Circle0_old_pos
        for _i in range(1,len(self.Circles)):
            self.Circles[_i].move(self.Circles[_i].pos()+moved)
        self.Circle0_old_pos = self.Circles[0].pos()    
        
            
    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)        
        qp.setPen(QtGui.QPen(QtCore.Qt.blue, 8))
        if self.ploygon_trnasparent:
            qp.setBrush(QtCore.Qt.NoBrush)
            qp.setOpacity(1)
        else:
            qp.setBrush(QtGui.QBrush(QtCore.Qt.black))
            qp.setOpacity(0.3)        
        if self.Circles!=[]:
            points = [_circle.geometry().center() for _circle in self.Circles]
            points[2],points[3] = points[3],points[2]
            points = QtGui.QPolygon(points)
            qp.drawPolygon(points)            
        qp.end()       
        self.update()


'''##################################################'''
'''##################################################'''    
class MoveingCircle(QtWidgets.QWidget):
    def __init__(self, parent=None, color=1):
        super(MoveingCircle, self).__init__(parent)
        self.mousePressPos = None
        self.mouseMovePos = None
        self.parent = parent
        self.color = color      
        self.setWindowFlags(QtCore.Qt.SubWindow)        
        self.show()

    def paintEvent(self, event):
        window_size = self.size()
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        qp.setOpacity(1)        
        if self.color==1:            
            brCircle = QtGui.QBrush(QtGui.QColor(QtCore.Qt.red))
            penBorder = QtGui.QPen(QtCore.Qt.red,3)
        else:
            brCircle = QtGui.QBrush(QtGui.QColor(QtCore.Qt.magenta))
            penBorder = QtGui.QPen(QtCore.Qt.magenta,3)                
        qp.setBrush(brCircle)
        qp.setPen(penBorder)
        _cs = 10                    
        qp.drawEllipse((window_size.width()/2)-(_cs/2), 
                       (window_size.height()/2)-(_cs/2),_cs, _cs)         
        qp.end()

    def mousePressEvent(self, event):
        if  event.button() == QtCore.Qt.LeftButton:
            self.setCursor(QtGui.QCursor(QtCore.Qt.ClosedHandCursor))            
            self.mousePressPos = event.globalPos()                # global
            self.mouseMovePos = event.globalPos() - self.pos()    # local
        super(MoveingCircle, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if  event.buttons() & QtCore.Qt.LeftButton:                         
            globalPos = event.globalPos()
            moved = globalPos - self.mousePressPos
            if moved.manhattanLength() > 0:
                diff = globalPos - self.mouseMovePos
                self.move(diff)                     
                self.mouseMovePos = globalPos - self.pos()
        super(MoveingCircle, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):        
        if self.mousePressPos is not None:
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        super(MoveingCircle, self).mouseReleaseEvent(event)
'''##################################################'''
'''##################################################'''
class ResizableRubberBand(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ResizableRubberBand, self).__init__(parent)
        self.draggable = True
        self.dragging_threshold = 5
        self.mousePressPos = None
        self.mouseMovePos = None
        self.parent = parent
        self.setWindowFlags(QtCore.Qt.SubWindow)
        
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)        
        layout.addWidget(
            QtWidgets.QSizeGrip(self), 0,0,
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)        
        layout.addWidget(
            QtWidgets.QSizeGrip(self), 0,1,
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)        
        layout.addWidget(
            QtWidgets.QSizeGrip(self), 1,0,
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)        
        layout.addWidget(
            QtWidgets.QSizeGrip(self), 1,1,
            QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
        self.show()

    def paintEvent(self, event):
        window_size = self.size()
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        
        qp.setOpacity(0.2)
        qp.setBrush(QtGui.QBrush(QtCore.Qt.green))    
        qp.drawRect(0, 0, window_size.width(), window_size.height())
               
        qp.setPen(QtGui.QPen(QtCore.Qt.magenta,15))
        qp.setBrush(QtCore.Qt.NoBrush)    
        qp.setOpacity(1)        
        qp.drawRect(0, 0, window_size.width(), window_size.height())        
        qp.end()

    def mousePressEvent(self, event):
        if self.draggable and event.button() == QtCore.Qt.LeftButton:
            self.setCursor(QtGui.QCursor(QtCore.Qt.ClosedHandCursor))            
            self.mousePressPos = event.globalPos()                # global
            self.mouseMovePos = event.globalPos() - self.pos()    # local
        super(ResizableRubberBand, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.draggable and event.buttons() & QtCore.Qt.LeftButton:                          
            globalPos = event.globalPos()
            moved = globalPos - self.mousePressPos
            if moved.manhattanLength() > self.dragging_threshold:
                diff = globalPos - self.mouseMovePos
                self.move(diff)
                self.mouseMovePos = globalPos - self.pos()
        super(ResizableRubberBand, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):        
        if self.mousePressPos is not None:
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))            
            if event.button() == QtCore.Qt.LeftButton:
                moved = event.globalPos() - self.mousePressPos
                if moved.manhattanLength() > self.dragging_threshold:
                    event.ignore()
                self.mousePressPos = None
        super(ResizableRubberBand, self).mouseReleaseEvent(event)
'''##################################################'''
'''##################################################'''
root = QtWidgets.QApplication(sys.argv)
app = MainWindow()
app.showMaximized()
sys.exit(root.exec_())