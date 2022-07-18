# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import networks
from tensorflow.keras import models


class Ui_MainWindow(object):
    def __init__(self):
        self.RESULT = None
        self.IMGNAME = None

    # 界面UI设置
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setFixedSize(800, 600)
        MainWindow.setWindowIcon(QIcon('images/icon.jpg'))
        MainWindow.setStyleSheet("#MainWindow{border-image:url('images/back.jpg');}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Up_widget = QtWidgets.QWidget(self.centralwidget)
        self.Up_widget.setGeometry(QtCore.QRect(33, 20, 741, 371))
        self.Up_widget.setObjectName("Up_widget")
        self.label_2 = QtWidgets.QLabel(self.Up_widget)
        self.label_2.setGeometry(QtCore.QRect(75, 120, 256, 256))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_2.setStyleSheet('background-color:white;color:white;')
        self.label = QtWidgets.QLabel(self.Up_widget)
        self.label.setGeometry(QtCore.QRect(120, 60, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_3 = QtWidgets.QLabel(self.Up_widget)
        self.label_3.setGeometry(QtCore.QRect(410, 120, 256, 256))
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_3.setStyleSheet('background-color:white;color:white;')
        self.label_4 = QtWidgets.QLabel(self.Up_widget)
        self.label_4.setGeometry(QtCore.QRect(500, 60, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.Down_widget = QtWidgets.QWidget(self.centralwidget)
        self.Down_widget.setGeometry(QtCore.QRect(33, 420, 741, 111))
        self.Down_widget.setObjectName("Down_widget")
        self.pushButton = QtWidgets.QPushButton(self.Down_widget)
        self.pushButton.setGeometry(QtCore.QRect(90, 20, 100, 40))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.Down_widget)
        self.pushButton_2.setGeometry(QtCore.QRect(300, 20, 100, 40))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.Down_widget)
        self.pushButton_3.setGeometry(QtCore.QRect(550, 20, 100, 40))
        self.pushButton_3.setObjectName("pushButton_3")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(self.open_file)  # signal-slot connect
        self.pushButton_2.clicked.connect(self.pix2pix)  # signal-slot connect
        self.pushButton_3.clicked.connect(self.save_file)  # signal-slot connect

        self.Down_widget.setStyleSheet('''
              QPushButton{
                border:none;color:white;
                font-size:18px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                background:#0099CC;
                border-radius:5px;
              }
              QLabel{
                font-size:22px;
                color:white;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
              }
              QLineEdit{
                font-size:18px;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
              }
              QPushButton:hover{
                border-left:4px solid white;
                font-weight:700;
                
              }
            ''')

        self.centralwidget.setStyleSheet('''
            QWidget#Down_widget{
              
              border-radius:10px;
            }
            QWidget#Up_widget{
              
              border-radius:10px;
            }
            ''')

    # 打开文件
    def open_file(self):
        self.label_3.setText(' ')
        imgName, imgType = QFileDialog.getOpenFileName(None, "打开图片", "", "*.png;;*.jpg;;All Files(*)")
        if imgName != None:
            self.IMGNAME = imgName
            jpg = QtGui.QPixmap(imgName).scaled(self.label_2.width(), self.label_2.height())
            print(imgName)
            self.label_2.setPixmap(jpg)

    # 人像卡通化
    def pix2pix(self):
        if self.IMGNAME is None:
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '未选择图片！')
            msg_box.exec_()
        else:
            generator = models.load_model('model/g_p_500.h5', compile=False)
            img = cv2.cvtColor(cv2.imread(self.IMGNAME), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img = img.astype('float32') / 255

            # 增加图像维度
            img = img[None]
            result = generator(img, training=True).numpy()
            # 删除图像第一维度
            self.RESULT = np.squeeze(result)

            result = self.RESULT * 255
            result = result.astype("uint8")
            im = QtGui.QImage(result.data, result.shape[1], result.shape[0], result.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap(im).scaled(self.label_3.width(), self.label_3.height())
            self.label_3.setPixmap(pix)

    # 保存结果
    def save_file(self):
        if self.IMGNAME is None:
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '未选择图片！')
            msg_box.exec_()
        elif self.RESULT is None:
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '未分割图片！')
            msg_box.exec_()
        else:
            filename = QFileDialog.getSaveFileName(None, "save file", "", "Images (*.png *.xpm *.jpg);;all file(*)")
            if filename[0] != '':
                result = self.RESULT * 255
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename[0], result)
                msg_box = QMessageBox(QMessageBox.Information, '提示', '保存成功！')
                msg_box.exec_()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "人像卡通化"))
        self.label.setText(_translate("MainWindow", "原始人像"))
        self.label_4.setText(_translate("MainWindow", "卡通化"))
        self.pushButton.setText(_translate("MainWindow", "打开"))
        self.pushButton_2.setText(_translate("MainWindow", "转换"))
        self.pushButton_3.setText(_translate("MainWindow", "保存"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()  # 装载需要的各种组件、控件
    ui = Ui_MainWindow()  # ui类的实例化对象
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
