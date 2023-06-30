import sys
import time
from enum import Enum

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor, QFont
from PyQt5.QtWidgets import QWidget, QPushButton, QMainWindow, QVBoxLayout, QLabel, QHBoxLayout, \
    QRadioButton, QComboBox, QTextEdit, QButtonGroup, QApplication

import statistic
from statistic_mlp import MLPSearch

MAX = 5000
MIN = 300


class Mode(Enum):
    NN = 0
    ML = 1


MODE = Mode.ML


class AnalyzeThread(QThread):
    statusUpdated = pyqtSignal(str)
    resultReady = pyqtSignal(str)

    def __init__(self, text, type):
        super().__init__()
        self.text = text
        self.type = type

    def run(self):
        self.statusUpdated.emit("Вычисляем...")
        res = get_style(MODE, self.type, self.text)
        self.statusUpdated.emit("")
        self.resultReady.emit(res)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        widget = QWidget()

        self.setCentralWidget(widget)
        self.setGeometry(0, 0, 1000, 600)

        self.statusBar()
        mainVBox = QVBoxLayout(widget)
        mainInputHBox = QHBoxLayout()
        settingsVBox = QVBoxLayout()
        inputVBox = QVBoxLayout()
        resultHBox = QHBoxLayout()

        self.group = QButtonGroup()

        self.rbN = QRadioButton("Нейросетевой анализ текста")
        self.rbN.setFont(QFont('Arial', 12))
        self.rbS = QRadioButton("Статистический анализ текста")
        self.rbS.setFont(QFont('Arial', 12))
        self.cb = QComboBox()
        self.cb.setFont(QFont('Arial', 12))
        btn = QPushButton("Запуск")
        btn.setFont(QFont('Arial', 12))
        self.inputTextEdit = QTextEdit()
        self.inputTextEdit.setFont(QFont('Arial', 10))
        self.inputTextEdit.setAcceptRichText(False)

        self.result_lbl = QLabel("Результат")
        self.result_lbl.setFont(QFont('Arial', 12))
        min_text_length_lbl = QLabel("Введите как минимум %d символов " % MIN)

        self.outputLbl = QLabel("")

        self.group.addButton(self.rbS)
        self.group.addButton(self.rbN)

        self.rbN.setChecked(True)
        settingsVBox.addWidget(self.rbN)
        settingsVBox.addWidget(self.rbS)
        settingsVBox.addWidget(self.cb)
        settingsVBox.addWidget(btn)

        resultHBox.addWidget(self.result_lbl)
        resultHBox.addWidget(self.outputLbl)
        inputVBox.addWidget(min_text_length_lbl)
        inputVBox.addWidget(self.inputTextEdit)

        mainInputHBox.addLayout(settingsVBox)
        mainInputHBox.addLayout(inputVBox)

        mainVBox.addLayout(mainInputHBox, stretch=4)
        mainVBox.addLayout(resultHBox, stretch=1)
        widget.setLayout(mainVBox)

        btn.clicked.connect(self.run_analyze)
        self.group.buttonClicked.connect(self.toggle)
        self.inputTextEdit.textChanged.connect(self.text_changed)
        self.toggle(self.rbN)

    def toggle(self, btn):
        self.cb.clear()
        global MODE
        if btn == self.rbN:

            MODE = Mode.NN
            self.cb.addItem("CharCNN + TokenCNN")
            self.cb.addItem("CharCNN + TokenCNNRNN")
            self.cb.addItem("Word Embeddings")
        else:
            MODE = Mode.ML
            self.cb.addItem("Logistic Regression")
            self.cb.addItem("Decision Tree")
            self.cb.addItem("Random Forest")
            self.cb.addItem("SVC")
            self.cb.addItem("K-Nearest Neighbors")
            self.cb.addItem("XGBoost")
            self.cb.addItem("CatBoost")
            self.cb.addItem("LightGBM")
            self.cb.addItem("MLP")
        self.cb.setCurrentIndex(0)

    def run_analyze(self):
        text = self.inputTextEdit.toPlainText()
        type = self.cb.currentText()
        if len(text) > MIN:
            self.analyzeThread = AnalyzeThread(text, type)
            self.analyzeThread.statusUpdated.connect(self.statusBar().showMessage)
            self.analyzeThread.resultReady.connect(self.showResult)
            self.analyzeThread.start()

        else:
            self.outputLbl.setFont(QFont('Arial', 10))
            self.outputLbl.setStyleSheet("color: red")
            self.outputLbl.setText("Введите как минимум %d символов " % MIN)

    def showResult(self, res):
        self.outputLbl.setFont(QFont('Arial', 12))
        self.outputLbl.setStyleSheet("color: green")
        self.outputLbl.setText(res)

    def text_changed(self):
        text2 = self.inputTextEdit.toPlainText()
        if len(text2) > MAX:
            self.inputTextEdit.setText(text2[:len(text2) - 1])
            self.inputTextEdit.moveCursor(QTextCursor.End, QTextCursor.MoveAnchor)


def get_style(mode, type, text):
    global Network, TokenCNN, ModelCompilation, ClassificationHead, CustomConv1D, \
        DataModule, CharCNN, TokenCNNRNN, predict, predictEmb

    if mode == Mode.ML:
        prob = statistic.predict(text, type)
    else:
        if type == "Word Embeddings":
            from sentence_embeddings import Network, ModelCompilation, DataModule, predictEmb

            prob = predictEmb(text)
        else:
            from sentence import Network, TokenCNN, ModelCompilation, \
                ClassificationHead, CustomConv1D, DataModule, CharCNN, TokenCNNRNN, predict
            prob = predict(text, type)

    styles = ["Научный стиль", "Официально-деловой стиль",
              "Публицистический стиль", "Разговорный стиль", "Художественный стиль"]
    result = {}
    for i in range(len(styles)):
        style = styles[i]
        result[style] = prob[0][i]
    result_sorted = sorted(result.items(), key=lambda x: x[1], reverse=True)

    res = ''
    for s, p in result_sorted:
        res = res + f'{s} : {p:.5f}\n'

    return res


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.setWindowTitle("Text classification")
    main_window.show()
    app.exec()
