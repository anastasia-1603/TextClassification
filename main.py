import sys
from enum import Enum

import numpy as np
from PyQt5.QtGui import QTextCursor, QFont
from PyQt5.QtWidgets import QWidget, QPushButton, QMainWindow, QVBoxLayout, QLabel, QHBoxLayout, \
    QRadioButton, QComboBox, QTextEdit, QButtonGroup, QApplication
import statistic
from statistic_mlp import MLPSearch

MAX = 3000
MIN = 300


class Mode(Enum):
    NN = 0
    ML = 1


MODE = Mode.ML


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        widget = QWidget()
        self.setCentralWidget(widget)
        self.setGeometry(0, 0, 800, 400)

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

        self.result_lbl = QLabel("Результат")
        self.result_lbl.setFont(QFont('Arial', 15))
        min_text_length_lbl = QLabel("Введите как минимум %d символов " % MIN)

        self.outputLbl = QLabel("...")


        self.group.addButton(self.rbS)
        self.group.addButton(self.rbN)

        self.rbN.setChecked(True)
        settingsVBox.addWidget(self.rbN)
        settingsVBox.addWidget(self.rbS)
        settingsVBox.addWidget(self.cb)
        settingsVBox.addWidget(btn)

        # self.progress = QtGui.QProgressBar(self)
        # self.progress.setGeometry(200, 80, 250, 20)
        resultHBox.addWidget(self.result_lbl)
        resultHBox.addWidget(self.outputLbl)
        # inputVBox.addLayout(settingsVBox)
        inputVBox.addWidget(min_text_length_lbl)
        inputVBox.addWidget(self.inputTextEdit)

        mainInputHBox.addLayout(settingsVBox)
        mainInputHBox.addLayout(inputVBox)

        mainVBox.addLayout(mainInputHBox, stretch=1)
        mainVBox.addLayout(resultHBox)
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
            self.outputLbl.setFont(QFont('Arial', 15))
            self.outputLbl.setStyleSheet("color: green")
            self.result_lbl.setText("Вычисляем...")
            res = get_style(MODE, type, text)
            self.outputLbl.setText(res)
            self.result_lbl.setText("Результат")

        else:
            self.outputLbl.setFont(QFont('Arial', 10))
            self.outputLbl.setStyleSheet("color: red")
            self.outputLbl.setText("Введите как минимум %d символов " % MIN)

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
            from sentence_embeddings import Network, TokenCNN, ModelCompilation, \
                ClassificationHead, CustomConv1D, DataModule, predictEmb

            prob = predictEmb(text)
        else:
            from sentence import Network, TokenCNN, ModelCompilation, \
                ClassificationHead, CustomConv1D, DataModule, CharCNN, TokenCNNRNN, predict
            prob = predict(text, type)
    styles = ["Научный стиль", "Официально-деловой стиль",
              "Публицистический стиль", "Разговорный стиль", "Художественный стиль"]
    i = np.argmax(prob[0])
    style = styles[i]
    res = f'{style} : {prob[0][i]:.5f}'
    return res


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    app.exec()
    # text = '''Привет! Здравствуйте, как я могу вам помочь? Я ищу подарок для своей сестры на ее день рождения,
    # что вы могли бы предложить? Намного проще будет, если я знаю интересы вашей сестры. Она занимается каким-то хобби
    # или у нее есть любимый цвет? Она увлекается рисованием и любит всю гамму зеленых цветов. Могу предложить ей набор
    # кистей и атрибутов для рисования, а может быть, замечательный зеленый шарф, который будет сочетаться с любой
    # одеждой. Звучит отлично! Я думаю, она будет в восторге от набора кистей. Отлично, я помогу вам найти то,
    # что вам нужно. Какой предпочитаете ценовой диапазон? Не больше 50 долларов, я хочу не только порадовать сестру,
    # но и оставить себе возможность купить себе что-то вкусное. Понял, буду искать в этом диапазоне. А вы готовы
    # купить сейчас или хотите подумать еще? Я хотел бы сделать покупку сейчас, если вы уверены, что это лучший выбор.
    # Я уверен, что это отличный выбор. Давайте перейдем к кассе, чтобы я мог принять ваш заказ. Спасибо, что выбрали
    # наш магазин! '''

    # print("CharCNN + TokenCNN " + get_style(Mode.NN, "Word Embeddings", text))
    # print("CharCNN + TokenCNNRNN " + get_style(Mode.NN, "CharCNN + TokenCNN", text))
    # print("Word Embeddings " + get_style(Mode.NN, "CharCNN + TokenCNNRNN", text))

    # print("Logistic Regression " + get_style(Mode.ML, "Logistic Regression", text))
    # print("Decision Tree " + get_style(Mode.ML, "Decision Tree", text))
    # print("Random Forest " + get_style(Mode.ML, "Random Forest", text))
    # print("SVC " + get_style(Mode.ML, "SVC", text))
    # print("K-Nearest Neighbors " + get_style(Mode.ML, "K-Nearest Neighbors", text))
    # print("XGBoost " + get_style(Mode.ML, "XGBoost", text))
    # print("CatBoost " + get_style(Mode.ML, "CatBoost", text))
    # print("LightGBM " + get_style(Mode.ML, "LightGBM", text))
    # print("MLP " + get_style(Mode.ML, "MLP", text))

    # proba = statistic.predict(text, "MLP")
    # proba = sentence.predict(text, "Word Embeddings")
    # proba = predict_emb(text)

    # from classic import *
    # proba = predict_classic(text, "CharCNN + TokenCNN")
    # styles = ["Научный стиль", "Официально-деловой стиль",
    #           "Публицистический стиль", "Разговорный стиль", "Художественный стиль"]
    # i = np.argmax(proba[0])
    # style = styles[i]
    #
    # print(f'{style} : {proba[0][i]:.5f}')
    #
    # # from embeddings import *
    # proba = predict_emb(text)
    #
    # i = np.argmax(proba[0])
    # style = styles[i]
    #
    # print(f'{style} : {proba[0][i]:.5f}')
