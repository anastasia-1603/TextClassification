import pickle

import joblib
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torchmetrics
from lexicalrichness import LexicalRichness
from natasha import Segmenter, Doc, NewsEmbedding, NewsMorphTagger, MorphVocab
from torch import nn


class FeatureExtractionModule:
    def __init__(self):
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.morph_vocab = MorphVocab()

    # Преобразование набора данных текстов в набор данных векторов признаков
    def fromCsvToCsv(self, csv_file_path):
        if csv_file_path != 'texts.csv':
            dataframe = pd.read_csv(csv_file_path, sep=';', encoding="cp1251")
        else:
            dataframe = pd.read_csv(csv_file_path, sep=';', encoding="cp1251", header=1).dropna().reset_index(drop=True)
        dataframe.loc[:, "Style"] = dataframe.loc[:, "Style"].astype(
            'category').cat.codes
        dataframe_rows = dataframe.shape[0]

        dataset = pd.DataFrame()
        for csv_row in range(dataframe_rows):
            text = dataframe.loc[csv_row, "Text"]
            doc = Doc(text)
            doc.segment(self.segmenter)
            doc.tag_morph(self.morph_tagger)

            dataset.loc[csv_row, 'average_word_length'] = self.averageWordLength(doc)
            dataset.loc[csv_row, 'average_word_count'] = self.averageWordCount(doc)
            dataset.loc[csv_row, 'log_connection'] = self.logicalConnectionCoefficient(doc)

            dataset.loc[csv_row, 'verb_freq'] = self.posFrequencyCoefficient(doc, 'VERB')
            dataset.loc[csv_row, 'noun_freq'] = self.posFrequencyCoefficient(doc, 'NOUN')
            dataset.loc[csv_row, 'adv_freq'] = self.posFrequencyCoefficient(doc, 'ADV')
            dataset.loc[csv_row, 'adj_freq'] = self.posFrequencyCoefficient(doc, 'ADJ')
            dataset.loc[csv_row, 'propn_freq'] = self.posFrequencyCoefficient(doc, 'PROPN')

            dataset.loc[csv_row, 'verb_noun_freq'] = self.posCombinationFrequencyCoefficient(doc, 'VERB', 'NOUN')
            dataset.loc[csv_row, 'verb_adv_freq'] = self.posCombinationFrequencyCoefficient(doc, 'VERB', 'ADV')
            dataset.loc[csv_row, 'noun_noun_freq'] = self.posCombinationFrequencyCoefficient(doc, 'NOUN', 'NOUN')
            dataset.loc[csv_row, 'adv_noun_freq'] = self.posCombinationFrequencyCoefficient(doc, 'ADJ', 'NOUN')

            dataset.loc[csv_row, 'dinamism_static'] = self.dynamismStaticTextCoefficient(doc)

            dataset.loc[csv_row, 'dot_freq'] = self.punctFrequencyCoefficient(doc, '.')
            dataset.loc[csv_row, 'comma_freq'] = self.punctFrequencyCoefficient(doc, ',')
            dataset.loc[csv_row, 'colon_freq'] = self.punctFrequencyCoefficient(doc, ':')
            dataset.loc[csv_row, 'semicolon_freq'] = self.punctFrequencyCoefficient(doc, ';')
            dataset.loc[csv_row, 'quote_freq'] = self.punctFrequencyCoefficient(doc, '"')
            dataset.loc[csv_row, 'exclamation_freq'] = self.punctFrequencyCoefficient(doc, '!')
            dataset.loc[csv_row, 'question_freq'] = self.punctFrequencyCoefficient(doc, '?')
            dataset.loc[csv_row, 'dash_freq'] = self.punctFrequencyCoefficient(doc, '—')

            dataset.loc[csv_row, 'lex_rich'] = self.lexicalRichnessCoefficient(doc)

            dataset.loc[csv_row, 'y (style)'] = dataframe.loc[csv_row, 'Style']
        return dataset

    # Преобразование 1 текста (для предсказания) в вектор признаков
    def fromTextToVector(self, text):
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        average_word_length = self.averageWordLength(doc)
        average_word_count = self.averageWordCount(doc)
        log_connection = self.logicalConnectionCoefficient(doc)

        verb_freq = self.posFrequencyCoefficient(doc, 'VERB')
        noun_freq = self.posFrequencyCoefficient(doc, 'NOUN')
        adv_freq = self.posFrequencyCoefficient(doc, 'ADV')
        adj_freq = self.posFrequencyCoefficient(doc, 'ADJ')
        propn_freq = self.posFrequencyCoefficient(doc, 'PROPN')

        verb_noun_freq = self.posCombinationFrequencyCoefficient(doc, 'VERB', 'NOUN')
        verb_adv_freq = self.posCombinationFrequencyCoefficient(doc, 'VERB', 'ADV')
        noun_noun_freq = self.posCombinationFrequencyCoefficient(doc, 'NOUN', 'NOUN')
        adv_noun_freq = self.posCombinationFrequencyCoefficient(doc, 'ADJ', 'NOUN')

        dinamism_static = self.dynamismStaticTextCoefficient(doc)

        dot_freq = self.punctFrequencyCoefficient(doc, '.')
        comma_freq = self.punctFrequencyCoefficient(doc, ',')
        colon_freq = self.punctFrequencyCoefficient(doc, ':')
        semicolon_freq = self.punctFrequencyCoefficient(doc, ';')
        quote_freq = self.punctFrequencyCoefficient(doc, '"')
        exclamation_freq = self.punctFrequencyCoefficient(doc, '!')
        question_freq = self.punctFrequencyCoefficient(doc, '?')
        dash_freq = self.punctFrequencyCoefficient(doc, '—')

        lex_rich = self.lexicalRichnessCoefficient(doc)

        return np.array([average_word_length, average_word_count, log_connection, verb_freq, noun_freq,
                         adv_freq, adj_freq, propn_freq, verb_noun_freq, verb_adv_freq, noun_noun_freq,
                         adv_noun_freq, dinamism_static, dot_freq, comma_freq, colon_freq, semicolon_freq,
                         quote_freq, exclamation_freq, question_freq, dash_freq, lex_rich])

    # Подсчёт количества слов в тексте (исключая знаки препинания и цифры и ошибочные слова)
    def wordCount(self, doc):
        token_count = 0
        for token in doc.tokens:
            if token.pos != 'PUNCT' and token.pos != 'X' and token.pos != 'NUM':
                token_count = token_count + 1
        return token_count

    # Подсчёт количества препинанйи в тексте
    def punctCount(self, doc):
        punct_count = 0
        for token in doc.tokens:
            if token.pos == 'PUNCT':
                punct_count = punct_count + 1
        return punct_count

    # Показатель среднего размера токена
    def averageWordLength(self, doc):
        token_count = self.wordCount(doc)
        token_len = []
        for token in doc.tokens:
            if token.pos != 'PUNCT' and token.pos != 'X' and token.pos != 'NUM':
                token_len.append(len(token.text))
        return np.mean(token_len)

    # Показатель среднего размера предложения
    def averageWordCount(self, doc):
        sent_count = len(doc.sents)
        token_count = self.wordCount(doc)
        return token_count / sent_count

    # Коэффицент частотности части речи
    def posFrequencyCoefficient(self, doc, pos_tag):
        all_token_count = self.wordCount(doc)
        pos_count = 0
        for token in doc.tokens:
            if token.pos == pos_tag:
                pos_count = pos_count + 1
        return pos_count / all_token_count

    # Коэффицент количества частиречной сочетаемости
    def posCombinationFrequencyCoefficient(self, doc, pos_tag_1, pos_tag_2):
        all_token_count = self.wordCount(doc)
        pos_count = 0
        for i in range(1, len(doc.tokens)):
            if doc.tokens[i].pos == pos_tag_1 and doc.tokens[i - 1].pos == pos_tag_2:
                pos_count = pos_count + 1
        return pos_count / (all_token_count - 1)

    # Частота встречаемости знаков препинания  (точка, запятая, двоеточие, точка с запятой, кавычки, скобки, вопросительный знак и тире)
    def punctFrequencyCoefficient(self, doc, punct_type):
        all_punct_count = self.punctCount(doc)
        punct_count = 0
        for token in doc.tokens:
            if token.text == punct_type:
                punct_count = punct_count + 1
        return punct_count / all_punct_count

    # Коэффицент соотношения динамичности и статичности текста.
    def dynamismStaticTextCoefficient(self, doc):
        verb_noun = self.posCombinationFrequencyCoefficient(doc, 'VERB', 'NOUN')
        verb_adv = self.posCombinationFrequencyCoefficient(doc, 'VERB', 'ADV')
        noun_noun = self.posCombinationFrequencyCoefficient(doc, 'NOUN', 'NOUN')
        adv_noun = self.posCombinationFrequencyCoefficient(doc, 'ADJ', 'NOUN')
        if noun_noun + adv_noun != 0:
            return (verb_noun + verb_adv) / (noun_noun + adv_noun)
        else:
            return 0

    # Коэффицент логичной связности
    def logicalConnectionCoefficient(self, doc):
        all_token_count = self.wordCount(doc)
        service_word_count = 0
        for token in doc.tokens:
            if (
                    token.pos == 'ADP' or token.pos == 'PART' or token.pos == 'CONJ' or token.pos == 'CCONJ' or token.pos == 'INTJ' or token.pos == 'SCONJ'):
                service_word_count = service_word_count + 1
        return service_word_count / (3 * all_token_count)

    def lexicalRichnessCoefficient(self, doc, method='voc-D'):
        list_of_tokens = []
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
            list_of_tokens.append(token.lemma)
        lex = LexicalRichness(list_of_tokens, preprocessor=None, tokenizer=None)
        if method == 'TTR':
            return lex.ttr
        if method == 'RTTR':
            return lex.rttr
        if method == 'CTTR':
            return lex.cttr
        if method == 'MSTTR':
            return lex.msttr(segment_window=25)
        if method == 'MATTR':
            return lex.mattr(window_size=25)
        if method == 'MTLD':
            return lex.mtld(threshold=0.72)
        if method == 'HD-D':
            return lex.hdd(draws=42)
        if method == 'voc-D':
            return lex.vocd(ntokens=40, within_sample=100, iterations=3)
        if method == 'Herdan':
            return lex.Herdan
        if method == 'Summer':
            return lex.Summer
        if method == 'Dugast':
            return lex.Dugast
        if method == 'Maas':
            return lex.Maas
        if method == 'YuleK':
            return lex.yulek
        if method == 'YuleI':
            return lex.yulei
        if method == 'HerdanVm':
            return lex.herdanvm
        if method == 'SimpsonD':
            return lex.simpsond


class MLPSearch(nn.Module):
    def __init__(self,
                 in_features,
                 num_classes,
                 n_layers_out_features,
                 dropout):
        super().__init__()
        layers = []

        input_dim = in_features
        for output_dim in n_layers_out_features:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ModelCompilation(pl.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 metrics: dict,
                 loss_function,
                 optimizer: torch.optim,
                 learning_rate: float):
        super().__init__()
        self.model = model
        self.metrics = metrics
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.save_hyperparameters(logger=False)

    def forward(self, x):
        pred = self.model.forward(x)
        return pred

    def configure_optimizers(self):
        train_optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return train_optimizer

    def training_step(self, batch, batch_idx):
        loss, pred, y = self.common_step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, y = self.common_step(batch, batch_idx, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        loss, pred, y = self.common_step(batch, batch_idx, 'test')
        return loss

    def common_step(self, batch, batch_idx, stage):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_function(pred, y)
        if (stage == 'test') or (stage == 'val'):
            on_step = False
        else:
            on_step = True

        [self.log(stage + '_' + metric_name, metric(pred, y), on_step=on_step, on_epoch=True, prog_bar=True,
                  logger=True) for metric_name, metric in self.metrics.items()]
        self.log(stage + '_' + 'loss', loss, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)
        return loss, pred, y


def load_pretrained_model(model_params, check_point=None):
    num_classes = 5
    task = 'multiclass'
    metrics = {'accuracy': torchmetrics.Accuracy(task=task, num_classes=num_classes)}
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam

    num_classes = 5
    in_features = model_params['n_best_features']
    learning_rate = model_params['learning_rate']
    dropout = model_params['dropout']
    output_dims = []
    for key, value in model_params.items():
        if 'n_units_l' in key:
            output_dims.append(value)

    network = MLPSearch(in_features, num_classes, output_dims, dropout)
    model = ModelCompilation(network, metrics, loss_function, optimizer, learning_rate)
    if check_point:
        model = model.load_from_checkpoint(check_point)
    return model


def mlpPredict(model, text, feature_extraction, scaler, features_indexes, features):
    features_indexes = features_indexes[0:features]
    text_features = feature_extraction.fromTextToVector(text)[None, :]
    text_features = scaler.transform(text_features)
    text_features = torch.tensor(text_features[:, features_indexes]).to(torch.float32)

    model.eval()
    logits = model.forward(text_features)
    proba = torch.nn.functional.softmax(logits.data, dim=1)
    return proba


def predict(text):
    with open('models/mlp_parameters.pkl', 'rb') as file:
        model_params = pickle.load(file)

    scaler = joblib.load('models/mlp_scaler.pkl')
    checkpoint = 'models/model_mlp_0_93.ckpt'
    network = load_pretrained_model(model_params, checkpoint)
    feature_extraction = FeatureExtractionModule()
    features_indexes = [4, 0, 12, 5, 7, 3, 11, 10, 2, 18, 19, 14, 6, 9, 15, 21, 8, 20, 13, 1, 16, 17]
    return mlpPredict(network, text, feature_extraction, scaler, features_indexes, model_params['n_best_features'])
