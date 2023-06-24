import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from lexicalrichness import LexicalRichness
from natasha import Segmenter, Doc, NewsEmbedding, NewsMorphTagger, MorphVocab
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import statistic_mlp


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
            'category').cat.codes  # Style from string to int categories
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

    # Подсчёт количества знаков препинаний в тексте
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


def plot_and_print_mi_scores(X_data, y_data):
    mi_scores = mutual_info_classif(X_data, y_data)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X_data.columns)
    mi_scores = mi_scores.sort_values(ascending=True)
    ax = mi_scores.plot(kind='barh', figsize=(12, 8), title='Mutual Information Scores', legend=False)
    ax.bar_label(ax.containers[0], label_type='edge')
    return mi_scores.index[::-1]


class BestEstimatorExtraction:
    def __init__(self,
                 x_data,
                 y_data):
        self.x = x_data
        self.y = y_data

        self.log_reg_model = LogisticRegression()
        self.log_reg_best_parameters_for_all_features = {'C': [50], 'max_iter': [1000], 'penalty': ['l1'],
                                                         'random_state': [42], 'solver': ['saga']}
        self.log_reg_best_parameters = {'C': [50], 'max_iter': [100], 'penalty': ['l2'], 'random_state': [42],
                                        'solver': ['sag']}
        self.log_features = [True, True, True, False, True, True, False, True, True, True, True, True, True, True, True,
                             True, True, True, True, True, True, True]
        self.log_reg_search_parameters = {"estimator__max_iter": [100, 500, 1000, 5000],
                                          "estimator__C": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1,
                                                           5, 10, 50, 100],
                                          "estimator__penalty": ["elasticnet", "l1", "l2"],
                                          "estimator__solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                                          "estimator__random_state": [42]}

        self.dec_tree_model = DecisionTreeClassifier()
        self.dec_tree_best_parameters_for_all_features = {'ccp_alpha': [0.01], 'criterion': ['entropy'],
                                                          'max_depth': [8], 'max_features': [0.8],
                                                          'min_samples_leaf': [1], 'min_samples_split': [2],
                                                          'random_state': [42]}
        self.dec_tree_best_parameters = {'ccp_alpha': [0.0], 'criterion': ['entropy'], 'max_depth': [8],
                                         'max_features': [0.8], 'min_samples_leaf': [5], 'min_samples_split': [2],
                                         'random_state': [42]}
        self.dec_tree_features = [True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                  True, True, True, False, True, True, True, True]
        self.dec_tree_search_parameters = {'estimator__max_features': ['sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
                                           'estimator__ccp_alpha': [0.1, .01, .001, .0],
                                           'estimator__min_samples_leaf': [1, 5, 8, 11],
                                           'estimator__min_samples_split': [2, 3, 5, 7, 9],
                                           'estimator__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                           'estimator__criterion': ['gini', 'entropy', 'log_loss'],
                                           'estimator__random_state': [42]}

        self.random_forest_model = RandomForestClassifier()
        self.random_forest_best_parameters_for_all_features = {'ccp_alpha': [0.001], 'criterion': ['entropy'],
                                                               'max_depth': [None], 'max_features': ['sqrt'],
                                                               'min_samples_leaf': [1], 'min_samples_split': [2],
                                                               'n_estimators': [1000], 'random_state': [42]}
        self.random_forest_best_parameters = {'ccp_alpha': [0.001], 'criterion': ['entropy'], 'max_depth': [9],
                                              'max_features': ['sqrt'], 'min_samples_leaf': [1],
                                              'min_samples_split': [2], 'n_estimators': [500], 'random_state': [42]}
        self.random_forest_features = [True, True, True, True, True, True, True, True, True, True, True, True, True,
                                       True, True, True, True, True, True, True, True, True]
        self.random_forest_search_parameters = {'estimator__n_estimators': [100, 250, 500, 1000],
                                                'estimator__max_features': ['sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
                                                'estimator__ccp_alpha': [0.1, .01, .001, 0.0],
                                                'estimator__min_samples_leaf': [1, 5, 8, 11],
                                                'estimator__min_samples_split': [2, 3, 5, 7, 9],
                                                'estimator__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                'estimator__criterion': ['gini', 'entropy', 'log_loss'],
                                                'estimator__random_state': [42]}

        self.svc_model = SVC()
        self.svc_best_parameters_for_all_features = {'C': [10], 'gamma': ['scale'], 'kernel': ['rbf'],
                                                     'random_state': [42]}
        self.svc_best_parameters = {'C': [10], 'gamma': ['scale'], 'kernel': ['linear'], 'random_state': [42]}
        self.svc_features = [True, True, True, True, True, True, False, True, True, False, True, True, False, False,
                             True, True, True, True, True, True, True, True]
        self.svc_search_parameters = {'estimator__C': [0.1, 1, 10, 100, 1000],
                                      'estimator__gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001],
                                      'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                      'estimator__random_state': [42]}

        self.knn_model = KNeighborsClassifier()
        self.knn_best_parameters = {'n_neighbors': [17], 'p': [1], 'weights': ['distance']}
        self.knn_features = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                             True, True, True, True, True, True, True]

        self.xgboost_model = xgb.XGBClassifier(objective='multi:softmax', num_class=5, tree_method='hist')
        self.xgboost_best_parameters = {'booster': ['gbtree'], 'n_estimators': [500], 'learning_rate': [0.05],
                                        'max_depth': [4], 'subsample': [0.4], 'reg_alpha': [0.1], 'reg_lambda': [1.0],
                                        "random_state": [42]}
        self.xgboost_features = [True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                 True, True, True, True, True, True, True, True]
        self.xgboost_search_parameters = {'estimator__booster': ['gbtree'],
                                          'estimator__n_estimators': [100, 250, 500, 1000],
                                          'estimator__learning_rate': [0.05, 0.1, 0.2, 0.3],
                                          'estimator__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                          'estimator__subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                          'estimator__reg_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                          'estimator__reg_lambda': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                                    1],
                                          'estimator__random_state': [42]
                                          }

        self.catboost_model = CatBoostClassifier(verbose=False)
        self.catboost_best_parameters = {'n_estimators': [1000], 'learning_rate': [0.2], 'max_depth': [7],
                                         'l2_leaf_reg': [3], 'random_strength': [0.2], 'bagging_temperature': [1.0],
                                         'border_count': [254], 'random_state': [42]}
        self.catboost_features = [True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                  True, True, True, True, True, True, True, True]
        self.catboost_search_parameters = {'n_estimators': [100, 250, 500, 1000],
                                           'learning_rate': [0.05, 0.1, 0.2, 0.3],
                                           'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                           'l2_leaf_reg': [1, 3, 5, 10, 100],
                                           'random_strength': [0.2, 0.5, 0.8, 1.1, 1.4],
                                           'bagging_temperature': [0.03, 0.09, 0.25, 0.75, 1.0],
                                           'border_count': [254],
                                           'random_state': [42]}

        self.lgbm_model = lgb.LGBMClassifier(objective='multiclass', n_jobs=-1)
        self.lgbm_best_parameters = {'n_estimators': [250], 'max_depth': [None], 'learning_rate': [0.3],
                                     'subsample': [0.4], 'reg_alpha': [0.1], 'reg_lambda': [0.1], 'num_leaves': [32],
                                     'is_unbalance': [False], 'boost_from_average': [False], "random_state": [42]}
        self.lgbm_features = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                              True, True, True, True, True, True, True]
        self.lgbm_search_parameters = {'estimator__n_estimators': [100, 250, 500, 1000],
                                       'estimator__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                       'estimator__learning_rate': [0.05, 0.1, 0.2, 0.3],
                                       'estimator__subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                       'estimator__reg_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                       'estimator__reg_lambda': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                       'estimator__num_leaves': [32],
                                       'estimator__is_unbalance': [False],
                                       'estimator__boost_from_average': [False],
                                       "estimator__random_state": [42]}


def modelPredict(model, text, feature_extraction, scaler, features):
    text_features = feature_extraction.fromTextToVector(text)[None, :]
    text_features = scaler.transform(text_features)
    text_features = text_features[:, features]
    proba = model.predict_proba(text_features)
    return proba


def predict(text, mode):
    scaler = joblib.load('scaler.pkl')
    feature_extraction = FeatureExtractionModule()
    bee = BestEstimatorExtraction(None, None)
    proba = None

    if mode == "Logistic Regression":
        log_reg = joblib.load('models/log_reg.pkl')
        log_features = bee.log_features
        proba = modelPredict(log_reg, text, feature_extraction, scaler, log_features)
    elif mode == "Decision Tree":
        dec_tree = joblib.load('models/dec_tree.pkl')
        dec_tree_features = bee.dec_tree_features
        proba = modelPredict(dec_tree, text, feature_extraction, scaler, dec_tree_features)
    elif mode == "Random Forest":
        random_forest = joblib.load('models/random_forest.pkl')
        random_forest_features = bee.random_forest_features
        proba = modelPredict(random_forest, text, feature_extraction, scaler, random_forest_features)
    elif mode == "SVC":
        svc = joblib.load('models/svc.pkl')
        svc_features = bee.svc_features
        proba = modelPredict(svc, text, feature_extraction, scaler, svc_features)
    elif mode == "K-Nearest Neighbors":
        knn = joblib.load('models/knn.pkl')
        knn_features = bee.knn_features
        proba = modelPredict(knn, text, feature_extraction, scaler, knn_features)
    elif mode == "XGBoost":
        xgboost = joblib.load('models/xgboost.pkl')
        xgboost_features = bee.xgboost_features
        proba = modelPredict(xgboost, text, feature_extraction, scaler, xgboost_features)
    elif mode == "CatBoost":
        catboost = joblib.load('models/catboost.pkl')
        catboost_features = bee.catboost_features
        proba = modelPredict(catboost, text, feature_extraction, scaler, catboost_features)
    elif mode == "LightGBM":
        lgbm = joblib.load('models/lgbm.pkl')
        lgbm_features = bee.lgbm_features
        proba = modelPredict(lgbm, text, feature_extraction, scaler, lgbm_features)
    elif mode == "MLP":
        proba = statistic_mlp.predict(text)

    return proba
