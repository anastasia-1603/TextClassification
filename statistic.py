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
from statistic_mlp import FeatureExtractionModule


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
    scaler = joblib.load('models/scaler.pkl')
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
