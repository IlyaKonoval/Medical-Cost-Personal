import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from joblib import dump

# Игнорирование конкретных предупреждений
warnings.filterwarnings(action='ignore', category=UserWarning, message='.*')

if __name__ == '__main__':
    # Загрузка данных
    df = pd.read_csv('preprocessed_insurance.csv')

    # Подготовка данных
    X = df[['age', 'children', 'bmi']]
    y = df[['charges']]
    y_class = (y > 11012).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=52)
    y_train = y_train.values.ravel()

    # Наивный Баес
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_gnb = gnb.predict(X_test)
    print("Наивный Баес:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gnb))
    print("Accuracy:", accuracy_score(y_test, y_pred_gnb))
    print("F1-score:", f1_score(y_test, y_pred_gnb))

    # KNN
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    print("\nKNN:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
    print("Accuracy:", accuracy_score(y_test, y_pred_knn))
    print("F1-score:", f1_score(y_test, y_pred_knn))

    # Логистическая регрессия
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print("\nЛогистическая регрессия:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
    print("Accuracy:", accuracy_score(y_test, y_pred_lr))
    print("F1-score:", f1_score(y_test, y_pred_lr))

    # Нормализация данных
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Определение параметров для GridSearch
    params_knn = {'n_neighbors': np.arange(1, 31),
                  'weights': ['uniform', 'distance'],
                  'p': [1, 2]}
    params_lr_l1 = {'C': np.logspace(-3, 3, 7),
                    'penalty': ['l1'],
                    'solver': ['liblinear']}
    params_lr_l2 = {'C': np.logspace(-3, 3, 7),
                    'penalty': ['l2'],
                    'solver': ['lbfgs']}
    params_NB = {'var_smoothing': np.logspace(0, -9, num=100)}

    # GridSearchCV для моделей
    gs_knn = GridSearchCV(KNeighborsClassifier(), params_knn, scoring='f1', cv=3, n_jobs=-1, verbose=3)
    gs_gnb = GridSearchCV(GaussianNB(), params_NB, scoring='f1', cv=3, n_jobs=-1, verbose=3)
    gs_lr_l1 = GridSearchCV(LogisticRegression(), params_lr_l1, scoring='f1', cv=3, n_jobs=-1, verbose=3)
    gs_lr_l2 = GridSearchCV(LogisticRegression(), params_lr_l2, scoring='f1', cv=3, n_jobs=-1, verbose=3)

    # Наивный Баес с гиперпараметрами
    gs_gnb.fit(X_train_scaled, y_train)
    gnb_best = GaussianNB(**gs_gnb.best_params_)
    gnb_best.fit(X_train_scaled, y_train)
    y_pred_gnb_best = gnb_best.predict(X_test_scaled)
    print("\nНаивный Баес с гиперпараметрами:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gnb_best))
    print("Accuracy:", accuracy_score(y_test, y_pred_gnb_best))
    print("F1-score:", f1_score(y_test, y_pred_gnb_best))

    # KNN с гиперпараметрами
    gs_knn.fit(X_train_scaled, y_train)
    knn_model_best = KNeighborsClassifier(**gs_knn.best_params_)
    knn_model_best.fit(X_train_scaled, y_train)
    y_pred_knn_best = knn_model_best.predict(X_test_scaled)
    print("\nKNN с гиперпараметрами:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn_best))
    print("Accuracy:", accuracy_score(y_test, y_pred_knn_best))
    print("F1-score:", f1_score(y_test, y_pred_knn_best))

    # Логистическая регрессия с гиперпараметрами
    gs_lr_l1.fit(X_train_scaled, y_train)
    gs_lr_l2.fit(X_train_scaled, y_train)
    if gs_lr_l1.best_score_ > gs_lr_l2.best_score_:
        lr_best = gs_lr_l1.best_estimator_
    else:
        lr_best = gs_lr_l2.best_estimator_

    lr_best.fit(X_train_scaled, y_train)
    y_pred_lr_best = lr_best.predict(X_test_scaled)

    print("\nЛогистическая регрессия с гиперпараметрами:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr_best))
    print("Accuracy:", accuracy_score(y_test, y_pred_lr_best))
    print("F1-score:", f1_score(y_test, y_pred_lr_best))

    # Полная модель с категориальными признаками и гиперпараметрами
    X = df.drop('charges', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=52)
    y_train = y_train.values.ravel()

    categorical = ['sex', 'smoker', 'region']
    numeric_features = ['age', 'bmi', 'children']

    ct = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical),
        ('scaling', MinMaxScaler(), numeric_features)
    ])

    pipe = Pipeline([
        ('transformer', ct),
        ('model', KNeighborsClassifier())
    ])

    params = {'model__n_neighbors': np.arange(1, 31),
              'model__weights': ['uniform', 'distance'],
              'model__p': [1, 2]}

    gs = GridSearchCV(pipe, params, scoring='f1', cv=3, n_jobs=-1, verbose=2)
    gs.fit(X_train, y_train)

    pipe_best = gs.best_estimator_
    pred_pipe = pipe_best.predict(X_test)

    print("\nПолная модель с категориальными признаками и гиперпараметрами:")
    print("Accuracy:", accuracy_score(y_test, pred_pipe))
    print("F1-score:", f1_score(y_test, pred_pipe))

    dump(pipe_best, 'final_model_pipeline.joblib')
