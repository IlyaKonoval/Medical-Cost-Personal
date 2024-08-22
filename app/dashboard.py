import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from joblib import load
from sklearn.model_selection import train_test_split
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import warnings

# Игнорирование конкретных предупреждений
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# Определение путей
relative_path_csv = "preprocessed_insurance.csv"
absolute_path_csv = os.path.abspath(relative_path_csv)
df = pd.read_csv(absolute_path_csv)

relative_path_model = "final_model_pipeline.joblib"
absolute_path_model = os.path.abspath(relative_path_model)
model = load(absolute_path_model)

# Обработка данных
X = df.drop('charges', axis=1)
y = (df['charges'] > 11012).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

categorical = ['sex', 'smoker', 'region']
numeric_features = ['age', 'bmi', 'children']

ct = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical),
    ('scaling', MinMaxScaler(), numeric_features)
])

X_train_transformed = ct.fit_transform(X_train)
X_test_transformed = ct.transform(X_test)

ohe_feature_names = ct.named_transformers_['ohe'].get_feature_names_out(categorical)
new_columns = list(ohe_feature_names) + numeric_features

X_test_transformed = pd.DataFrame(X_test_transformed, columns=new_columns)

# Создание ExplainerDashboard
explainer = ClassifierExplainer(model,
                                X_test_transformed,
                                y_test,
                                labels=['Low Charges', 'High Charges'],
                                cats=ohe_feature_names)

db = ExplainerDashboard(explainer)
db.to_yaml("dashboard.yaml", explainerfile="explainer.dill", dump_explainer=True)
