import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

root_dir = Path(__file__).resolve().parent
train_path = root_dir / "data" / "train.csv"
test_path = root_dir / "data" / "test.csv"

if not train_path.exists():
    raise FileNotFoundError(f"Training file not found: {train_path}")

train_df = pd.read_csv(train_path)

test_df = None
if test_path.exists():
    test_df = pd.read_csv(test_path)
else:
    print(f"Warning: {test_path} not found. Skipping test data processing.")

train_df.head()
sns.countplot(x='Survived', data=train_df)
plt.show()
train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

rare_titles = ['Lady', 'Countess','Capt', 'Col',
               'Don', 'Dr', 'Major', 'Rev',
               'Sir', 'Jonkheer', 'Dona']

train_df['Title'] = train_df['Title'].replace(rare_titles, 'Rare')
train_df['Title'] = train_df['Title'].replace({
    'Mlle':'Miss',
    'Ms':'Miss',
    'Mme':'Mrs'
})
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)
train_df['HasCabin'] = train_df['Cabin'].notnull().astype(int)
train_df['Age'] = train_df.groupby('Title')['Age']\
    .transform(lambda x: x.fillna(x.median()))
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
if test_df is not None:
    test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
train_df.drop(columns=['Cabin'], inplace=True)
categorical_cols = ['Sex', 'Embarked', 'Title']

train_df = pd.get_dummies(train_df,
                          columns=categorical_cols,
                          drop_first=True)
drop_cols = ['PassengerId', 'Name', 'Ticket']

X = train_df.drop(columns=drop_cols + ['Survived'])
y = train_df['Survived']
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

rf_model.fit(X_train, y_train)
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)
preds = xgb_model.predict(X_valid)
probs = xgb_model.predict_proba(X_valid)[:,1]

print("Accuracy:", accuracy_score(y_valid, preds))
print("ROC-AUC:", roc_auc_score(y_valid, probs))

print(classification_report(y_valid, preds))
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance.head(10)
)

plt.title("Top Features")
plt.show()
explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(X_valid)

shap.summary_plot(shap_values, X_valid)
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_valid.iloc[0]
)
model_dir = root_dir / "models"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "titanic_model.pkl"
joblib.dump(xgb_model, model_path)
model = joblib.load(model_path)