import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings

warnings.filterwarnings("ignore")

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Load data
train_data = pd.read_csv(r"C:\Users\Asus\Downloads\train.csv")
test_data = pd.read_csv(r"C:\Users\Asus\Downloads\test.csv")
test_data['Survived'] = np.nan
combined = pd.concat([train_data, test_data], axis=0)
print(f"Combined dataset shape: {combined.shape}")

print("\nMissing values in the dataset:")
print(combined.isnull().sum())

# Extract title
def extract_title(name):
    return name.split(',')[1].split('.')[0].strip()

combined['Title'] = combined['Name'].apply(extract_title)
title_counts = combined['Title'].value_counts()
rare_titles = title_counts[title_counts < 10].index
combined['Title'] = combined['Title'].apply(lambda x: 'Rare' if x in rare_titles else x)

combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)

combined['Deck'] = combined['Cabin'].str.slice(0, 1).fillna('U')

# Impute missing Age using Linear Regression
def impute_age(df):
    df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
    age_df = df[['Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
    known_age = age_df[age_df['Age'].notnull()]
    unknown_age = age_df[age_df['Age'].isnull()]

    X = known_age.drop('Age', axis=1)
    y = known_age['Age']
    regressor = LinearRegression().fit(X, y)
    predicted_ages = regressor.predict(unknown_age.drop('Age', axis=1))
    df.loc[df['Age'].isnull(), 'Age'] = predicted_ages
    return df

print("\nImputing missing ages using Linear Regression...")
combined = impute_age(combined)

# Impute missing Embarked using KNN
embarked_data = pd.get_dummies(combined[['Pclass', 'Sex', 'Fare']])
embarked_data['Embarked'] = combined['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
knn_imputer = KNNImputer(n_neighbors=5)
embarked_imputed = knn_imputer.fit_transform(embarked_data)
embarked_mapping = {0: 'S', 1: 'C', 2: 'Q'}
combined.loc[combined['Embarked'].isnull(), 'Embarked'] = [
    embarked_mapping[int(val)] for val in embarked_imputed[combined['Embarked'].isnull(), -1]
]

# Feature engineering
combined['FarePerPerson'] = combined['Fare'] / combined['FamilySize']

# Boxplot: FarePerPerson vs Pclass & Survived
plt.figure(figsize=(12, 6))
sns.boxplot(x='Pclass', y='FarePerPerson', hue='Survived', data=combined[combined['Survived'].notnull()])
plt.title('Fare Per Person by Pclass and Survival Status')
plt.xlabel('Passenger Class')
plt.ylabel('Fare Per Person')
plt.tight_layout()
plt.show()

# T-test for FarePerPerson
train_survived = combined[combined['Survived'] == 1]['FarePerPerson'].dropna()
train_not_survived = combined[combined['Survived'] == 0]['FarePerPerson'].dropna()
t_stat, p_value = stats.ttest_ind(train_survived, train_not_survived, equal_var=False)
print(f"\nT-test for FarePerPerson between survived and non-survived groups:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("There is a significant difference in fare per person between survivors and non-survivors")
else:
    print("There is no significant difference in fare per person between survivors and non-survivors")

# Prepare data
train = combined[combined['Survived'].notnull()].copy()
test = combined[combined['Survived'].isnull()].copy()

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
            'Title', 'FamilySize', 'IsAlone', 'Deck', 'FarePerPerson']
X = train[features]
y = train['Survived'].astype(int)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numerical_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'FarePerPerson']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'IsAlone']
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Model pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Grid Search
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}
print("\nPerforming GridSearchCV for hyperparameter tuning...")
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)[:, 1]
print("\nValidation set metrics:")
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_pred):.4f}")
print(f"Recall: {recall_score(y_val, y_pred):.4f}")
print(f"F1-score: {f1_score(y_val, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_val, y_prob):.4f}")

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    best_model, X, y, cv=cv,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    return_train_score=False
)
print("\n5-fold Cross-Validation Results:")
for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    print(f"{metric.capitalize()}: {cv_results[f'test_{metric}'].mean():.4f} (±{cv_results[f'test_{metric}'].std():.4f})")

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Feature importance
feature_names = []
for name, transformer, features in preprocessor.transformers_:
    if name == 'num':
        feature_names.extend(features)
    else:
        for feature in features:
            if hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                cat_features = transformer.named_steps['onehot'].get_feature_names_out([feature])
                feature_names.extend(cat_features)
            else:
                cat_values = combined[feature].unique()
                for val in cat_values:
                    if isinstance(val, str):
                        feature_names.append(f"{feature}_{val}")

importances = best_model.named_steps['classifier'].feature_importances_
sorted_indices = np.argsort(importances)[::-1]

# New bar plot for top features
top_n = min(15, len(importances))
top_features = [feature_names[i] if i < len(feature_names) else f'feature_{i}' for i in sorted_indices[:top_n]]
top_importances = importances[sorted_indices][:top_n]

plt.figure(figsize=(12, 6))
sns.barplot(x=top_features, y=top_importances, palette="viridis")
plt.xticks(rotation=45, ha='right')
plt.ylabel('Importance Score')
plt.title('Top 15 Most Important Features')
plt.tight_layout()
plt.show()

# Final prediction
print("\nxong")
test_predictions = best_model.predict(test[features])
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('submission.csv', index=False)
print("\nTest predictions saved to 'submission.csv'")
