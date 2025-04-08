import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import os

#Load the Titanic dataset using Seaborn
try:
    titanic = sns.load_dataset('titanic')
except Exception as e:
    print("Error loading Titanic dataset. Please ensure Seaborn's built-in datasets are available.")
    print("Alternatively, download the Titanic dataset from a reliable source and load it manually.")
    raise e
titanic.head()

#Select relevant features and the target
titanic.count()

#Task  1. How balanced are the classes?
y = titanic['survived']  # Define the target variable
print(y.value_counts())
# Handle missing values in categorical features before encoding
# Define the feature matrix X by selecting relevant features
X = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]  # Example features

# Handle missing values in categorical features before encoding
X['sex'] = X['sex'].fillna('missing')
X['embarked'] = X['embarked'].fillna('missing')
X = pd.get_dummies(X, columns=['sex', 'embarked'], drop_first=True)  # Encode categorical features

#Task  2. Split the data into training and testing sets
# Define the feature matrix X by selecting relevant features
X = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]  # Example features
X = pd.get_dummies(X, columns=['sex', 'embarked'], drop_first=True)  # Encode categorical features

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#Define preprocessing transformers for numerical and categorical features
numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

#Define separate preprocessing pipelines for both feature types
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#Combine the transformers into a single column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Ensure the cache directory exists
cache_directory = 'cache_directory'
os.makedirs(cache_directory, exist_ok=True)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
], memory=cache_directory)

#Define a parameter grid
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

import logging

# Configure logging to write output to a file
logging.basicConfig(filename='grid_search.log', level=logging.INFO, format='%(asctime)s - %(message)s')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=1)

# Cross-validation method (already defined earlier, removing redundancy)

#Task  3. Train the pipeline model
model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)
model.fit(X_train, y_train)

#Task 4. Get the model predictions from the grid search estimator on the unseen data
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, normalize='true')
print(cm)

#Task  5. Plot the confusion matrix
# Enter your code here:
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.title('Titanic Confusion Matrix', fontsize=14)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.2f',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])

# Set the title and axis labels
plt.title('🧊 Titanic Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)

plt.title('Titanic Confusion Matrix', fontsize=14)
plt.tight_layout()
plt.show()

# Feature importances
# Fit the OneHotEncoder with the training data
model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].fit(X_train[categorical_features])

# Get the feature names after fitting
model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)

feature_importances = model.best_estimator_['classifier'].feature_importances_

model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].fit(X_train[categorical_features])
feature_names_out = model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names = numerical_features + list(model.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

# Display the feature importances in a bar plot

# Create a DataFrame for feature importances

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis() 
plt.title('🔍 Most Important Features for Predicting Survival on the Titanic')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Print test set accuracy
test_score = model.score(X_test, y_test)
print(f"\n✅ Test set accuracy: {test_score:.2%}")

#Task  6. These are interesting results to consider.

# Replace RandomForestClassifier with LogisticRegression
pipeline.set_params(classifier=LogisticRegression(random_state=42))

# update the model's estimator to use the new pipeline
model.estimator = pipeline

# Define a new grid with Logistic Regression parameters
param_grid = {
    # 'classifier__n_estimators': [50, 100],
    # 'classifier__max_depth': [None, 10, 20],
    # 'classifier__min_samples_split': [2, 5],
    # Adding multiple solvers for Logistic Regression to explore different optimization methods
    'classifier__solver': ['liblinear', 'lbfgs', 'saga'],
    'classifier__penalty': ['l1', 'l2', 'elasticnet'],  # Note: 'l1' and 'elasticnet' are not supported by all solvers
    'classifier__class_weight': [None, 'balanced']
}

model.param_grid = param_grid

# Fit the updated pipeline with Logistic Regression
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

#Task  7. Display the clasification report for the new model and compare the results to your previous model.
print(classification_report(y_test, y_pred))

#Task  8. Display the confusion matrix for the new model and compare the results to your previous model.
# Generate the confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()

# Extract the logistic regression feature coefficients and plot their magnitude in a bar chart.
if hasattr(model.best_estimator_.named_steps['classifier'], 'coef_'):
    coefficients = model.best_estimator_.named_steps['classifier'].coef_[0]
# Combine numerical and categorical feature names
numerical_feature_names = model.best_estimator_.named_steps['preprocessor'].transformers_[0][2]
# Fit the OneHotEncoder with the training data
model.best_estimator_.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].fit(X_train[categorical_features])

# Get the feature names after fitting
categorical_feature_names = (model.best_estimator_.named_steps['preprocessor']
                                     .named_transformers_['cat']
                                     .named_steps['onehot']
                                     .get_feature_names_out(categorical_features)
                            )
feature_names = list(numerical_feature_names) + list(categorical_feature_names)
# Check pandas version and sort accordingly
if pd.__version__ >= '1.1.0':
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', ascending=False, key=abs)  # Sort by absolute values
else:
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    importance_df['AbsCoefficient'] = importance_df['Coefficient'].abs()
    importance_df = importance_df.sort_values(by='AbsCoefficient', ascending=False).drop(columns=['AbsCoefficient'])
# Create a DataFrame for the coefficients
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False, key=abs)  # Sort by absolute values
# Print test score (already printed earlier, removing redundancy)
plt.barh(importance_df['Feature'], importance_df['Coefficient'].abs(), color='skyblue')
plt.gca().invert_yaxis()
plt.title('Feature Coefficient magnitudes for Logistic Regression model')
plt.xlabel('Coefficient Magnitude')
plt.show()

# Print test score
test_score = model.best_estimator_.score(X_test, y_test)
print(f"\nTest set accuracy: {test_score:.2%}")
