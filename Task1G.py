import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt

#Loading the dataset
titanic = pd.read_csv('titanic/train.csv')

#preprocessing steps
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)
titanic.drop('Cabin', axis=1, inplace=True)

#drop unwanted columns
titanic.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

titanic['Sex'] = LabelEncoder().fit_transform(titanic['Sex'])
titanic = pd.get_dummies(titanic, columns=['Embarked'], drop_first=True)

#Scaling numerical features
scaler = StandardScaler()
titanic[['Age', 'Fare']] = scaler.fit_transform(titanic[['Age', 'Fare']])

#Spliting the data
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

#define a grid of parameters to search through
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5,6, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)
dt_grid_search.fit(X, y)

#search best fine tuning model
best_dt = dt_grid_search.best_estimator_
print(f"Best Decision Tree Parameters: {dt_grid_search.best_params_}")

#Plotting the Decision Tree Map
plt.figure(figsize=(20, 12))
plot_tree(best_dt, feature_names=X.columns.tolist(), class_names=['Not Survived', 'Survived'], filled=True, rounded=True)
plt.title("Fine Tuned Decision Tree")
plt.savefig("decision_tree.png", dpi=300)
print("Decision tree plot saved as 'decision_tree.png'")

#five fold cross-validation
dt_scores = cross_val_score(best_dt, X, y, cv=5, scoring='accuracy')
print(f"Fine-Tuned Decision Tree 5-Fold CV Mean Accuracy: {dt_scores.mean():.4f}\n")

#Defining a parameter grid
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 2, 4]
}

#Initialize the grid search
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X, y)

#Getting the best fine tuning model
best_rf = rf_grid_search.best_estimator_
print(f"Best Random Forest Parameters: {rf_grid_search.best_params_}")

#Applying cross-validation
rf_scores = cross_val_score(best_rf, X, y, cv=5, scoring='accuracy')
print(f"Fine-Tuned Random Forest 5-Fold CV Mean Accuracy: {rf_scores.mean():.4f}\n")

#model accuracy comparision
print("*****Comparison and Conclusion******")
print(f"Decision Tree: {dt_scores.mean():.4f}")
print(f"Random Forest: {rf_scores.mean():.4f}\n")

if rf_scores.mean() > dt_scores.mean():
    print("Random Forest is the better than Decision Tree")
else:

    print("Decision Tree is the better than Random Forest")
