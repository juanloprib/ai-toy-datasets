import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

columns_names = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B", 
    "LSTAT",
    "MEDV"
]

boston_df = pd.read_csv("boston.csv", sep=r"\s+", names=columns_names)

# Basic Analysis

print(boston_df.head())
print(boston_df.describe())
print("skew", boston_df.skew())

boston_df.hist(bins=20, figsize=(12, 12))
plt.show()

fig, axs = plt.subplots(4, 4, figsize=(12, 12))
for i in range(0, len(columns_names) - 1):
    axs[int(i / 4), i % 4].scatter(boston_df[columns_names[i]], boston_df["MEDV"])
    axs[int(i / 4), i % 4].set_title(columns_names[i])
plt.show()

# Preprocessing

boston_df['CRIM'] = np.log1p(boston_df['CRIM'])
boston_df['CHAS'] = np.log1p(boston_df['CHAS'])
boston_df['ZN'] = np.log1p(boston_df['ZN'])
boston_df['B'] = np.square(boston_df['B']) 
# boston_df.drop(["B", "LSTAT"], axis=1, inplace=True)

X = boston_df.iloc[:, 0:-1]
y = boston_df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Linear Regression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
pipe.fit(X_train, y_train)
scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
print("Linear Regression Cross Validation", scores)
print("Linear Regression", pipe.score(X_test, y_test))


# Ridge Regression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])
pipe.fit(X_train, y_train)
print("Ridge", pipe.score(X_test, y_test))


# SVR

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVR(kernel='linear'))
])
pipe.fit(X_train, y_train)
print("SVR", pipe.score(X_test, y_test))


# Random Forest

pipe = Pipeline([
    ("model", RandomForestRegressor(n_estimators=100))
])
pipe.fit(X_train, y_train)
print("Random Forest", pipe.score(X_test, y_test))


# Random Forest: Best Hyperparameters

pipe = Pipeline([
    ("model", RandomForestRegressor(n_estimators=100))
])
param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [None, 5, 10, 15, 20, 25],
    "model__min_samples_split": [2, 5, 10],
    "model__max_features": ["sqrt", "log2"],
    "model__min_samples_leaf": [1, 2, 5]
}
grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
print("Random Forest: Best Hyperparameters", grid.best_params_, grid.score(X_test, y_test))


# Graph True vs Predicted

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

plt.plot(y_test, y_test, label="True")
plt.scatter(y_test, y_pred, label="Predicted")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.legend()
plt.show()