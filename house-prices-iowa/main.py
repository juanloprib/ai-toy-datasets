import pandas as pd
from lib.analyzedata import AnalyzeData
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np


# load

train_data = pd.read_csv('./train.csv')
train_data_features = train_data.drop('SalePrice', axis=1)
train_data_target = np.log1p(train_data['SalePrice'])


# analyze

analyzer = AnalyzeData(train_data_features, train_data_target)
analyzer.basic_info()
analyzer.analyze()


# drop

cols_to_drop = ['Id']
train_data_features.drop(cols_to_drop, axis=1, inplace=True)


# fill

cols_vals_to_fill = {
    'LotFrontage': train_data_features['LotFrontage'].mean(),
    'Alley': 'No alley access',
    'MasVnrType': 'None',
    'MasVnrArea': 0,
    'BsmtQual': 'No basement',
    'BsmtCond': 'No basement',
    'BsmtExposure': 'No basement',
    'BsmtFinType1': 'No basement',
    'BsmtFinType2': 'No basement',
    'Electrical': train_data_features['Electrical'].mode()[0],
    'FireplaceQu': 'No fireplace',
    'GarageType': 'No garage',
    'GarageYrBlt': train_data_features['GarageYrBlt'].min(),
    'GarageFinish': 'No garage',
    'GarageQual': 'No garage',
    'GarageCond': 'No garage',
    'PoolQC': 'No pool',
    'Fence': 'No fence',
    'MiscFeature': 'None'
}
train_data_features.fillna(value=cols_vals_to_fill, inplace=True)


# remove problematic rows

for index, row in train_data_features.iterrows():
    if row['BsmtExposure'] == 'No basement' and row['BsmtCond'] != 'No basement':
        train_data_features.drop(index, inplace=True)
        train_data_target.drop(index, inplace=True)


# ordinal categories

ordinal_mappings = {
    'LotShape': ['IR3', 'IR2', 'IR1', 'Reg'],
    'LandContour': ['Low', 'HLS', 'Bnk', 'Lvl'],
    'Utilities': ['ELO', 'NoSeWa', 'NoSewr', 'AllPub'],
    'LandSlope': ['Sev', 'Mod', 'Gtl'],
    'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtQual': ['No basement', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtCond': ['No basement', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtExposure': ['No basement', 'No', 'Mn', 'Av', 'Gd'],
    'BsmtFinType1': ['No basement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'BsmtFinType2': ['No basement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'CentralAir': ['N', 'Y'],
    'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
    'FireplaceQu': ['No fireplace', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageFinish': ['No garage', 'Unf', 'RFn', 'Fin'],
    'GarageQual': ['No garage', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageCond': ['No garage', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'PoolQC': ['No pool', 'Fa', 'TA', 'Gd', 'Ex'],
}

for col, categories in ordinal_mappings.items():
    ordinal_encoder = OrdinalEncoder(categories=[categories])
    train_data_features[col] = ordinal_encoder.fit_transform(train_data_features[[col]])


# nominal categories

nominal_categories = [
    'MSZoning',
    'Street',
    'Alley',
    'LotConfig',
    'Condition1',
    'Condition2',
    'BldgType',
    'HouseStyle',
    'RoofStyle',
    'RoofMatl',
    'MasVnrType',
    'Foundation',
    'Heating',
    'Electrical',
    'GarageType',
    'PavedDrive',
    'Fence',
    'MiscFeature',
    'SaleType',
    'SaleCondition',
]
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
nominal_encoded = onehot_encoder.fit_transform(train_data_features[nominal_categories])
nominal_encoded = pd.DataFrame(nominal_encoded, columns=onehot_encoder.get_feature_names_out(nominal_categories))
nominal_encoded.index = train_data_features.index
train_data_features.drop(nominal_categories, axis=1, inplace=True)
train_data_features = pd.concat([train_data_features, nominal_encoded], axis=1)


# frequency encoding

cols_to_freq_encode = [
    'Neighborhood',
    'Exterior1st',
    'Exterior2nd',
]
for col in cols_to_freq_encode:
    freqs = train_data_features[col].value_counts(normalize=True)
    train_data_features[col] = train_data_features[col].map(freqs)


# skew

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(train_data_features.skew())
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

high_skew_features = train_data_features.skew()[train_data_features.skew() > 1.2].index
train_data_features[high_skew_features] = np.log1p(train_data_features[high_skew_features])


# analyze again

analyzer.update_data(train_data_features, train_data_target)
analyzer.basic_info()
analyzer.analyze()


# split

X_train, X_test, y_train, y_test = train_test_split(
    train_data_features,
    train_data_target,
    test_size=0.2,
    random_state=42
)
y_test = np.expm1(y_test)


# LinearRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
pipe.fit(X_train, y_train)
predictions = np.expm1(pipe.predict(X_test))

print("\nLinearRegression:")
print("MAE", mean_absolute_error(y_test, predictions))
print("MSE", mean_squared_error(y_test, predictions))
print("RMSE", root_mean_squared_error(y_test, predictions))
print("R2", r2_score(y_test, predictions))


# Ridge

pipe = Pipeline([
    ("scaler", RobustScaler()),
    ("model", Ridge())
])
pipe.fit(X_train, y_train)
predictions =  np.expm1(pipe.predict(X_test))

print("\nRidge:")
print("MAE", mean_absolute_error(y_test, predictions))
print("MSE", mean_squared_error(y_test, predictions))
print("RMSE", root_mean_squared_error(y_test, predictions))
print("R2", r2_score(y_test, predictions))


# Ridge hyperparameters grid

alphas = np.logspace(-3, 3, 10)
param_grid = {
    "model__alpha": alphas,
}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("\nRidge hyperparameters grid:")
print("Best parameters:", grid.best_params_)
predictions = np.expm1(grid.predict(X_test))

print("MAE", mean_absolute_error(y_test, predictions))
print("MSE", mean_squared_error(y_test, predictions))
print("RMSE", root_mean_squared_error(y_test, predictions))
print("R2", r2_score(y_test, predictions))


# graph

plt.scatter(y_test, predictions)
plt.plot([0, 500000], [0, 500000], color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Perfect predictions vs actual')
plt.show()