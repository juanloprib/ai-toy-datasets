import pandas as pd
from pandas import DataFrame


class AnalyzeData:
    def __init__(self, features: DataFrame, target: DataFrame):
        self.features: DataFrame = features
        self.target: DataFrame = target
        self.max_skew = 1.2
        self.min_skew = -1.2

    
    def update_data(self, features: DataFrame, target: DataFrame):
        self.features = features
        self.target = target
    

    def analyze(self):
        print("Dataset analysis:")
        if self.has_empty_values():
            print(f"⛔ The dataset has empty values in columns")
        else:
            print("✅ The dataset has no empty values")
        if self.has_duplicated_rows():
            print(f"⛔ The dataset has {self.has_duplicates()} duplicated rows")
        else:
            print("✅ The dataset has no duplicated rows")
        if self.has_duplicated_cols():
            print(f"⛔ The dataset has {self.has_duplicated_cols()} duplicated columns")
        else:
            print("✅ The dataset has no duplicated columns")
        if not self.all_rows_have_target():
            print("⛔ Not all rows have target values")
        else:
            print("✅ All rows have target values")
        if self.has_object_cols():
            print("⛔ The dataset has object columns")
        else:
            print("✅ The dataset has no object columns")
        """ if self.is_skewed():
            print("⛔ The dataset has skewed columns")
        else:
            print("✅ The dataset has no skewed columns") """
        print()

    
    def is_skewed(self):
        return self.features.skew().any() > self.max_skew or self.features.skew().any() < self.min_skew
        

    def has_empty_values(self):
        return self.features.isnull().values.any() or self.target.isnull().values.any()
    

    def has_duplicated_rows(self):
        return self.features.duplicated().sum()
    

    def has_duplicated_cols(self):
        return self.features.columns.duplicated().sum()
    

    def all_rows_have_target(self):
        return len(self.features) == len(self.target)
    

    def has_object_cols(self):
        return self.features.select_dtypes(include=['object']).columns.any()


    def basic_info(self):
        self.print_all_vals()
        self.custom_info_table()
        print(self.features.describe(), end='\n\n')
        self.categorical_info()
        print(self.target.describe(), end='\n\n')
        self.print_default_vals()


    def custom_info_table(self):
        data_info = pd.DataFrame(self.features.columns, columns=['Column'])
        data_info['Data Type'] = self.features.dtypes.values
        data_info['Nulls'] = self.features.isnull().sum().values
        data_info['Nulls %'] = (self.features.isnull().sum().values / len(self.features) * 100).round(2)
        data_info['Unique Values'] = self.features.nunique().values
        print(data_info, end='\n\n')


    def categorical_info(self):
        categorical_cols = self.features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(col, self.features[col].unique(), len(self.features[col].unique()))
        print()


    def print_all_vals(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)


    def print_default_vals(self):
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        