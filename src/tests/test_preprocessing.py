import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.semf.preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame({
            'A': np.random.rand(100),
            'B': np.random.rand(100),
            'C': np.random.rand(100),
            'D': np.random.rand(100)
        })
        self.y_col = 'D'
        self.train_size = 0.6
        self.valid_size = 0.2

    def test_init(self):
        preprocessor = DataPreprocessor(self.df, self.y_col, self.train_size, self.valid_size)
        self.assertEqual(preprocessor.y_col, self.y_col)
        self.assertEqual(preprocessor.train_size, self.train_size)
        self.assertEqual(preprocessor.valid_size, self.valid_size)

    def test_check_for_duplicates(self):
        df_with_duplicates = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        preprocessor = DataPreprocessor(df_with_duplicates, self.y_col, self.train_size, self.valid_size)
        preprocessor.check_for_duplicates()
        self.assertEqual(len(preprocessor.df), len(self.df))

    def test_split_data(self):
        preprocessor = DataPreprocessor(self.df, self.y_col, self.train_size, self.valid_size)
        preprocessor.split_data()
        total_length = len(preprocessor.df_train) + len(preprocessor.df_valid) + len(preprocessor.df_test)
        self.assertEqual(total_length, len(self.df))

    def test_scale_data(self):
        preprocessor = DataPreprocessor(self.df, self.y_col, self.train_size, self.valid_size)
        preprocessor.split_data()
        preprocessor.scale_data()
        for col in preprocessor.df_train.columns.difference([self.y_col]):
            self.assertAlmostEqual(preprocessor.df_train[col].mean(), 0, delta=0.1)
            self.assertAlmostEqual(preprocessor.df_train[col].std(), 1, delta=0.1)

    def test_get_train_valid_test(self):
        preprocessor = DataPreprocessor(self.df, self.y_col, self.train_size, self.valid_size)
        preprocessor.split_data()
        df_train, df_valid, df_test = preprocessor.get_train_valid_test()
        total_length = len(df_train) + len(df_valid) + len(df_test)
        self.assertEqual(total_length, len(self.df))

    def test_split_X_y(self):
        preprocessor = DataPreprocessor(self.df, self.y_col, self.train_size, self.valid_size)
        df_X, df_y = preprocessor.split_X_y(self.df)
        self.assertEqual(df_X.shape[1], self.df.shape[1] - 1)
        self.assertEqual(df_y.shape[1], 1)
        self.assertEqual(df_y.columns[0], self.y_col)


if __name__ == '__main__':
    unittest.main()
