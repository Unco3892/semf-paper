import unittest
import numpy as np
import pandas as pd
from src.semf.neural_simulator import MissingDataSimulator
import tensorflow as tf

class TestMissingDataSimulator(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        X = np.random.normal(size=(100, 4))
        mask = np.random.rand(*X.shape) < 0.5
        keep_columns = [0, 2]
        mask[:, keep_columns] = False
        X[mask] = np.nan
        self.df = pd.DataFrame(X, columns=[f"col{i+1}" for i in range(X.shape[1])])
        self.simulator = MissingDataSimulator(self.df, R=10, epochs=2, patience=1)

    def test_prepare_xi_data(self):
        self.simulator.prepare_xi_data()
        self.assertIsNotNone(self.simulator.complete_data)
        self.assertIsNotNone(self.simulator.missing_data)
        self.assertTrue(len(self.simulator.missing_col_names) > 0)
        self.assertTrue(len(self.simulator.complete_col_names) > 0)

    def test_run_xi_sampling(self):
        self.simulator.prepare_xi_data()
        data = self.simulator.run_xi_sampling(sampling_seed=0)
        self.assertIn("X2_mat", data)
        self.assertIn("X1_mat", data)
        self.assertEqual(data["X2_mat"].shape[0], 10 * len(self.simulator.missing_data))
        self.assertEqual(data["X1_mat"].shape[1], len(self.simulator.complete_data))

    def test_train_xi_model(self):
        self.simulator.prepare_xi_data()
        data = self.simulator.run_xi_sampling(sampling_seed=0)
        self.simulator.train_xi_model(data["X2_mat"], data["X1_mat"])
        self.assertIsNotNone(self.simulator.xi_model)
        self.assertIsInstance(self.simulator.xi_model, tf.keras.Model)

    def test_replace_na_in_df(self):
        self.simulator.prepare_xi_data()
        data = self.simulator.run_xi_sampling(sampling_seed=0)
        self.simulator.train_xi_model(data["X2_mat"], data["X1_mat"])
        df_filled = self.simulator.replace_na_in_df()
        self.assertFalse(df_filled.isnull().any().any())

    def test_transform_new_data(self):
        new_df = self.df.copy()
        new_df.iloc[0, 1] = np.nan  # Introduce a missing value
        transformed_data = self.simulator.transform_new_data(new_df)
        self.assertIn("complete_data", transformed_data)
        self.assertIn("missing_data", transformed_data)
        self.assertIn("missing_col_names", transformed_data)
        self.assertIn("complete_col_names", transformed_data)

if __name__ == '__main__':
    unittest.main()
