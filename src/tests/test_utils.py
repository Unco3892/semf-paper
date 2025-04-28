import unittest
import numpy as np
import pandas as pd
import os
from src.semf.utils import (
    sample_z_r, reshape_z_t, set_seed, sample_z_r_reshaped, print_diagnostics,
    check_early_stopping, calculate_performance, flatten_3d_array,
    format_model_metrics, load_openml_dataset, print_data_completeness
)

class TestUtils(unittest.TestCase):

    def test_sample_z_r(self):
        z_means = np.array([[0.5, 1], [1.5, 2], [2.5, 3]])
        desired_sd = np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
        result = sample_z_r(2, z_means, 3, 4, desired_sd)
        self.assertEqual(result.shape, (3, 2, 4))

    def test_reshape_z_t(self):
        mat = np.array([[0.5, 1], [1.5, 2], [2.5, 3], [1.4, 8], [9, 2.7], [0.6, 1.9]])
        result = reshape_z_t(mat, 2)
        self.assertEqual(result.shape, (2, 2, 3))

    def test_set_seed(self):
        set_seed(42)
        self.assertEqual(np.random.randint(0, 100), 51)

    def test_sample_z_r_reshaped(self):
        z_R_means = np.array([[[0, 1, 0, 1], [0, 1, 0, 1]], [[0, 1, 0, 1], [0, 1, 0, 1]], [[0, 1, 0, 1], [0, 1, 0, 1]]])
        desired_sd = np.array([[[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]], [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]], [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]])
        result = sample_z_r_reshaped(z_R_means, desired_sd)
        self.assertEqual(result.shape, z_R_means.shape)

    def test_calculate_performance(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        result = calculate_performance(y_true, y_pred)
        self.assertIn('MAE', result)
        self.assertIn('RMSE', result)
        self.assertIn('R2', result)

    def test_flatten_3d_array(self):
        arr_3d = np.random.rand(4, 3, 2)
        result = flatten_3d_array(arr_3d)
        self.assertEqual(result.shape, (8, 3))

    def test_load_openml_dataset(self):
        dataset_names = {"iris": 61}
        cache_dir = "./cache"
        os.makedirs(cache_dir, exist_ok=True)
        data, target_name = load_openml_dataset("iris", dataset_names, cache_dir)
        self.assertTrue(isinstance(data, pd.DataFrame))
        self.assertEqual(target_name, "class")
        os.remove(os.path.join(cache_dir, "iris.csv"))
        os.rmdir(cache_dir)

    # Additional example for print_diagnostics
    def test_print_diagnostics(self):
        train_perf = {'R2': 0.9, 'RMSE': 0.1, 'MAE': 0.05}
        valid_perf = {'R2': 0.85, 'RMSE': 0.15, 'MAE': 0.07}
        print_diagnostics(train_perf, valid_perf, metrics=["R2", "RMSE", "MAE"])

    # Additional example for check_early_stopping
    def test_check_early_stopping(self):
        valid_perf = [{'MAE': 0.6}, {'MAE': 0.7}, {'MAE': 0.8}]
        result = check_early_stopping(3, valid_perf, 'MAE', 5)
        self.assertEqual(result, 6)  # Based on the input, it should return 6 (5 + 1)

if __name__ == '__main__':
    unittest.main()
