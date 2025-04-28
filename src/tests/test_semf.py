import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.semf.semf import SEMF
from src.semf.preprocessing import DataPreprocessor
from src.semf import utils
import time
import os
import shutil

class TestSEMF(unittest.TestCase):
    def setUp(self):
        utils.set_seed(10)
        self.n_R = 3
        self.n_obs = 1000
        self.max_it = 2
        self.seed = 10
        self.z_norm_sd = 0.01
        self.nodes = np.array([2, 3, 4])
        self.verbose = True
        self.stopping_metric = "RMSE"
        self.patience = 5
        self.return_mean_default = True
        self.base_dir = "models"

        # Create a complete dataset
        self.df_complete = pd.DataFrame(np.random.rand(self.n_obs, 4), columns=["x1", "x2", "x3", "y"])
        self.df_train_complete, self.df_test_complete = train_test_split(self.df_complete, test_size=0.2, random_state=0)

    def initialize_data_preprocessor(self, df_train, df_test):
        data_preprocessor = DataPreprocessor(df_train, y_col="y")
        data_preprocessor.split_data(df_test)
        data_preprocessor.scale_data(scale_output=True)
        return data_preprocessor

    def test_semf_complete_data(self):
        data_preprocessor = self.initialize_data_preprocessor(self.df_train_complete, self.df_test_complete)
        semf = SEMF(data_preprocessor, R=self.n_R, nodes_per_feature=self.nodes, seed=self.seed, z_norm_sd=self.z_norm_sd, return_mean_default=self.return_mean_default, stopping_metric=self.stopping_metric, stopping_patience=self.patience, max_it=self.max_it, verbose=self.verbose)
        st = time.time()
        result = semf.train_semf()
        et = time.time()
        elapsed_time = et - st

        # Update the assertion to check the existence of train_perf or similar results attribute
        self.assertTrue(hasattr(semf, 'train_perf') and semf.train_perf, "SEMF training did not complete successfully.")
        print(f"Execution Time (Complete Data): {elapsed_time:.2f} seconds")

    def test_save_and_load_semf(self):
        ds_name_complete = "complete_data"
        data_preprocessor_complete = self.initialize_data_preprocessor(self.df_train_complete, self.df_test_complete)
        semf_complete = SEMF(data_preprocessor_complete, R=self.n_R, nodes_per_feature=self.nodes, seed=self.seed, z_norm_sd=self.z_norm_sd, return_mean_default=self.return_mean_default, stopping_metric=self.stopping_metric, stopping_patience=self.patience, max_it=self.max_it, verbose=self.verbose)
        semf_complete.train_semf()
        
        sample_x_test_complete = semf_complete.x_test.iloc[[0]]
        preds_before_saving_complete = semf_complete.infer_semf(sample_x_test_complete)

        # Save the model and data preprocessor
        semf_complete.save_semf(data_preprocessor_complete, ds_name_complete, self.base_dir)

        # Load the model and data preprocessor
        loaded_semf_complete, loaded_data_preprocessor_complete = SEMF.load_semf(ds_name_complete, self.base_dir)
        preds_after_loading_complete = loaded_semf_complete.infer_semf(sample_x_test_complete)

        # Validate the predictions
        self.assertTrue(np.allclose(preds_before_saving_complete, preds_after_loading_complete), "Loaded model does not produce the same predictions!")
        print("Model loading test passed!")

    def tearDown(self):
        # Clean up the models directory after tests
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)

if __name__ == '__main__':
    unittest.main()
