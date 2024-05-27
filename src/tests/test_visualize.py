import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set the backend to 'Agg' to avoid displaying plots during tests
plt.switch_backend('Agg')

# Import the functions from the module
from src.semf.visualize import (
    visualize_prediction_intervals,
    visualize_prediction_intervals_kde,
    visualize_prediction_intervals_kde_multiple,
    plot_violin,
    get_confidence_intervals,
    plot_confidence_intervals,
    plot_tr_val_metrics
)

class SuppressShow:
    """
    A context manager to suppress plt.show() during tests.
    """
    def __enter__(self):
        self._original_show = plt.show
        plt.show = lambda: None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.show = self._original_show

class TestPlottingFunctions(unittest.TestCase):

    def setUp(self):
        """Set up test data."""
        self.y_preds = np.random.rand(10, 100)
        self.y_true = np.random.rand(10)
        self.df_confidence_intervals = get_confidence_intervals(self.y_preds, self.y_true)
    
    def test_visualize_prediction_intervals(self):
        """Test visualize_prediction_intervals function."""
        with SuppressShow():
            fig = visualize_prediction_intervals(self.y_preds, self.y_true, return_fig=True)
            self.assertIsInstance(fig, plt.Figure)

    def test_visualize_prediction_intervals_kde(self):
        """Test visualize_prediction_intervals_kde function."""
        with SuppressShow():
            fig = visualize_prediction_intervals_kde(self.y_preds[0], self.y_true[0], return_fig=True)
            self.assertIsInstance(fig, plt.Figure)

    def test_visualize_prediction_intervals_kde_multiple(self):
        """Test visualize_prediction_intervals_kde_multiple function."""
        with SuppressShow():
            fig = visualize_prediction_intervals_kde_multiple(self.y_preds, self.y_true, n_instances=5, return_fig=True)
            self.assertIsInstance(fig, plt.Figure)

    def test_plot_violin(self):
        """Test plot_violin function."""
        with SuppressShow():
            plot_violin(self.y_preds, self.y_true, n_instances=5)
            self.assertTrue(True)  # The test passes if no error is raised

    def test_get_confidence_intervals(self):
        """Test get_confidence_intervals function."""
        df = get_confidence_intervals(self.y_preds, self.y_true)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(list(df.columns), ['observation', 'y_test', 'y_mean', 'y_median', 'y_lower', 'y_upper'])

    def test_plot_confidence_intervals(self):
        """Test plot_confidence_intervals function."""
        with SuppressShow():
            fig = plot_confidence_intervals(self.df_confidence_intervals, n_instances=5, return_fig=True)
            self.assertIsInstance(fig, plt.Figure)

    def test_plot_tr_val_metrics(self):
        """Test plot_tr_val_metrics function."""
        class DummySEMF:
            def __init__(self):
                self.train_perf = [{'MAE': np.random.rand(), 'RMSE': np.random.rand(), 'R2': np.random.rand()} for _ in range(10)]
                self.valid_perf = [{'MAE': np.random.rand(), 'RMSE': np.random.rand(), 'R2': np.random.rand()} for _ in range(10)]

        dummy_semf = DummySEMF()
        with SuppressShow():
            fig = plot_tr_val_metrics(dummy_semf, optimal_i_value=5, return_fig=True)
            self.assertIsInstance(fig, plt.Figure)

if __name__ == '__main__':
    unittest.main()