from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from quantile_forest import ExtraTreesQuantileRegressor
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
import os
import sys

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.utils.fixes import parse_version, sp_version
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '..', '..'))

from semf import utils
from semf.models import MLP, QNN
from sklearn.ensemble import ExtraTreesRegressor


class BenchmarkSEMF:
    """
    A class for performing benchmarking of the SEMF model.

    Parameters:
    - df_train (DataFrame): The training dataset.
    - df_valid (DataFrame): The validation dataset.
    - df_test (DataFrame): The test dataset.
    - y_col (str): The column name of the target variable.
    - missing_rate (float): The rate of missing data in the dataset.
    - semf_model (object): The SEMF model object.
    - alpha (float): The significance level for confidence intervals.
    - knn_neighbors (int): The number of nearest neighbors for KNN imputation.
    - base_model (str): The name of the base model for benchmarking. If set to 'all', all models will be benchmarked.
    - test_with_wide_intervals (bool): Whether to test SEMF with wide intervals.
    - seed (int): The random seed for reproducibility.
    - inference_R (int): The number of inference sampling operations for SEMF.
    - tree_n_estimators (int): The number of estimators for XGBoost models.
    - xgb_max_depth (int): The maximum depth of XGBoost models.
    - et_max_depth (int): The maximum depth of Random Forest models.
    - nn_batch_size (int): Batch size for training neural network models.
    - nn_epochs (int): Number of training epochs for neural network models.
    - nn_lr (float): Learning rate for neural network training.
    - nn_load_into_memory (bool): Whether to load data into memory for faster processing during neural network training.
    - device (str): The computation device ('cpu' or 'gpu') for running models, especially deep learning models.
    - models_val_split (float): The validation split ratio for early stopping during training.
    - xgb_patience (int): The number of epochs to wait before early stopping for XGBoost models (both XGB and Quantile_XGB).
    - nn_patience (int): The number of epochs to wait before early stopping for neural network models (both MLP and QNN).

    Attributes:
    - df_train (DataFrame): The training dataset.
    - df_valid (DataFrame): The validation dataset.
    - df_test (DataFrame): The test dataset.
    - y_col (str): The column name of the target variable.
    - missing_rate (float): The rate of missing data in the dataset.
    - semf_model (object): The SEMF model object.
    - knn_neighbors (int): The number of nearest neighbors for KNN imputation.
    - trained_models (dict): A dictionary to store the trained models.
    - alpha (float): The significance level for confidence intervals.
    - base_model (str): The name of the base model for benchmarking.
    - test_with_wide_intervals (bool): Whether to test SEMF with wide intervals.
    - inference_R (int): The number of inference sampling operations for SEMF.
    - tree_n_estimators (int): The number of estimators for XGBoost models.
    - alphas (list): The alpha values for confidence intervals.
    - percentiles (list): The percentiles corresponding to the alpha values.
    - imputation_methods (list): The methods for imputing missing data.
    - models (dict): A dictionary of models for benchmarking.
    """
    
    def __init__(self, df_train, df_valid, df_test, y_col='Y', missing_rate=None, semf_model=None, alpha=0.05, knn_neighbors=5, base_model="all", test_with_wide_intervals=True, seed=0, inference_R=50, tree_n_estimators=100, xgb_max_depth=5, et_max_depth=10, nn_batch_size=64, nn_epochs=1000, nn_lr=0.001, nn_load_into_memory=True, device="cpu", models_val_split=0.1, xgb_patience=10, nn_patience=50):
        utils.set_seed(seed)
        self.seed = seed
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        self.y_col = y_col
        self.missing_rate = missing_rate
        self.semf_model = semf_model
        self.knn_neighbors = knn_neighbors
        self.trained_models = {}
        self.alpha = alpha
        self.base_model = base_model
        self.test_with_wide_intervals = test_with_wide_intervals
        self.inference_R = inference_R
        self.tree_n_estimators = tree_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.et_max_depth = et_max_depth
        self.nn_batch_size = nn_batch_size
        self.nn_epochs = int(nn_epochs)  # must be divided by original R since we have R less rows
        self.nn_lr = nn_lr
        self.nn_load_into_memory = nn_load_into_memory
        self.device = device
        self.models_val_split = models_val_split
        self.xgb_patience = xgb_patience
        self.nn_patience = nn_patience

        self.alphas = [(self.alpha / 2), (1 - (self.alpha / 2))]
        self.percentiles = [100 * alpha for alpha in self.alphas]
        self.imputation_methods = ["median", "mean", f"{self.knn_neighbors}nn", "iterative"]

        quantile_tree_params = {"objective": "reg:quantileerror", "n_estimators": self.tree_n_estimators, "tree_method": "hist", "quantile_alpha": np.array(self.alphas), "random_state": seed}
        self.models = {"SEMF": semf_model}

        model_options = {
            "XGB": (xgb.XGBRegressor(tree_method="hist", n_estimators=self.tree_n_estimators, max_depth=self.xgb_max_depth, random_state=self.seed), xgb.XGBRegressor(max_depth=self.xgb_max_depth, **quantile_tree_params)),
            "ET": (ExtraTreesRegressor(n_estimators=self.tree_n_estimators, random_state=self.seed, max_depth=self.xgb_max_depth), ExtraTreesQuantileRegressor(n_estimators=self.tree_n_estimators, max_depth=self.et_max_depth, random_state=self.seed)),
            "MLP": (MLP(input_size=self.df_train.shape[1] - 1, output_size=1, device=self.device), QNN(input_size=self.df_train.shape[1] - 1, output_size=1, device=self.device))
        }

        if self.base_model in model_options:
            self.models[self.base_model], self.models["Quantile_" + self.base_model] = model_options[self.base_model]
        elif self.base_model == "all":
            for model_name, (model, quant_model) in model_options.items():
                self.models[model_name] = model
                self.models["Quantile_" + model_name] = quant_model

    def impute_data(self, df, strategy="median"):
        """
        Imputes missing data in a DataFrame using the specified strategy.

        Parameters:
        - df (DataFrame): The dataset with missing values.
        - strategy (str): The imputation strategy to use.

        Returns:
        - DataFrame: The dataset with imputed values.
        """
        feature_columns = [col for col in df.columns if col != self.y_col]
        y_col_present = self.y_col in df.columns

        if strategy == f"{self.knn_neighbors}nn":
            imputer = KNNImputer(n_neighbors=self.knn_neighbors)
        elif strategy == "iterative":
            imputer = IterativeImputer(random_state=self.seed)
        else:
            imputer = SimpleImputer(strategy=strategy)

        imputer.fit(self.df_train[feature_columns].copy())
        X_imputed = imputer.transform(df[feature_columns].copy())
        X_imputed = pd.DataFrame(X_imputed, columns=feature_columns)

        if y_col_present:
            X_imputed[self.y_col] = df.copy()[self.y_col].values

        return X_imputed

    def get_imputed_datasets(self):
        """
        Returns a dictionary of imputed datasets.

        Returns:
        - dict: A dictionary containing imputed datasets for each imputation method.
        """
        datasets = {}

        if self.missing_rate is not None and self.missing_rate > 0:
            for method in self.imputation_methods:
                datasets[method] = {
                    "train": self.impute_data(self.df_train, strategy=method),
                    "valid": self.impute_data(self.df_valid, strategy=method),
                    "test": self.impute_data(self.df_test, strategy=method)
                }
        else:
            datasets["original"] = {"train": self.df_train, "valid": self.df_valid, "test": self.df_test}
        return datasets

    def train_benchmark(self, model_name, X_train, y_train, impute_method):
        """
        Trains a benchmark model on the training data.

        Parameters:
        - model_name (str): The name of the model to train.
        - X_train (DataFrame): The training data features.
        - y_train (Series): The training data target variable.
        - impute_method (str): The imputation method used.

        Returns:
        - None
        """
        utils.set_seed(self.seed)
        full_model_name = f"{model_name}_{impute_method}"
        model = self.models.copy()[model_name]

        if model_name in ["XGB", "Quantile_XGB"]:
            if self.models_val_split > 0 and self.models_val_split < 1:
                X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, test_size=self.models_val_split, random_state=self.seed)
                model.fit(X_train_part, y_train_part, early_stopping_rounds=self.xgb_patience, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train)
        elif model_name in ["MLP", "Quantile_MLP"]:
            if y_train.ndim == 1:
                y_train = y_train.to_numpy().reshape(-1, 1)
            X_train_tensor, y_train_tensor = utils.to_tensor(X_train), utils.to_tensor(y_train)
            model = model.train_model(X_train_tensor, y_train_tensor, batch_size=self.nn_batch_size, epochs=self.nn_epochs, lr=self.nn_lr, load_into_memory=self.nn_load_into_memory, nn_patience=self.nn_patience, val_split=self.models_val_split, verbose=False)
        else:
            model.fit(X_train, y_train)

        self.trained_models[full_model_name] = model

    def get_semf_intervals(self, X):
        """
        Returns the lower and upper bounds of the SEMF intervals for a given input.

        Parameters:
        - X (DataFrame): The input data.

        Returns:
        - tuple: Lower and upper bounds of the intervals.
        """
        preds_interval_semf = self.semf_model.infer_semf(X, return_type='interval', use_wide_intervals=self.test_with_wide_intervals, R=self.inference_R)
        lower, upper = np.percentile(preds_interval_semf, self.percentiles, axis=1)
        return lower, upper

    def get_model(self, model_name):
        """
        Returns the trained model with the specified name.

        Parameters:
        - model_name (str): The name of the model to retrieve.

        Returns:
        - object: The trained model.
        """
        return self.trained_models.copy().get(model_name)

    def evaluate_model_pointpred(self, model_name, model, X, y):
        """
        Evaluates a model's performance on point predictions.

        Parameters:
        - model_name (str): The name of the model.
        - model (object): The model to evaluate.
        - X (DataFrame): The input data.
        - y (Series): The true target values.

        Returns:
        - dict: A dictionary containing evaluation metrics.
        """
        y_pred = model.predict(X) if model_name != "SEMF" else self.semf_model.infer_semf(X, return_type='point', use_wide_intervals=False)
        mae = mean_absolute_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        r2 = r2_score(y, y_pred)
        return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

    def run_pointpred(self):
        """
        Runs the benchmarking process for point predictions.

        Returns:
        - dict: A dictionary containing the results of point predictions for each dataset subset.
        """
        datasets = self.get_imputed_datasets()
        results = {'train': [], 'valid': [], 'test': []}
        relative_rmse_values = []
        relative_mae_values = []

        default_relative_metric_value = (None, None)

        for subset, df in {'train': self.df_train, 'valid': self.df_valid, 'test': self.df_test}.items():
            X, y = df.drop(self.y_col, axis=1), df[self.y_col]
            result = self.evaluate_model_pointpred("SEMF", self.semf_model, X, y)
            result['Imputation'] = "Original"
            result['top1_relative_rmse'] = default_relative_metric_value
            result['top1_relative_mae'] = default_relative_metric_value
            results[subset].append(result)

        for subset in results:
            relative_rmse_values = []
            relative_mae_values = []

            for impute_method, data in datasets.items():
                for model_name, model in self.models.items():
                    if model_name != "SEMF":
                        X, y = data[subset].drop(self.y_col, axis=1), data[subset][self.y_col]
                        if subset == 'train':
                            self.train_benchmark(model_name=model_name, X_train=X, y_train=y, impute_method=impute_method)
                        if model_name not in ["Quantile_XGB", "Quantile_ET", "Quantile_MLP"]:
                            model = self.get_model(f"{model_name}_{impute_method}")
                            result = self.evaluate_model_pointpred(model_name, model, X, y)
                            result['Imputation'] = impute_method.capitalize()
                            results[subset].append(result)
                        if model_name == self.base_model:
                            semf_metrics = next((r for r in results[subset] if r['Model'] == 'SEMF' and r['Imputation'] == 'Original'), None)
                            metrics_info = {'MAE': True, 'RMSE': True}
                            relative_metrics = self.calculate_relative_metrics(semf_metrics, result, impute_method, metrics_info)
                            relative_rmse_values.append((relative_metrics['relative_RMSE_' + impute_method + '_100'], impute_method))
                            relative_mae_values.append((relative_metrics['relative_MAE_' + impute_method + '_100'], impute_method))

            relative_rmse_values.sort()
            relative_mae_values.sort()
            top_rmse = relative_rmse_values[:1]
            top_mae = relative_mae_values[:1]

            for result in results[subset]:
                if result['Model'] == 'SEMF':
                    result['top1_relative_rmse'] = top_rmse[0] if len(top_rmse) > 0 else None
                    result['top1_relative_mae'] = top_mae[0] if len(top_mae) > 0 else None

        return results

    def predict_intervals(self, model_name, X, impute_method=None):
        """
        Predicts the intervals for a given model and input.

        Parameters:
        - model_name (str): The name of the model.
        - X (DataFrame): The input data.
        - impute_method (str): The imputation method used.

        Returns:
        - tuple: Lower and upper bounds of the predicted intervals.
        """
        if model_name == "SEMF":
            return self.get_semf_intervals(X)
        try:
            model = self.get_model(f"{model_name}_{impute_method}")
            if model_name in ["Quantile_XGB"]:
                preds = model.predict(X)
            else:
                if model_name in ["Quantile_ET"] and isinstance(X, pd.DataFrame):
                    X = X.values.copy()
                preds = model.predict(X, quantiles=self.alphas)
            return preds[:, 0], preds[:, 1]
        except KeyError:
            raise ValueError(f"Model name '{model_name}' not recognized or imputation method '{impute_method}' is incorrect.")

    def evaluate_model_intervals(self, model_name, y, lower, upper, picp_desired=None, eta=0.5):
        """
        Evaluates a model's performance on interval predictions.

        Parameters:
        - model_name (str): The name of the model.
        - y (Series): The true target values.
        - lower (ndarray): The lower bounds of the predicted intervals.
        - upper (ndarray): The upper bounds of the predicted intervals.
        - picp_desired (float): The desired prediction interval coverage probability.
        - eta (float): The weight for the interval width.

        Returns:
        - dict: A dictionary containing evaluation metrics for interval predictions.
        """
        if picp_desired is None:
            picp_desired = 1 - self.alpha
        picp = np.mean((y >= lower) & (y <= upper))
        mpiw = np.mean(upper - lower)
        y_range = np.max(y) - np.min(y)
        nmpiw = mpiw / y_range if y_range != 0 else 0
        picp_nmpiw_ratio = picp / nmpiw
        result = {
            'Model': model_name,
            'PICP': round(picp, 3),
            'MPIW': round(mpiw, 3),
            'NMPIW': round(nmpiw, 3),
            "CWR": round(picp_nmpiw_ratio, 3)
        }
        return result

    def calculate_relative_metrics(self, semf_metrics, base_metrics, imputation, metrics_info):
        """
        Generic function to calculate relative metrics, considering whether higher or lower is better.

        Parameters:
        - semf_metrics (dict): SEMF model metrics.
        - base_metrics (dict): Base model metrics.
        - imputation (str): Imputation method used.
        - metrics_info (dict): Dictionary with metric names as keys and boolean indicating if lower values are better.

        Returns:
        - dict: Dictionary with relative metrics.
        """
        relative_metrics = {}
        for metric, lower_is_better in metrics_info.items():
            if metric in semf_metrics and metric in base_metrics:
                if lower_is_better:
                    relative_metric = 100 * (base_metrics[metric] - semf_metrics[metric]) / base_metrics[metric]
                else:
                    relative_metric = 100 * (semf_metrics[metric] - base_metrics[metric]) / base_metrics[metric]
                relative_metric = round(relative_metric, 2)
                relative_metrics[f'relative_{metric}_{imputation}_100'] = relative_metric
        return relative_metrics

    def run_intervals(self):
        """
        Runs the benchmarking process for interval predictions.

        Returns:
        - dict: A dictionary containing the results of interval predictions for each dataset subset.
        """
        datasets = self.get_imputed_datasets()
        results = {'train': [], 'valid': [], 'test': []}
        default_relative_metric_value = (None, None, None)

        for subset, df in {'train': self.df_train, 'valid': self.df_valid, 'test': self.df_test}.items():
            X, y = df.drop(self.y_col, axis=1), df[self.y_col]
            lower, upper = self.predict_intervals("SEMF", X)
            result = self.evaluate_model_intervals("SEMF", y, lower, upper)
            result['Imputation'] = "Original"
            results[subset].append(result)
            result['top1_relative_picp'] = default_relative_metric_value
            result['top1_relative_nmpiw'] = default_relative_metric_value
            result['top1_relative_cwr'] = default_relative_metric_value

        for impute_method, data in datasets.items():
            for model_name in self.models:
                if model_name in ["Quantile_XGB", "Quantile_ET", "Quantile_MLP"]:
                    for subset, df in data.items():
                        X, y = df.drop(self.y_col, axis=1), df[self.y_col]
                        lower, upper = self.predict_intervals(model_name, X, impute_method=impute_method)
                        result = self.evaluate_model_intervals(model_name, y, lower, upper)
                        result['Imputation'] = impute_method.capitalize()
                        results[subset].append(result)

        for subset in results:
            relative_picp_values = []
            relative_nmpiw_values = []
            relative_picpnmpiwratio_values = []

            for result in results[subset]:
                if result['Model'] == 'SEMF':
                    semf_metrics = result
                    for impute_method, _ in datasets.items():
                        base_metrics = next((r for r in results[subset] if r['Model'] == ("Quantile_" + self.base_model) and r['Imputation'] == impute_method.capitalize()), None)
                        if base_metrics:
                            metrics_info = {'PICP': False, 'NMPIW': True, 'CWR': False}
                            relative_metrics = self.calculate_relative_metrics(semf_metrics, base_metrics, impute_method, metrics_info)
                            relative_picp_values.append((relative_metrics['relative_PICP_' + impute_method + '_100'], impute_method))
                            relative_nmpiw_values.append((relative_metrics['relative_NMPIW_' + impute_method + '_100'], impute_method))
                            relative_picpnmpiwratio_values.append((relative_metrics['relative_CWR_' + impute_method + '_100'], impute_method))

            relative_picp_values.sort()
            relative_nmpiw_values.sort()
            relative_picpnmpiwratio_values.sort()

            top_picp = relative_picp_values[0] if len(relative_picp_values) > 0 else None
            top_nmpiw = relative_nmpiw_values[0] if len(relative_nmpiw_values) > 0 else None
            top_picpnmpiwratio = relative_picpnmpiwratio_values[0] if len(relative_picpnmpiwratio_values) > 0 else None

            for result in results[subset]:
                if result['Model'] == 'SEMF':
                    result['top1_relative_picp'] = top_picp
                    result['top1_relative_nmpiw'] = top_nmpiw
                    result['top1_relative_cwr'] = top_picpnmpiwratio

        return results

    def plot_predicted_intervals(self, X_eval, y_eval, sample_size=100, return_fig=False):
        """
        Plots predicted intervals for the given evaluation data.

        Parameters:
        - X_eval (DataFrame): The evaluation data features.
        - y_eval (Series): The true target values.
        - sample_size (int): The number of samples to plot.
        - return_fig (bool): Whether to return the figure object.

        Returns:
        - Figure: The plot figure if return_fig is True.
        """
        max_sample_size = X_eval.shape[0]
        sample_size = min(sample_size, max_sample_size)

        colors = sns.color_palette("bright", len(self.models))

        indices = np.random.choice(X_eval.shape[0], sample_size, replace=False)
        X_sample = X_eval.iloc[indices]
        y_sample = y_eval.iloc[indices]

        plt.figure(figsize=(12, 8))
        plt.scatter(np.arange(sample_size), y_sample, edgecolor='k', s=100, label="True Value")

        for color_index, model_name in enumerate(self.models):
            impute_method = "original"
            if model_name in ["SEMF", "Quantile_XGB", "Quantile_ET", "Quantile_MLP"]:
                if model_name == "SEMF":
                    lower, upper = self.predict_intervals(model_name, X_sample)
                    label = f"{model_name} Predicted Interval"
                    plt.fill_between(np.arange(sample_size), lower, upper, alpha=0.1, color=colors[color_index], label=label)
                else:
                    if X_eval.isnull().values.any():
                        for impute_method in self.imputation_methods:
                            X_imputed = self.impute_data(X_sample, strategy=impute_method)
                            lower, upper = self.predict_intervals(model_name, X_imputed, impute_method=impute_method)
                            label = f"{model_name} Predicted Interval ({impute_method.capitalize()})"
                            plt.fill_between(np.arange(sample_size), lower, upper, alpha=0.1, color=colors[color_index], label=label)
                    else:
                        lower, upper = self.predict_intervals(model_name, X_sample, impute_method=impute_method)
                        label = f"{model_name} Interval"
                        plt.fill_between(np.arange(sample_size), lower, upper, alpha=0.1, color=colors[color_index], label=label)

        y_axis_label = y_eval.name if isinstance(y_eval, pd.Series) else (y_eval.columns[0] if isinstance(y_eval, pd.DataFrame) and y_eval.columns.size == 1 else "Predicted Value")

        plt.xlabel("Test Instance (Sampled)")
        plt.ylabel(y_axis_label)
        plt.legend(loc="upper right")
        plt.tight_layout()

        if return_fig:
            return plt.gcf()
        else:
            plt.show()


def display_results(results, sort_descending_by=None, include_imputation=True):
    """
    Displays the benchmarking results.

    Parameters:
    - results (dict): The benchmarking results.
    - sort_descending_by (str): The column name to sort the results by in descending order.
    - include_imputation (bool): Whether to include the imputation method in the display.

    Returns:
    - None
    """
    for dataset, dataset_results in results.items():
        print(f"\nResults for {dataset.upper()} dataset:")
        df = pd.DataFrame(dataset_results)
        if not include_imputation:
            df = df.drop(columns=['Imputation'], errors='ignore')
        if sort_descending_by:
            df = df.sort_values(by=[sort_descending_by], ascending=False)
        print(df.to_string(index=False))


if __name__ == '__main__':
    class MockSEMFModel:
        """
        Mock SEMF Model for demonstration purposes.
        """

        def __init__(self, x_train, y_train):
            self.x_train = x_train
            self.y_train = y_train

        def infer_semf(self, X, return_type='point', use_wide_intervals=True, R=None):
            """
            Infers predictions using the mock SEMF model.

            Parameters:
            - X (DataFrame): The input data.
            - return_type (str): The type of prediction to return ('point' or 'interval').
            - use_wide_intervals (bool): Whether to use wide intervals.
            - R (int): The number of inference sampling operations.

            Returns:
            - ndarray: The predictions.
            """
            point = np.random.rand(X.shape[0])
            interval = np.random.rand(X.shape[0], 100)
            if use_wide_intervals:
                interval += 0.1
            if return_type == 'point':
                return point
            if return_type == 'interval':
                return interval
            elif return_type == 'both':
                return (point, interval)

        @staticmethod
        def introduce_missing_data(df, missing_rate):
            """
            Introduces missing values into the DataFrame.

            Parameters:
            - df (DataFrame): The dataset to introduce missing values into.
            - missing_rate (float): The rate of missing data.

            Returns:
            - DataFrame: The dataset with missing values introduced.
            """
            feature_columns = df.columns[df.columns != 'Y']
            n_missing = int(np.floor(missing_rate * df[feature_columns].size))
            missing_indices = np.random.choice(df[feature_columns].size, n_missing, replace=False)
            df_flat = df[feature_columns].values.flatten()
            df_flat[missing_indices] = np.nan
            df[feature_columns] = df_flat.reshape(df[feature_columns].shape)
            return df

    np.random.seed(10)
    n_obs = 1000
    df = pd.DataFrame(np.random.rand(n_obs, 4), columns=['x1', 'x2', 'x3', 'Y'])
    df_train, df_remaining = train_test_split(df, test_size=0.3, random_state=0)
    df_valid, df_test = train_test_split(df_remaining, test_size=0.5, random_state=0)

    mock_semf_model = MockSEMFModel(df_train.drop('Y', axis=1), df_train['Y'])

    print("Benchmarking with complete data...")

    benchmark_semf_complete = BenchmarkSEMF(df_train, df_valid, df_test, semf_model=mock_semf_model, alpha=0.05, knn_neighbors=5, test_with_wide_intervals=True, seed=1, inference_R=50, tree_n_estimators=100, device="gpu", nn_batch_size=None, nn_epochs=1000, nn_lr=0.001, nn_load_into_memory=True)

    point_benchmark_complete = benchmark_semf_complete.run_pointpred()
    interval_benchmark_complete = benchmark_semf_complete.run_intervals()
    print("\nPoint Prediction Results with Complete Data:")
    display_results(point_benchmark_complete, sort_descending_by='MAE', include_imputation=True)
    print("\nInterval Prediction Results with Complete Data:")
    display_results(interval_benchmark_complete, sort_descending_by='CWR', include_imputation=True)

    benchmark_semf_complete.plot_predicted_intervals(df_test.drop('Y', axis=1), df_test['Y'], sample_size=50)

    print("\n\nBenchmarking with 50% missing data...")
    missing_rate = 0.5
    df_train_missing = MockSEMFModel.introduce_missing_data(df_train.copy(), missing_rate)
    df_valid_missing = MockSEMFModel.introduce_missing_data(df_valid.copy(), missing_rate)
    df_test_missing = MockSEMFModel.introduce_missing_data(df_test.copy(), missing_rate)

    benchmark_semf_missing = BenchmarkSEMF(df_train_missing, df_valid_missing, df_test_missing, missing_rate=missing_rate, semf_model=mock_semf_model, alpha=0.05, knn_neighbors=5, test_with_wide_intervals=True, seed=1, inference_R=50, tree_n_estimators=100, device="gpu", nn_batch_size=None, nn_epochs=500, nn_lr=0.001, nn_load_into_memory=True)
    point_benchmark_missing = benchmark_semf_missing.run_pointpred()
    interval_benchmark_missing = benchmark_semf_missing.run_intervals()
    print("\nPoint Prediction Results with 50% missing data:")
    display_results(point_benchmark_missing, sort_descending_by='MAE')
    print("\nInterval Prediction Results with 50% missing data:")
    display_results(interval_benchmark_missing, sort_descending_by='CWR')

    benchmark_semf_missing.plot_predicted_intervals(df_test_missing.drop('Y', axis=1), df_test_missing['Y'], sample_size=50)

    print(utils.format_model_metrics(point_benchmark_complete))
    print(utils.format_model_metrics(interval_benchmark_complete))
    print(utils.format_model_metrics(point_benchmark_missing))
    print(utils.format_model_metrics(interval_benchmark_missing))
