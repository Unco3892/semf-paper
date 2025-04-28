from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_pinball_loss
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
import os
import sys
from mapie.regression import MapieRegressor
# from quantile_forest import RandomForestQuantileRegressor
from quantile_forest import ExtraTreesQuantileRegressor
from sklearn.ensemble import ExtraTreesRegressor
import copy
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
# Suppress specific FutureWarning from xgboost.data related to pandas deprecations
# warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost.data") # This wasn't enough, reverting to broader filter
from sklearn.utils.fixes import parse_version, sp_version
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '..', '..'))

from semf import utils
from semf.models import MLP, QNN
# ----------------------------
# Helper functions (moved outside or made private methods)
# ----------------------------
def _calculate_coverage_width(y_true, y_lower, y_upper):
    """Calculates empirical coverage and average width of prediction intervals."""
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
    width = np.mean(y_upper - y_lower)
    return coverage, width

def _calculate_crps(y_true, lower, upper, epsilon=1e-8):
    """
    Calculates the Continuous Ranked Probability Score assuming a uniform
    forecast distribution U(lower, upper). Handles zero-width intervals.
    """
    y_true = np.asarray(y_true).squeeze()
    lower = np.asarray(lower).squeeze()
    upper = np.asarray(upper).squeeze()

    d = upper - lower
    # Handle cases where lower == upper (point prediction or zero width)
    crps_zero_width = np.abs(y_true - lower)

    # Avoid division by zero or issues with very small d
    d_safe = np.where(np.isclose(d, 0), epsilon, d)

    # Calculate CRPS components for non-zero width intervals
    term1_lower = ((lower + upper) / 2 - y_true) - d_safe / 6 # y_true < lower
    term1_upper = (y_true - (lower + upper) / 2) - d_safe / 6 # y_true > upper
    term1_within = (((y_true - lower)**2 + (upper - y_true)**2) / (2 * d_safe)) - d_safe / 6 # lower <= y_true <= upper

    # Combine using np.where, ensuring conditions cover all cases
    crps = np.where(np.isclose(d, 0), crps_zero_width,
                    np.where(y_true < lower, term1_lower,
                             np.where(y_true > upper, term1_upper,
                                      term1_within)))
    return np.mean(crps)

def _apply_cqr(y_calib_true, calib_lower_initial, calib_upper_initial,
               test_lower_initial, test_upper_initial, alpha):
    """
    Applies Conformalized Quantile Regression (CQR).

    Args:
        y_calib_true (np.ndarray): True target values for calibration set.
        calib_lower_initial (np.ndarray): Initial lower quantile preds on calibration set.
        calib_upper_initial (np.ndarray): Initial upper quantile preds on calibration set.
        test_lower_initial (np.ndarray): Initial lower quantile preds on test set.
        test_upper_initial (np.ndarray): Initial upper quantile preds on test set.
        alpha (float): Desired miscoverage rate.

    Returns:
        tuple: (cqr_lower, cqr_upper)
               - cqr_lower: Adjusted lower bounds for the test set.
               - cqr_upper: Adjusted upper bounds for the test set.
    """
    y_calib_arr = np.asarray(y_calib_true).squeeze()
    calib_lower_initial = np.asarray(calib_lower_initial).squeeze()
    calib_upper_initial = np.asarray(calib_upper_initial).squeeze()
    test_lower_initial = np.asarray(test_lower_initial).squeeze()
    test_upper_initial = np.asarray(test_upper_initial).squeeze()
    n_cal = len(y_calib_arr)

    # 1. Calculate conformity scores on the calibration set
    conformity_scores = np.maximum(calib_lower_initial - y_calib_arr, y_calib_arr - calib_upper_initial)

    # 2. Find the CQR quantile correction factor 'q_hat'
    q_level = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
    q_level = min(q_level, 1.0) # Ensure q_level does not exceed 1
    # Use 'higher' interpolation for conservative quantile estimate
    q_hat = np.quantile(conformity_scores, q_level, method="higher")

    # 3. Adjust the original test intervals using q_hat
    cqr_lower = test_lower_initial - q_hat
    cqr_upper = test_upper_initial + q_hat

    return cqr_lower, cqr_upper

class BenchmarkSEMF:
    """
    A class for performing benchmarking of the SEMF model.

    Parameters:
    - df_train (DataFrame): The training dataset.
    - df_valid (DataFrame): The validation dataset (used for conformal calibration).
    - df_test (DataFrame): The test dataset.
    - y_col (str): The column name of the target variable.
    - semf_model (object): The SEMF model object.
    - alpha (float): The significance level for confidence intervals.
    - base_model (str): The name of the base model for benchmarking ('XGB', 'ET', 'MLP').
                         If set to 'all', all models will be benchmarked.
    - baseline_interval_method (str): Method for baseline intervals ('quantile' or 'conformal_point').
    - test_with_wide_intervals (bool): Whether to use wide intervals for SEMF.
    - seed (int): The random seed for reproducibility.
    - inference_R (int): The number of inference sampling operations for SEMF.
    - tree_n_estimators (int): The number of estimators for tree-based models.
    - xgb_max_depth (int): The maximum depth of XGBoost models.
    - et_max_depth (int): The maximum depth of ExtraTrees models.
    - nn_batch_size (int): Batch size for training neural network models.
    - nn_epochs (int): Number of training epochs for neural network models.
    - nn_lr (float): Learning rate for neural network training.
    - nn_load_into_memory (bool): Whether to load data into memory for faster NN training.
    - device (str): The computation device ('cpu' or 'gpu').
    - models_val_split (float): The validation split ratio used *within* model training for early stopping.
    - xgb_patience (int): Early stopping patience for XGBoost.
    - nn_patience (int): Early stopping patience for NNs.
    - mapie_cv (int or 'prefit'): Cross-validation strategy for MAPIE. Use 'prefit' if model is already trained, otherwise e.g., 5.

    Attributes:
        (Many attributes are implicitly defined/used, focusing on key ones)
        trained_models (dict): Stores trained baseline models (point and quantile).
        mapie_models (dict): Stores fitted MAPIE regressors wrapping point models.
        alphas (list): Quantile levels [alpha/2, 1-alpha/2].
        baseline_interval_method (str): Method for baseline intervals ('quantile' or 'conformal_point').
    """

    def __init__(self, df_train, df_valid, df_test, y_col='Y', semf_model=None, alpha=0.05, base_model="all", baseline_interval_method="quantile", test_with_wide_intervals=True, seed=0, inference_R=50, tree_n_estimators=100, xgb_max_depth=None, et_max_depth=10, nn_batch_size=None, nn_epochs=1000, nn_lr=0.001, nn_load_into_memory=True, device="cpu", models_val_split=0.15, xgb_patience=10, nn_patience=50, mapie_cv=5):
        utils.set_seed(seed)
        self.seed = seed
        self.df_train = df_train
        self.df_valid = df_valid # Calibration set
        self.df_test = df_test
        self.y_col = y_col
        self.semf_model = semf_model # Assumed already trained
        self.alpha = alpha
        self.base_model = base_model
        self.baseline_interval_method = baseline_interval_method # Store the chosen method
        self.test_with_wide_intervals = test_with_wide_intervals # Store this setting
        self.inference_R = inference_R
        self.tree_n_estimators = tree_n_estimators
        self.xgb_max_depth = xgb_max_depth if xgb_max_depth is not None else 6 # Default if None
        self.et_max_depth = et_max_depth if et_max_depth is not None else 10 # Default if None
        self.nn_batch_size = nn_batch_size
        self.nn_epochs = int(nn_epochs)
        self.nn_lr = nn_lr
        self.nn_load_into_memory = nn_load_into_memory
        self.device = device
        self.models_val_split = models_val_split
        self.xgb_patience = xgb_patience
        self.nn_patience = nn_patience
        self.mapie_cv = mapie_cv

        self.alphas = [self.alpha / 2, 1 - self.alpha / 2]
        self.percentiles = [100 * q for q in self.alphas] # For SEMF percentile calc

        # --- Train all baseline models upfront ---
        self.trained_models = {}
        self.mapie_models = {} # To store fitted MAPIE models

        X_train, y_train = self.df_train.drop(self.y_col, axis=1), self.df_train[self.y_col]
        X_train_np = X_train.to_numpy().astype(np.float32) # For potential skorch usage
        y_train_np = y_train.to_numpy().ravel().astype(np.float32)

        # Define potential models
        model_definitions = {
            "XGB": {
                "point": lambda: xgb.XGBRegressor(objective='reg:squarederror', tree_method="hist", n_estimators=self.tree_n_estimators, max_depth=self.xgb_max_depth, random_state=self.seed, early_stopping_rounds=self.xgb_patience),
                "quantile": lambda: xgb.XGBRegressor(objective='reg:quantileerror', tree_method="hist", n_estimators=self.tree_n_estimators, max_depth=self.xgb_max_depth, quantile_alpha=np.array(self.alphas), random_state=self.seed, early_stopping_rounds=self.xgb_patience)
            },
            "ET": {
                "point": lambda: ExtraTreesRegressor(n_estimators=self.tree_n_estimators, random_state=self.seed, max_depth=self.et_max_depth, n_jobs=-1),
                "quantile": lambda: ExtraTreesQuantileRegressor(n_estimators=self.tree_n_estimators, random_state=self.seed, max_depth=self.et_max_depth, n_jobs=-1)
            },
            "MLP": {
                "point": lambda: MLP(input_size=X_train.shape[1], output_size=1, device=self.device),
                "quantile": lambda: QNN(input_size=X_train.shape[1], output_size=1, device=self.device) # QNN takes tau as input, not quantiles at init
            }
        }

        models_to_train = []
        if self.base_model == "all":
            models_to_train = list(model_definitions.keys())
        elif self.base_model in model_definitions:
            models_to_train = [self.base_model]

        # Instead of re-splitting, use the provided validation set directly.
        X_train_part, y_train_part = X_train, y_train
        # Assuming X_valid and y_valid are passed (or can be derived) from the DataPreprocessor splits:
        X_val_internal, y_val_internal = self.df_valid.drop(self.y_col, axis=1), self.df_valid[self.y_col]

        for model_key in models_to_train:
            print(f"Training baseline models for {model_key}...")
            # Train Point Model
            point_model_name = model_key
            point_model = model_definitions[model_key]["point"]()
            if model_key in ["XGB"]:
                point_model.fit(X_train_part, y_train_part,
                                eval_set=[(X_val_internal, y_val_internal)], verbose=False)
            elif model_key == "MLP":
                 # Pass the full training data; MLP.train_model will handle internal validation split
                 y_t = y_train.to_numpy().reshape(-1, 1)
                 point_model.train_model(utils.to_tensor(X_train), utils.to_tensor(y_t),
                                          # No explicit val_inputs/val_outputs needed here
                                          batch_size=self.nn_batch_size, epochs=self.nn_epochs, lr=self.nn_lr,
                                          load_into_memory=self.nn_load_into_memory, nn_patience=self.nn_patience,
                                          val_split=self.models_val_split, # Ensure internal split uses the same ratio
                                          verbose=False)
            else: # ET
                point_model.fit(X_train, y_train) # ET doesn't have easy early stopping via validation set
            self.trained_models[point_model_name] = point_model
            print(f"  Trained {point_model_name}")

            # Conditionally Setup and Fit MAPIE only if needed
            if self.baseline_interval_method == "conformal_point":
                print(f"  Setting up and fitting MAPIE for {point_model_name}...")
                # Clone the fitted model using deepcopy *before* modifying it
                estimator_for_mapie = copy.deepcopy(point_model)
                # Disable early stopping for XGB to avoid error in MAPIE's fit on the *copy*
                if hasattr(estimator_for_mapie, "set_params"):
                    estimator_for_mapie.set_params(early_stopping_rounds=None)

                mapie_model = MapieRegressor(estimator=estimator_for_mapie, cv=self.mapie_cv, method='base', n_jobs=-1)
                mapie_model.fit(X_train, y_train.values.ravel())  # Fit MAPIE on full training data
                self.mapie_models[point_model_name] = mapie_model
                print(f"    Fitted MAPIE for {point_model_name}")

            # Conditionally train Quantile Model only if needed
            if self.baseline_interval_method == "quantile":
                quantile_model_name = f"Quantile_{model_key}"
                quantile_model = model_definitions[model_key]["quantile"]()
                print(f"  Training {quantile_model_name}...")
                if model_key in ["XGB"]:
                    # Quantile XGB needs separate models per quantile if not using built-in multi-output support correctly
                    # For simplicity, let's assume we train ONE model predicting BOTH quantiles like the setup suggests
                    quantile_model.fit(X_train_part, y_train_part,
                                      eval_set=[(X_val_internal, y_val_internal)], verbose=False)
                elif model_key == "MLP":
                    # Pass the full training data; QNN.train_model will handle internal validation split
                    y_t = y_train.to_numpy().reshape(-1, 1)
                    quantile_model.train_model(utils.to_tensor(X_train), utils.to_tensor(y_t),
                                                # No explicit val_inputs/val_outputs needed here
                                                batch_size=self.nn_batch_size, epochs=self.nn_epochs, lr=self.nn_lr,
                                                load_into_memory=self.nn_load_into_memory, nn_patience=self.nn_patience,
                                                val_split=self.models_val_split, # Ensure internal split uses the same ratio
                                                verbose=False)
                else: # ET
                    quantile_model.fit(X_train, y_train) # Quantile ET fits on full train data
                self.trained_models[quantile_model_name] = quantile_model
                print(f"    Trained {quantile_model_name}")

        print("Baseline model training complete.")


    def get_data_subsets(self):
        """
        Returns a dictionary of dataset subsets (train, valid, test).
        """
        return {
            "train": self.df_train,
            "valid": self.df_valid, # Use validation set for calibration
            "test": self.df_test
        }

    def get_semf_intervals(self, X):
        """
        Returns the raw lower and upper percentile bounds of the SEMF intervals for a given input X.
        These are the *initial* intervals before conformalization.

        Parameters:
        - X (DataFrame): The input data.

        Returns:
        - tuple: Lower and upper bounds of the intervals.
        """
        preds_interval_semf = self.semf_model.infer_semf(X, return_type='interval', use_wide_intervals=self.test_with_wide_intervals, R=self.inference_R)
        # print(f"SEMF interval prediction shape: {preds_interval_semf.shape}")
        # lower, upper = np.percentile(preds_interval_semf, self.percentiles, axis=1)
        # get lower and upper intervals
        # lower, upper = np.percentile(preds_interval_semf, [0, 100], axis=1)
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
        """Evaluates point prediction performance for SEMF and baseline models."""
        results = {'train': [], 'valid': [], 'test': []}
        dfs = {'train': self.df_train, 'valid': self.df_valid, 'test': self.df_test}
        default_relative_metric_value = (np.nan, "") # Use NaN for missing relative values

        # Models to evaluate for point predictions
        point_model_keys = [k for k in self.trained_models.keys() if not k.startswith("Quantile_")]

        # Evaluate SEMF first
        if self.semf_model:
            for subset, df in dfs.items():
                X, y = df.drop(self.y_col, axis=1), df[self.y_col]
                try:
                    result = self.evaluate_model_pointpred("SEMF", self.semf_model, X, y)
                    # Initialize relative metrics placeholders
                    result['top1_rel_rmse'] = default_relative_metric_value
                    result['top1_rel_mae'] = default_relative_metric_value
                    results[subset].append(result)
                except Exception as e:
                    print(f"Error evaluating SEMF point prediction on {subset}: {e}")
                    results[subset].append({
                        'Model': 'SEMF', 'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan,
                        'top1_rel_rmse': default_relative_metric_value,
                        'top1_rel_mae': default_relative_metric_value
                    })

        # Evaluate pre-trained baseline point models
        for model_name in point_model_keys:
            model_trained = self.get_model(model_name)
            if model_trained is None:
                print(f"Warning: Pre-trained model {model_name} not found.")
                continue
            for subset, df in dfs.items():
                X, y = df.drop(self.y_col, axis=1), df[self.y_col]
                try:
                    result = self.evaluate_model_pointpred(model_name, model_trained, X, y)
                    results[subset].append(result)
                except Exception as e:
                    print(f"Error evaluating {model_name} point prediction on {subset}: {e}")
                    results[subset].append({'Model': model_name, 'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan})

        # Calculate relative metrics for SEMF compared to its specific baseline
        semf_base_model_name = self.base_model # e.g., 'XGB', 'ET', 'MLP'
        if self.base_model == "all":
             print("Warning: Cannot calculate specific relative metrics when base_model='all'. Add logic to pick one or skip.")
             # Or perhaps default to XGB? For now, skip specific comparison if 'all'
             semf_base_model_name = None

        if semf_base_model_name and self.semf_model:
            for subset in results:
                semf_result = next((r for r in results[subset] if r['Model'] == 'SEMF'), None)
                base_result = next((r for r in results[subset] if r['Model'] == semf_base_model_name), None)

                if semf_result and base_result and not np.isnan(semf_result['RMSE']) and not np.isnan(base_result['RMSE']):
                    metrics_info = {'MAE': True, 'RMSE': True} # Lower is better
                    relative_metrics = self.calculate_relative_metrics(semf_result, base_result, metrics_info)
                    # Store relative metrics directly in the SEMF result dictionary
                    # Note: 'top1_' naming convention is kept, but it now refers to the specific baseline comparison
                    semf_result['top1_rel_rmse'] = (relative_metrics.get('relative_RMSE', np.nan), "")
                    semf_result['top1_rel_mae'] = (relative_metrics.get('relative_MAE', np.nan), "")
                elif semf_result:
                     # Ensure placeholders exist even if comparison failed
                     semf_result['top1_rel_rmse'] = default_relative_metric_value
                     semf_result['top1_rel_mae'] = default_relative_metric_value

        return results

    def predict_intervals(self, model_name, X):
        if model_name == "SEMF":
            return self.get_semf_intervals(X)
        # If using conformal point baseline for a point model, return points as both bounds.
        if self.base_model == "conformal_point" and model_name in ["XGB", "ET", "MLP"]:
            model = self.get_model(model_name)
            preds = model.predict(X)  # point predictions (shape: [n_samples])
            # print(f"Point baseline prediction shape: {preds.shape}")
            return preds, preds
        try:
            model = self.get_model(model_name)
            if model_name in ["Quantile_XGB"]:
                preds = model.predict(X)
                # print(f"Interval baseline prediction shape: {preds.shape}")
            else:
                if model_name in ["Quantile_ET"] and isinstance(X, pd.DataFrame):
                    X = X.values.copy()
                preds = model.predict(X, quantiles=self.alphas)
            return preds[:, 0], preds[:, 1]
        except KeyError:
            raise ValueError(f"Model name '{model_name}' not recognized.")

    def evaluate_model_intervals(self, model_name, y, lower, upper):
        """
        Evaluates a model's performance on interval predictions.

        Parameters:
        - model_name (str): The name of the model.
        - y (Series): The true target values.
        - lower (ndarray): The lower bounds of the predicted intervals.
        - upper (ndarray): The upper bounds of the predicted intervals.

        Returns:
        - dict: A dictionary containing evaluation metrics for interval predictions.
        """
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
            'CWR': round(picp_nmpiw_ratio, 3)
        }
        
        # Use helper functions to compute CRPS and Pinball loss
        result['CRPS'] = round(self._compute_crps(y, lower, upper), 3)
        
        # Replace the custom pinball loss with scikit-learn's mean_pinball_loss.
        # For a two-sided pinball loss, we compute for the lower and upper quantiles.
        tau_lower = self.alpha / 2
        tau_upper = 1 - (self.alpha / 2)
        loss_lower = mean_pinball_loss(y, lower, alpha=tau_lower)
        loss_upper = mean_pinball_loss(y, upper, alpha=tau_upper)
        result['Pinball'] = round((loss_lower + loss_upper) / 2, 3)
        
        return result

    def calculate_relative_metrics(self, semf_metrics, base_metrics, metrics_info):
        relative_metrics = {}
        for metric, lower_is_better in metrics_info.items():
            if metric in semf_metrics and metric in base_metrics:
                if lower_is_better:
                    relative_metric = 100 * (base_metrics[metric] - semf_metrics[metric]) / base_metrics[metric]
                else:
                    relative_metric = 100 * (semf_metrics[metric] - base_metrics[metric]) / base_metrics[metric]
                relative_metric = round(relative_metric, 2)
                relative_metrics[f'relative_{metric}'] = relative_metric
        return relative_metrics

    def run_intervals(self):
        """Evaluates interval prediction performance for SEMF and baseline models using appropriate conformalization."""
        results = {'train': [], 'valid': [], 'test': []}
        dfs = {'train': self.df_train, 'valid': self.df_valid, 'test': self.df_test}
        default_relative_metric_value = (np.nan, "") # Use NaN for missing relative values

        # Get calibration data (using the validation set)
        X_calib, y_calib = self.df_valid.drop(self.y_col, axis=1), self.df_valid[self.y_col]

        # --- 1. Evaluate SEMF with CQR --- #
        if self.semf_model:
            print("Evaluating SEMF intervals with CQR...")
            try:
                # Get initial intervals for calibration and test sets
                semf_lower_calib_init, semf_upper_calib_init = self.get_semf_intervals(X_calib)
                semf_lower_test_init, semf_upper_test_init = self.get_semf_intervals(dfs['test'].drop(self.y_col, axis=1))
                semf_lower_train_init, semf_upper_train_init = self.get_semf_intervals(dfs['train'].drop(self.y_col, axis=1))

                # Apply CQR using calibration set
                semf_lower_test_cqr, semf_upper_test_cqr = _apply_cqr(
                    y_calib, semf_lower_calib_init, semf_upper_calib_init,
                    semf_lower_test_init, semf_upper_test_init, self.alpha
                )
                semf_lower_calib_cqr, semf_upper_calib_cqr = _apply_cqr(
                    y_calib, semf_lower_calib_init, semf_upper_calib_init,
                    semf_lower_calib_init, semf_upper_calib_init, self.alpha # Calibrate on itself
                )
                semf_lower_train_cqr, semf_upper_train_cqr = _apply_cqr(
                    y_calib, semf_lower_calib_init, semf_upper_calib_init,
                    semf_lower_train_init, semf_upper_train_init, self.alpha
                )

                # Evaluate conformalized intervals
                results['train'].append(self.evaluate_model_intervals("SEMF", dfs['train'][self.y_col], semf_lower_train_cqr, semf_upper_train_cqr))
                results['valid'].append(self.evaluate_model_intervals("SEMF", y_calib, semf_lower_calib_cqr, semf_upper_calib_cqr))
                results['test'].append(self.evaluate_model_intervals("SEMF", dfs['test'][self.y_col], semf_lower_test_cqr, semf_upper_test_cqr))

                # Initialize relative metric placeholders in SEMF results
                for subset in results:
                    for r in results[subset]:
                        if r['Model'] == 'SEMF':
                             r['top1_rel_picp'] = default_relative_metric_value
                             r['top1_rel_nmpiw'] = default_relative_metric_value
                             r['top1_rel_cwr'] = default_relative_metric_value
                             r['top1_rel_crps'] = default_relative_metric_value
                             r['top1_rel_pinball'] = default_relative_metric_value

            except Exception as e:
                print(f"Error evaluating SEMF intervals: {e}")
                # Add placeholder NaN results if evaluation fails
                for subset in results:
                    results[subset].append({
                        'Model': 'SEMF', 'PICP': np.nan, 'MPIW': np.nan, 'NMPIW': np.nan,
                        'CWR': np.nan, 'CRPS': np.nan, 'Pinball': np.nan,
                        'top1_rel_picp': default_relative_metric_value, 'top1_rel_nmpiw': default_relative_metric_value,
                        'top1_rel_cwr': default_relative_metric_value, 'top1_rel_crps': default_relative_metric_value,
                        'top1_rel_pinball': default_relative_metric_value
                    })

        # --- 2. Evaluate Baseline Models --- #
        models_to_evaluate = list(self.trained_models.keys())

        # Evaluate only the chosen type of baseline interval method
        if self.baseline_interval_method == "quantile":
            print("Evaluating Quantile Baselines with CQR...")
            quantile_model_keys = [k for k in models_to_evaluate if k.startswith("Quantile_")]
            for model_name in quantile_model_keys:
                print(f"  Evaluating {model_name}...")
                quantile_model = self.get_model(model_name)
                if quantile_model is None:
                    print(f"  Warning: Trained model {model_name} not found.")
                    continue
                try:
                    # Predict initial quantiles on calibration and target sets
                    if model_name == "Quantile_ET" and isinstance(X_calib, pd.DataFrame):
                        # QuantileForest expects numpy
                        pred_calib = quantile_model.predict(X_calib.values, quantiles=self.alphas)
                        pred_test = quantile_model.predict(dfs['test'].drop(self.y_col, axis=1).values, quantiles=self.alphas)
                        pred_train = quantile_model.predict(dfs['train'].drop(self.y_col, axis=1).values, quantiles=self.alphas)
                    elif model_name == "Quantile_MLP": # QNN expects tensors
                        pred_calib = quantile_model.predict(utils.to_tensor(X_calib), quantiles=self.alphas)
                        pred_test = quantile_model.predict(utils.to_tensor(dfs['test'].drop(self.y_col, axis=1)), quantiles=self.alphas)
                        pred_train = quantile_model.predict(utils.to_tensor(dfs['train'].drop(self.y_col, axis=1)), quantiles=self.alphas)
                    else: # Quantile_XGB
                        pred_calib = quantile_model.predict(X_calib) # Assumes XGB predicts both quantiles
                        pred_test = quantile_model.predict(dfs['test'].drop(self.y_col, axis=1))
                        pred_train = quantile_model.predict(dfs['train'].drop(self.y_col, axis=1))

                    lower_calib_init, upper_calib_init = pred_calib[:, 0], pred_calib[:, 1]
                    lower_test_init, upper_test_init = pred_test[:, 0], pred_test[:, 1]
                    lower_train_init, upper_train_init = pred_train[:, 0], pred_train[:, 1]

                    # Apply CQR
                    lower_test_cqr, upper_test_cqr = _apply_cqr(y_calib, lower_calib_init, upper_calib_init, lower_test_init, upper_test_init, self.alpha)
                    lower_calib_cqr, upper_calib_cqr = _apply_cqr(y_calib, lower_calib_init, upper_calib_init, lower_calib_init, upper_calib_init, self.alpha)
                    lower_train_cqr, upper_train_cqr = _apply_cqr(y_calib, lower_calib_init, upper_calib_init, lower_train_init, upper_train_init, self.alpha)

                    # Evaluate
                    results['train'].append(self.evaluate_model_intervals(model_name, dfs['train'][self.y_col], lower_train_cqr, upper_train_cqr))
                    results['valid'].append(self.evaluate_model_intervals(model_name, y_calib, lower_calib_cqr, upper_calib_cqr))
                    results['test'].append(self.evaluate_model_intervals(model_name, dfs['test'][self.y_col], lower_test_cqr, upper_test_cqr))

                except Exception as e:
                    print(f"  Error evaluating {model_name} intervals: {e}")
                    for subset in results:
                         results[subset].append({'Model': model_name, 'PICP': np.nan, 'MPIW': np.nan, 'NMPIW': np.nan, 'CWR': np.nan, 'CRPS': np.nan, 'Pinball': np.nan})

        elif self.baseline_interval_method == "conformal_point":
            print("Evaluating Point Baselines with MAPIE...")
            point_model_keys = [k for k in models_to_evaluate if not k.startswith("Quantile_")]
            for model_name in point_model_keys:
                print(f"  Evaluating {model_name}...")
                mapie_model = self.mapie_models.get(model_name)
                if mapie_model is None:
                    print(f"  Warning: Fitted MAPIE model for {model_name} not found.")
                    continue
                try:
                    # MAPIE directly gives conformal intervals
                    for subset, df in dfs.items():
                        X, y = df.drop(self.y_col, axis=1), df[self.y_col]
                        _, mapie_intervals = mapie_model.predict(X, alpha=self.alpha)
                        # MAPIE returns shape (n_samples, 2, 1) for single alpha -> (n_samples, 2)
                        mapie_lower = mapie_intervals[:, 0, 0]
                        mapie_upper = mapie_intervals[:, 1, 0]
                        results[subset].append(self.evaluate_model_intervals(model_name, y, mapie_lower, mapie_upper))

                except Exception as e:
                    print(f"  Error evaluating {model_name} intervals with MAPIE: {e}")
                    for subset in results:
                         results[subset].append({'Model': model_name, 'PICP': np.nan, 'MPIW': np.nan, 'NMPIW': np.nan, 'CWR': np.nan, 'CRPS': np.nan, 'Pinball': np.nan})

        else:
            print(f"Warning: Unknown baseline_interval_method '{self.baseline_interval_method}'. No baseline intervals evaluated.")


        # --- 3. Calculate Relative Metrics for SEMF --- #
        # Determine the actual baseline model name evaluated based on the chosen method
        if self.baseline_interval_method == "quantile":
            baseline_for_comparison_name = f"Quantile_{self.base_model}"
            baseline_method_label = "CQR"
        elif self.baseline_interval_method == "conformal_point":
            baseline_for_comparison_name = self.base_model
            baseline_method_label = "MAPIE"
        else:
            baseline_for_comparison_name = None # Cannot compare if method is unknown

        if baseline_for_comparison_name:
            for subset in results:
                semf_result = next((r for r in results[subset] if r['Model'] == 'SEMF'), None)
                base_result = next((r for r in results[subset] if r['Model'] == baseline_for_comparison_name), None)

                if semf_result and base_result and not np.isnan(semf_result['CWR']) and not np.isnan(base_result['CWR']):
                    metrics_info = {
                        'PICP': False, # Higher is better
                        'NMPIW': True,  # Lower is better
                        'CWR': False, # Higher is better
                        'CRPS': True,   # Lower is better
                        'Pinball': True # Lower is better
                    }
                    relative_metrics = self.calculate_relative_metrics(semf_result, base_result, metrics_info)
                    # Update the SEMF result dictionary with relative metrics
                    semf_result['top1_rel_picp'] = (relative_metrics.get('relative_PICP', np.nan), "")
                    semf_result['top1_rel_nmpiw'] = (relative_metrics.get('relative_NMPIW', np.nan), "")
                    semf_result['top1_rel_cwr'] = (relative_metrics.get('relative_CWR', np.nan), "")
                    semf_result['top1_rel_crps'] = (relative_metrics.get('relative_CRPS', np.nan), "")
                    semf_result['top1_rel_pinball'] = (relative_metrics.get('relative_Pinball', np.nan), "")
                elif semf_result:
                     # Ensure placeholders exist even if comparison failed
                     semf_result['top1_rel_picp'] = default_relative_metric_value
                     semf_result['top1_rel_nmpiw'] = default_relative_metric_value
                     semf_result['top1_rel_cwr'] = default_relative_metric_value
                     semf_result['top1_rel_crps'] = default_relative_metric_value
                     semf_result['top1_rel_pinball'] = default_relative_metric_value

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

        colors = sns.color_palette("bright", len(self.trained_models))

        indices = np.random.choice(X_eval.shape[0], sample_size, replace=False)
        X_sample = X_eval.iloc[indices]
        y_sample = y_eval.iloc[indices]

        plt.figure(figsize=(12, 8))
        plt.scatter(np.arange(sample_size), y_sample, edgecolor='k', s=100, label="True Value")

        for color_index, model_name in enumerate(self.trained_models):
            impute_method = "original"
            if model_name in ["SEMF", "Quantile_XGB", "Quantile_ET", "Quantile_MLP"]:
                if model_name == "SEMF":
                    lower, upper = self.predict_intervals(model_name, X_sample)
                    label = f"{model_name} Predicted Interval"
                    plt.fill_between(np.arange(sample_size), lower, upper, alpha=0.1, color=colors[color_index], label=label)
                else:
                    lower, upper = self.predict_intervals(model_name, X_sample)
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

    def _compute_crps(self, y, lower, upper):
        """
        Computes the Continuous Ranked Probability Score (CRPS) assuming a uniform distribution
        on [lower, upper].

        Parameters:
        - y (array-like): True target values.
        - lower (array-like): Lower bounds of the predicted intervals.
        - upper (array-like): Upper bounds of the predicted intervals.

        Returns:
        - float: The mean CRPS.
        """
        d = upper - lower
        crps = np.where(np.isclose(d, 0), np.abs(y - lower),
                        np.where(y < lower, ((lower + upper) / 2 - y) - d / 6,
                                 np.where(y > upper, (y - (lower + upper) / 2) - d / 6,
                                          (((y - lower) ** 2 + (upper - y) ** 2) / (2 * d)) - d / 6)))
        return np.mean(crps)

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


    np.random.seed(10)
    n_obs = 1000
    df = pd.DataFrame(np.random.rand(n_obs, 4), columns=['x1', 'x2', 'x3', 'Y'])
    df_train, df_remaining = train_test_split(df, test_size=0.3, random_state=0)
    df_valid, df_test = train_test_split(df_remaining, test_size=0.5, random_state=0)

    mock_semf_model = MockSEMFModel(df_train.drop('Y', axis=1), df_train['Y'])

    print("Benchmarking with complete data...")

    benchmark_semf_complete = BenchmarkSEMF(df_train, df_valid, df_test, semf_model=mock_semf_model, alpha=0.05, test_with_wide_intervals=True, seed=1, inference_R=50, tree_n_estimators=100, device="gpu", nn_batch_size=None, nn_epochs=1000, nn_lr=0.001, nn_load_into_memory=True)

    point_benchmark_complete = benchmark_semf_complete.run_pointpred()
    interval_benchmark_complete = benchmark_semf_complete.run_intervals()
    print("\nPoint Prediction Results with Complete Data:")
    display_results(point_benchmark_complete, sort_descending_by='MAE', include_imputation=True)
    print("\nInterval Prediction Results with Complete Data:")
    display_results(interval_benchmark_complete, sort_descending_by='CWR', include_imputation=True)

    benchmark_semf_complete.plot_predicted_intervals(df_test.drop('Y', axis=1), df_test['Y'], sample_size=50)

    print("\n\nBenchmarking with 50% missing data...")
    df_train_missing = MockSEMFModel.introduce_missing_data(df_train.copy())
    df_valid_missing = MockSEMFModel.introduce_missing_data(df_valid.copy())
    df_test_missing = MockSEMFModel.introduce_missing_data(df_test.copy())

    benchmark_semf_missing = BenchmarkSEMF(df_train_missing, df_valid_missing, df_test_missing, semf_model=mock_semf_model, alpha=0.05, test_with_wide_intervals=True, seed=1, inference_R=50, tree_n_estimators=100, device="gpu", nn_batch_size=None, nn_epochs=500, nn_lr=0.001, nn_load_into_memory=True)
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