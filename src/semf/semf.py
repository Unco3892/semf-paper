import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.stats as st
from scipy.stats import norm
import time
import json
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append("..")
import utils
from preprocessing import DataPreprocessor
from models import MultiXGBs, MultiETs, MultiMLPs
from scipy.interpolate import interp1d
import pickle

# Model mapping dictionary
MODEL_MAPPING = {"MultiXGBs": MultiXGBs, "MultiETs": MultiETs, "MultiMLPs": MultiMLPs}

class SEMF:
    """
    Class representing SEMF (Supervised Expectation-Maximization Framework).

    Args:
        data_preprocessor: Object that preprocesses the data for the algorithm.
        R (int, optional): Number of (sampling) replications for the algorithm. Default is 10.
        nodes_per_feature (np.array, optional): Array specifying the number of nodes per feature group. Default is [1, 2, 3, 4].
        model_class (str, optional): Class of the model to be used. Default is 'MultiXGBs'.
        tree_config (dict, optional): Dictionary containing configurations for the tree models. Default is None.
        nn_config (dict, optional): Dictionary containing configurations for the neural network model. Default is None.
        models_val_split (float, optional): Proportion of validation set. Default is 0.15.
        parallel_type (str, optional): Parallelization type for training models. Default is 'semf_joblib'.
        device (str, optional): Device for training models, options are 'cpu' or 'gpu'. Default is 'cpu'.
        n_jobs (int, optional): Number of parallel jobs for training. Default is None.
        force_n_jobs (bool, optional): Force the number of parallel jobs specified in `n_jobs`. Default is False.
        max_it (int, optional): Maximum number of iterations for the algorithm. Default is 100.
        stopping_patience (int, optional): Patience iterations for early stopping. Default is 5.
        stopping_metric (str, optional): Metric used for early stopping. Default is 'MAE'.
        custom_sigma_R (float, optional): Custom standard deviation for residual calculations. Fixing the $\\sigma^{*})^2$ for the $p_\\theta$ models (Eq. 15). Default is None.
        z_norm_sd (float, optional): Standard deviation of the normal distribution for latent variable z ($\\sigma_{m_k}$). Default is 1.
        initial_z_norm_sd (float, optional): Initial standard deviation for z when the algorithm starts. Default is None.
        fixed_z_norm_sd (float, optional): Fixed standard deviation for z during updates. The standard deviation of $z$ during updates if other approaches like weighted residuals were used (not relevant an alternative to `z_norm_sd` so it can be disregarded). Default is None.
        return_mean_default (bool, optional): Flag indicating whether to return the mean prediction by default. Default is False.
        mode_type (str, optional): Method for calculating the mode during prediction. If return_mean_default is True, this is ignored. Default is 'approximate'.
        use_constant_weights (bool, optional): Use constant weights across training iterations. Default is False.
        verbose (bool, optional): Verbose output during execution. Default is True.
        x_group_size (int, optional): Size of groups for input features, which essentially controls how many columns should be treated as a single input. Default is 1.
        seed (int, optional): Seed for random number generation to ensure reproducibility. Default is 1.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        data_preprocessor,
        R=10,
        nodes_per_feature=np.array([1, 2, 3, 4]),
        model_class="MultiXGBs",
        tree_config=None,
        nn_config=None,
        models_val_split=0.15,
        parallel_type="semf_joblib",
        device="cpu",
        n_jobs=None,
        force_n_jobs=False,
        max_it=100,
        stopping_patience=5,
        stopping_metric="MAE",
        custom_sigma_R=None,
        z_norm_sd=1,
        initial_z_norm_sd=None,
        fixed_z_norm_sd=None,
        return_mean_default=False,
        mode_type="approximate",
        use_constant_weights=False,
        verbose=True,
        x_group_size=1,
        seed=1,
        **kwargs
    ):
        # Convert types from string to None type
        if custom_sigma_R == "None":
            custom_sigma_R = None
        if initial_z_norm_sd == "None":
            initial_z_norm_sd = None

        # Handle custom_sigma_R conversion
        if custom_sigma_R is not None:
            try:
                custom_sigma_R = float(custom_sigma_R)
            except ValueError:
                raise ValueError(f"custom_sigma_R must be a number, got {custom_sigma_R}")

        # Handle z_norm_sd conversion
        assert isinstance(custom_sigma_R, (type(None), float, int)), f"custom_sigma_R: {custom_sigma_R} <class '{type(custom_sigma_R).__name__}'>"
        valid_strings = ["train_residual_models", "compute_weighted_residuals", "sigma_R"]
        if isinstance(z_norm_sd, str):
            if z_norm_sd not in valid_strings:
                try:
                    z_norm_sd = float(z_norm_sd)
                except ValueError:
                    raise ValueError(f"Invalid z_norm_sd: {z_norm_sd}. Must be one of {valid_strings} or a non-negative number.")
        if isinstance(z_norm_sd, (float, int)):
            assert z_norm_sd >= 0, "z_norm_sd must be a non-negative number."
        elif isinstance(z_norm_sd, str):
            assert z_norm_sd in valid_strings, f"z_norm_sd: {z_norm_sd} must be one of {valid_strings}."
        elif z_norm_sd is not None:
            raise TypeError("z_norm_sd must be a non-negative number, one of the valid strings, or None.")
        assert isinstance(initial_z_norm_sd, (type(None), float, int)), f"initial_z_norm_sd: {initial_z_norm_sd} <class '{type(initial_z_norm_sd).__name__}'>"
        assert isinstance(max_it, int), f"max_it: {max_it} <class '{type(max_it).__name__}'>"
        assert isinstance(mode_type, str), f"mode_type: {mode_type} <class '{type(mode_type).__name__}'>"
        assert isinstance(model_class, str), f"model_class: {model_class} <class '{type(model_class).__name__}'>"
        if isinstance(tree_config, str):
            try:
                tree_config = json.loads(tree_config)
            except json.JSONDecodeError:
                raise ValueError("tree_config must be a valid JSON string representing a dictionary of the tree hyperparameters tree_n_estimators.")
        if isinstance(nn_config, str):
            try:
                nn_config = json.loads(nn_config)
            except json.JSONDecodeError:
                raise ValueError("nn_config must be a valid JSON string representing a dictionary of the nn hyperparameters nn_batch_size, nn_load_into_memory, nn_epochs, nn_lr.")
        assert isinstance(models_val_split, (int, float)), f"models_val_split: {models_val_split} <class '{type(models_val_split).__name__}'>"
        assert isinstance(nodes_per_feature, np.ndarray), f"nodes_per_feature: {nodes_per_feature} <class '{type(nodes_per_feature).__name__}'>"
        assert isinstance(parallel_type, (type(None), str)), f"parallel_type: {parallel_type} <class '{type(parallel_type).__name__}'>"
        assert isinstance(device, str), f"device: {device} <class '{type(device).__name__}'>"
        assert isinstance(n_jobs, (type(None), int)), f"n_jobs: {n_jobs} <class '{type(n_jobs).__name__}'>"
        assert isinstance(force_n_jobs, bool), f"force_n_jobs: {force_n_jobs} <class '{type(force_n_jobs).__name__}'>"
        assert isinstance(return_mean_default, bool), f"return_mean_default: {return_mean_default} <class '{type(return_mean_default).__name__}'>"
        assert isinstance(stopping_metric, str), f"stopping_metric: {stopping_metric} <class '{type(stopping_metric).__name__}'>"
        assert isinstance(stopping_patience, int), f"stopping_patience: {stopping_patience} <class '{type(stopping_patience).__name__}'>"
        assert isinstance(use_constant_weights, bool), f"use_constant_weights: {use_constant_weights} <class '{type(use_constant_weights).__name__}'>"
        assert isinstance(verbose, bool), f"verbose: {verbose} <class '{type(verbose).__name__}'>"
        
        self.seed = seed
        self.df_train, self.df_valid, self.df_test = data_preprocessor.get_train_valid_test()
        for df in [self.df_train, self.df_test, self.df_valid]:
            df.reset_index(drop=True, inplace=True)

        self.x_train, self.y_train = data_preprocessor.split_X_y(self.df_train)
        self.x_valid, self.y_valid = data_preprocessor.split_X_y(self.df_valid)
        self.x_test, self.y_test = data_preprocessor.split_X_y(self.df_test)

        self.R = R
        if model_class not in MODEL_MAPPING:
            raise ValueError(f"Invalid model_class: {model_class}. Available options are {list(MODEL_MAPPING.keys())}.")
        else:
            self.model_class = model_class
        
        self.tree_n_estimators = tree_config.get('tree_n_estimators', 100) if tree_config else 100
        self.xgb_max_depth = tree_config.get('xgb_max_depth', 5) if tree_config else 5
        self.xgb_patience = tree_config.get('xgb_patience', 10) if tree_config else 10
        self.et_max_depth = tree_config.get('et_max_depth', 10) if tree_config else None
        self.nn_batch_size = nn_config.get('nn_batch_size', None) if nn_config else None
        self.nn_load_into_memory = nn_config.get('nn_load_into_memory', True) if nn_config else True
        self.nn_epochs = nn_config.get('nn_epochs', 500) if nn_config else 500
        self.nn_lr = nn_config.get('nn_lr', 0.001) if nn_config else 0.001
        self.nn_patience = nn_config.get('nn_patience', 500) if nn_config else 500
        self.models_val_split = models_val_split

        assert isinstance(self.nn_batch_size, (type(None), int)), "nn_batch_size must be None or an integer"
        assert isinstance(self.nn_load_into_memory, bool), "nn_load_into_memory must be a boolean"
        assert isinstance(self.nn_epochs, int), "nn_epochs must be an integer"
        assert isinstance(self.nn_lr, (float, int)), "nn_lr must be a number"
        self.parallel_type = parallel_type
        self.device = device
        self.n_jobs = n_jobs
        self.force_n_jobs = force_n_jobs
        self.max_it = max_it
        self.stopping_patience = stopping_patience
        self.stopping_metric = stopping_metric

        self.x_group_size = x_group_size
        self.n_features = self.x_train.shape[1]
        self.num_groups = int(np.ceil(self.n_features / self.x_group_size))
        self.n_z_outcomes = nodes_per_feature
        if len(self.n_z_outcomes) != self.num_groups:
            self.n_z_outcomes = self.n_z_outcomes[: self.num_groups]
        self.hidden = np.sum(self.n_z_outcomes)
        self.d = self.n_features + 1
        self.sigmaR_p = 1
        self.start_index = np.concatenate([[0], np.cumsum(self.n_z_outcomes)[:-1]])
        self.end_index = np.cumsum(self.n_z_outcomes) - 1
        self.initial_z_norm_sd = initial_z_norm_sd
        if self.initial_z_norm_sd is None and isinstance(z_norm_sd, (float, int)):
            self.initial_z_norm_sd = z_norm_sd
        elif self.initial_z_norm_sd is None:
            self.initial_z_norm_sd = 1

        self.z_norm_sd = z_norm_sd
        self.fixed_z_norm_sd = fixed_z_norm_sd
        self.custom_sigma_R = custom_sigma_R
        if mode_type not in ["exact", "approximate", "scipy.stats.mode", "quantile_50th"]:
            raise ValueError(f"Invalid mode_type '{mode_type}'. Choose between 'exact' or 'approximate'.")
        self.mode_type = mode_type
        self.use_constant_weights = use_constant_weights
        self.verbose = verbose

        self._return_mean = return_mean_default

        self.x_comp = np.tile(self.x_train, (self.R, 1))
        self.y_comp = np.tile(self.y_train, (self.R, 1))
        self.w_comp = None
        self.y_train_flattened = self.y_train.to_numpy().flatten()
        self.modPhi = []
        self.modTheta = None
        utils.set_seed(self.seed)
        self.modPhi_p = np.random.normal(size=(self.d, self.hidden))
        self.modTheta_p = np.random.normal(size=(sum(self.n_z_outcomes) + 1))

        self.feat_start_indices = np.arange(0, self.num_groups * self.x_group_size, self.x_group_size)
        self.feat_end_indices = np.minimum(self.feat_start_indices + self.x_group_size, self.n_features)
        if (self.x_group_size > 1) and (self.n_features < self.x_group_size):
            self.feat_end_indices[-1] = self.n_features + 1

        self.modelL_perf = []
        self.modelR_perf = []
        self.sigmaR_perf = []
        self.modelL_sigma_perf = []
        self.sigma_z_perf = []
        self.train_perf = []
        self.valid_perf = []
        self.best_simulator = []

    def get_model_instance(self):
        """Get an instance of the model based on the specified model class."""
        return MODEL_MAPPING[self.model_class](parallel_type=self.parallel_type, n_jobs=self.n_jobs, force_n_jobs=self.force_n_jobs, device=self.device, seed=self.seed, tree_n_estimators=self.tree_n_estimators, xgb_max_depth=self.xgb_max_depth, xgb_patience=self.xgb_patience, nn_patience=self.nn_patience, val_split=self.models_val_split, et_max_depth=self.et_max_depth, nn_batch_size=self.nn_batch_size, nn_epochs=self.nn_epochs, nn_lr=self.nn_lr, nn_load_into_memory=self.nn_load_into_memory)

    def train_models(self, data_list, train_multiple=True):
        """Train models based on the provided data list.
        
        Args:
            data_list (list): List of data dictionaries containing 'inputs', 'outputs', and 'weights'.
            train_multiple (bool, optional): If True, train multiple models. Default is True.

        Returns:
            model: Trained model or models.
        """
        model = self.get_model_instance()
        if train_multiple:
            model.train_multiple(data_list)
        else:
            start_time = time.time()
            model = model.train_single(inputs=data_list["inputs"], outputs=data_list["outputs"], weights=data_list["weights"], n_jobs=-2)
            print(f"       ** Training time: {time.time() - start_time:.4f} seconds")
        return model

    def convert_to_tensors(self, inputs, outputs):
        """Convert inputs and outputs to tensors.
        
        Args:
            inputs (array-like): Input data.
            outputs (array-like): Output data.

        Returns:
            tuple: Tensors for inputs, outputs, and weights.
        """
        inputs = utils.to_tensor(inputs)
        outputs = utils.to_tensor(outputs)
        weights = utils.to_tensor(self.w_comp)
        return inputs, outputs, weights

    def generate_data_dict(self, inputs_i, outputs_i):
        """Generate a data dictionary for training.
        
        Args:
            inputs_i (array-like): Input data.
            outputs_i (array-like): Output data.

        Returns:
            dict: Data dictionary containing 'inputs', 'outputs', and 'weights'.
        """
        if self.model_class == "MultiMLPs":
            inputs_i, outputs_i, weights_i = self.convert_to_tensors(inputs_i, outputs_i)
        else:
            weights_i = self.w_comp
        return {"inputs": inputs_i, "outputs": outputs_i, "weights": weights_i}

    @staticmethod
    def print_metrics(r2, rmse, mae, label, indent=0):
        """Print performance metrics.
        
        Args:
            r2 (float): R-squared value.
            rmse (float): Root Mean Squared Error.
            mae (float): Mean Absolute Error.
            label (str): Label for the metrics.
            indent (int, optional): Indentation for the output. Default is 0.
        """
        indent_space = " " * indent
        print(f"{indent_space}{label}")
        print(f"{indent_space}    - R2: {r2:.4f}")
        print(f"{indent_space}    - RMSE: {rmse:.4f}")
        print(f"{indent_space}    - MAE: {mae:.4f}")

    def prepare_phi_inputs(self, outputs):
        """Prepare inputs for the phi models.
        
        Args:
            outputs (array-like): Output data.

        Returns:
            list: List of data dictionaries for the phi models.
        """
        phi_data_list = []
        for group_idx in range(self.num_groups):
            inputs_i = (self.x_comp[:, self.feat_start_indices[group_idx]:self.feat_end_indices[group_idx]]
            )
            outputs_i = outputs[group_idx]
            if outputs_i.ndim == 3:
                outputs_i = utils.flatten_3d_array(outputs_i)
            phi_data_list.append(self.generate_data_dict(inputs_i, outputs_i))
        return phi_data_list

    def assign_phi_models(self):
        """Assign phi models by training on the prepared data."""
        print("    * Training phi models...")
        self.phi_data_list = self.prepare_phi_inputs(outputs=self.z_R_sep)
        self.modPhi = self.train_models(data_list=self.phi_data_list, train_multiple=True)
        self.modPhi_p = self.modPhi.models
        self.modelL_perf.append(self.modPhi_p)

        if self.verbose:
            true_vals = [d["outputs"] for d in self.phi_data_list]
            pred_vals = self.modPhi.predict_multiple(self.phi_data_list)
            true_vals = np.concatenate([arr.flatten() for arr in true_vals])
            pred_vals = np.concatenate([arr.flatten() for arr in pred_vals])
            phi_model_metrics = utils.calculate_performance(true_vals, pred_vals, print_results=False)
            self.print_metrics(phi_model_metrics["R2"], phi_model_metrics["RMSE"], phi_model_metrics["MAE"], " ** Phi Model Metrics", indent=6)

    def assign_theta_model(self):
        """Assign theta model by training on the fused data."""
        print("    * Training theta model...")
        theta_data_list = self.generate_data_dict(self.z_fused_flat, self.y_comp)
        self.modTheta = self.train_models(data_list=theta_data_list, train_multiple=False)
        self.modTheta_p = self.modTheta
        self.modelR_perf.append(self.modTheta_p)

        if self.verbose:
            pred_vals = self.modTheta.predict(self.z_fused_flat)
            theta_model_metrics = utils.calculate_performance(self.y_comp.copy(), pred_vals, print_results=False)
            self.print_metrics(theta_model_metrics["R2"], theta_model_metrics["RMSE"], theta_model_metrics["MAE"], "** Theta Model Metrics", indent=6)

    @staticmethod
    def _compute_weighted_residuals(y_pred, y_true, weights, train_size):
        """Compute the weighted residuals.
        
        Args:
            y_pred (array-like): Predicted values.
            y_true (array-like): True values.
            weights (array-like): Weights for each observation.
            train_size (int): Size of the training set.

        Returns:
            float: Computed standard deviation based on the residuals.
        """
        residuals = y_true - y_pred
        weighted_sum_sq_residuals = np.sum(weights * (residuals**2), axis=0)
        weighted_variance = weighted_sum_sq_residuals / train_size
        return np.sqrt(weighted_variance)

    def predict_phi_models(self, data_to_predict, model):
        """Predict using phi models.
        
        Args:
            data_to_predict (DataFrame): Data to predict.
            model (list): List of trained phi models.

        Returns:
            list: List of predictions for each feature group.
        """
        preds = []
        for group_idx in range(self.num_groups):
            group_model = model[group_idx]
            group_data = data_to_predict.iloc[:, self.feat_start_indices[group_idx]:self.feat_end_indices[group_idx]]
            if isinstance(group_data, pd.DataFrame):
                group_data = group_data.values
            group_prediction = group_model.predict(group_data)
            preds.append(group_prediction)
        return preds

    def predict_sigma_z(self, data_to_predict, model, indices=None):
        """Predict the standard deviation of z.

        Args:
            data_to_predict (DataFrame): Data to predict.
            model (list): List of trained models.
            indices (list, optional): Indices of the features to predict. Default is None.

        Returns:
            list: List of predicted standard deviations for each feature group.
        """
        if self.z_norm_sd == "train_residual_models":
            z_norm_sd_value = self.predict_phi_models(data_to_predict=data_to_predict, model=model)
            if self.fixed_z_norm_sd is not None:
                z_norm_sd_value = [(np.column_stack(([self.fixed_z_norm_sd] * len(array), array[:, 1:])) if array.ndim > 1 else array) for array in z_norm_sd_value]
            z_norm_sd_value = [np.abs(array) for array in z_norm_sd_value]
        else:
            z_norm_sd_value = [np.tile(array, (data_to_predict.shape[0], 1)) for array in self.z_norm_sd_value]
        return z_norm_sd_value

    def compute_sigma_z(self):
        """Compute the standard deviation of z."""
        print("    * Compute Sigma Z...")
        if self.z_norm_sd in ["train_residual_models", "compute_weighted_residuals"]:
            phi_true_vals = [d["outputs"] for d in self.phi_data_list]
            phi_pred_vals = self.modPhi.predict_multiple(self.phi_data_list)
            for index, (phi_true, phi_pred) in enumerate(zip(phi_true_vals, phi_pred_vals)):
                if len(phi_pred.shape) == 1:
                    phi_pred = phi_pred[:, np.newaxis]
                    phi_pred_vals[index] = phi_pred
                if phi_true.shape != phi_pred.shape:
                    raise ValueError(f"Shape mismatch: true values shape {phi_true.shape} and predicted values shape {phi_pred.shape} must be the same.")
            if self.z_norm_sd == "train_residual_models":
                self.residuals = []
                for phi_true, phi_pred in zip(phi_true_vals, phi_pred_vals):
                    z_norm_sd_m_k = np.abs(phi_true - phi_pred)
                    if z_norm_sd_m_k.shape[1] > 1 and self.fixed_z_norm_sd is not None:
                        z_norm_sd_m_k[:, 0] = self.fixed_z_norm_sd
                    self.residuals.append(z_norm_sd_m_k)
                self.data_list_sd = self.prepare_phi_inputs(outputs=self.residuals)
                self.modPhi_sd = self.train_models(data_list=self.data_list_sd, train_multiple=True)
                self.z_norm_sd_value = self.modPhi_sd.models
            elif self.z_norm_sd == "compute_weighted_residuals":
                weights = [d["weights"] for d in self.phi_data_list]
                self.z_norm_sd_value = []
                for phi_true, phi_pred, weight in zip(phi_true_vals, phi_pred_vals, weights):
                    weight_reshaped = weight.reshape(-1, 1)
                    z_norm_sd_m_k = self._compute_weighted_residuals(y_pred=phi_pred, y_true=phi_true, weights=weight_reshaped, train_size=len(self.x_train))
                    if z_norm_sd_m_k.shape[0] > 1 and self.fixed_z_norm_sd is not None:
                        z_norm_sd_m_k[0] = self.fixed_z_norm_sd
                    self.z_norm_sd_value.append(z_norm_sd_m_k)
                ############################################################
                # # Debug: print detailed stats for each group's sigma_z value.
                # print("    * Debug Sigma Z details:")
                # for i, val in enumerate(self.z_norm_sd_value):
                #     try:
                #         print(f"      Group {i}: shape = {val.shape}, min = {np.min(val):.4f}, max = {np.max(val):.4f}, "
                #             f"mean = {np.mean(val):.4f}, std = {np.std(val):.4f}")
                #     except Exception as e:
                #         print(f"      Group {i}: unable to compute stats ({e}).")
                ############################################################
                self.modelL_sigma_perf.append(self.z_norm_sd_value)
        elif self.z_norm_sd == "sigma_R":
            self.z_norm_sd_value = [np.full(size, self.sigmaR_p) for size in self.n_z_outcomes]
        else:
            self.z_norm_sd_value = [np.full(size, self.z_norm_sd) for size in self.n_z_outcomes]
        self.sigma_z_perf.append(self.z_norm_sd_value)

    def compute_sigma_y(self):
        """Compute the standard deviation of y."""
        print("    * Compute Sigma Y...")
        if self.custom_sigma_R is not None:
            self.sigma_R = self.custom_sigma_R
        else:
            theta_true = self.y_comp.flatten()
            theta_pred = self.modTheta.predict(self.z_fused_flat).flatten()
            computed_sigma = self._compute_weighted_residuals(
                y_pred=theta_pred, 
                y_true=theta_true, 
                weights=self.w_comp, 
                train_size=len(self.df_train)
            )
            # Enforce a minimum sigma to prevent division by zero / underflow downstream
            self.sigma_R = max(computed_sigma, 1e-6)
        self.sigmaR_p = self.sigma_R
        print("    * Sigma Y: {:.4f}".format(self.sigmaR_p))
        self.sigmaR_perf.append(self.sigmaR_p)

    def simulate_complete_data(self, data_to_predict, input_length, R=None):
        """Simulate complete data.
        
        Args:
            data_to_predict (DataFrame): Data to predict.
            input_length (int): Length of the input data.
            R (int, optional): Number of (sampling) replications. Default is None.

        Returns:
            list: List of simulated z values.
        """
        if R is None:
            R = self.R
        z_t_selected = []
        if self.i == 1 and isinstance(self.modPhi_p, (np.ndarray, np.generic)):
            z_t_selected = [self.z_t[:, self.start_index[i]:self.end_index[i]+1] for i in range(len(self.n_z_outcomes))]
            z_norm_sd_value = [np.full((df.shape[0],), self.initial_z_norm_sd) for df in z_t_selected]
        else:
            z_t_selected = self.predict_phi_models(data_to_predict=data_to_predict, model=self.modPhi_p)
            z_norm_sd_value = self.predict_sigma_z(data_to_predict=data_to_predict, model=self.z_norm_sd_value)
        z_R_sep = []
        for p in range(len(self.n_z_outcomes)):
            if not isinstance(z_t_selected[p], (np.ndarray, np.generic)):
                z_t_selected[p] = np.array(z_t_selected[p])
            if len(z_t_selected[p].shape) == 1:
                z_t_selected[p] = z_t_selected[p][:, np.newaxis]
            if len(z_norm_sd_value[p].shape) == 1:
                z_norm_sd_value[p] = np.repeat(z_norm_sd_value[p][:, np.newaxis], self.n_z_outcomes[p], axis=1)
            z_R_sep.append(utils.sample_z_r(self.n_z_outcomes[p], z_means=z_t_selected[p], a_size=input_length, sampling_R=R, desired_sd=z_norm_sd_value[p]))
        return z_R_sep

    def compute_yR(self, z_R, y_R, input_length, wide_intervals=False, R=None):
        """Compute y_R values.
        
        Args:
            z_R (array-like): Array of z values.
            y_R (array-like): Array of y values.
            input_length (int): Length of the input data.
            wide_intervals (bool, optional): If True, compute wider intervals. Default is False.
            R (int, optional): Number of (sampling) replications. Default is None.

        Returns:
            array: Computed y_R values.
        """
        if R is None:
            R = self.R
        for r in range(R):
            z_slice = z_R[:, :, r]
            if self.i == 1 and isinstance(self.modTheta_p, (np.ndarray, np.generic)):
                y_R[:, r] = np.dot(np.column_stack((np.ones(input_length), z_slice)), self.modTheta_p).flatten()
            else:
                y_R[:, r] = self.modTheta_p.predict(z_slice).flatten()
                if wide_intervals and R > 1:
                    y_R[:, r] = np.random.normal(loc=y_R[:, r], scale=self.sigmaR_p)
        return y_R

    def compute_weights(self):
        """Compute the weights for the SEMF algorithm."""
        print("     * Computing weights ")
        if self.i == 1 or self.use_constant_weights:
            self.w_R = np.full_like(self.w_R, 1.0 / self.R)
            self.w_comp = self.w_R.flatten()
            # print("       ** The weights are set to constant:", self.w_comp)
            utils.print_first_and_last_weights("SEMF constant", self.w_comp, shape=False)
            return
        for r in range(self.R):
            y_estimate = self.y_R[:, r]
            self.w_R[:, r] = norm.pdf(x=self.y_train_flattened, loc=y_estimate, scale=(1 if self.i == 1 else self.sigma_R))

        if self.i > 1:
            assert self.sigma_R == self.sigmaR_p

        denominator = np.sum(self.w_R, axis=1).reshape(-1, 1)
        epsilon = 1e-20
        if np.any(denominator < epsilon):
            smoothed_denominator = np.where(denominator < epsilon, denominator + epsilon, denominator)
            self.w_R /= smoothed_denominator
            print("       ** Small weights -> smoothened the denominator")
        else:
            self.w_R /= denominator
            print("       ** Normal weights -> non-smoothened denominator")
        self.w_comp = self.w_R.flatten()
        if np.mean(self.w_comp) < epsilon or np.max(self.w_comp) == 0:
            if self.i == 1:
                print("Weights have become too small, making the weights constant for the first iteration!")
                self.w_comp = np.full_like(self.w_R, 1.0 / self.R).flatten()
            else:
                print("Stopping early: weights have become too small!")
                self.zero_weights = True
                self.stopping_rule()

        # print("       ** The weights are:", self.w_comp)
        utils.print_first_and_last_weights("SEMF", self.w_comp, shape=False)

    def stopping_rule(self):
        """Implements the stopping rule for the SEMF algorithm."""
        print("    * Checking for early stopping...")
        if not self.zero_weights:
            self.num_steps_without_improvement = utils.check_early_stopping(self.i, self.valid_perf, self.stopping_metric, self.num_steps_without_improvement)
        if self.num_steps_without_improvement >= self.stopping_patience or self.zero_weights:
            metrics = [x[self.stopping_metric] for x in self.valid_perf]
            if self.stopping_metric in ["R2", "Adjusted_R2"]:
                self.optimal_i = metrics.index(max(metrics))
            else:
                self.optimal_i = metrics.index(min(metrics))
            self.modPhi_p = self.modelL_perf[self.optimal_i]
            self.modTheta_p = self.modelR_perf[self.optimal_i]
            self.sigmaR_p = self.sigmaR_perf[self.optimal_i]
            self.z_norm_sd_value = self.sigma_z_perf[self.optimal_i]
            self.optimal_i += 1
            message = "Returning the best results from iteration " + str(self.optimal_i)
            message_width = len(message)
            print("*" * (message_width + 4))
            print("* " + message + " *")
            print("*" * (message_width + 4))
            self.continue_ = False
            return True
        if self.i >= self.max_it:
            self.continue_ = False
            return True
        if self.num_steps_without_improvement > 0:
            print("      ** Number of steps without improvement", self.num_steps_without_improvement)
        return False

    def train_semf(self):
        """Train the SEMF model."""
        self.continue_ = True
        self.num_steps_without_improvement = 0
        self.zero_weights = False
        self.i = 0
        self.optimal_i = 0
        self.train_input_length = self.x_train.shape[0]
        self.w_R = np.ones((self.train_input_length, self.R))
        self.y_R = np.zeros((self.train_input_length, self.R))
        utils.set_seed(self.seed)

        try:
            while self.continue_:
                self.i += 1
                self.optimal_i += 1
                print(f"- Iteration {self.i}")

                if self.i == 1:
                    self.z_t = np.dot(np.hstack([np.ones((self.train_input_length, 1)), self.x_train]), self.modPhi_p)
                self.z_R_sep = self.simulate_complete_data(data_to_predict=self.x_train, input_length=self.train_input_length)

                self.z_fused = np.concatenate(self.z_R_sep, axis=1)
                self.z_fused_flat = utils.flatten_3d_array(self.z_fused)
                self.y_R = self.compute_yR(self.z_fused, self.y_R, input_length=self.train_input_length, wide_intervals=False)
                self.compute_weights()
                self.assign_phi_models()
                self.assign_theta_model()
                self.compute_sigma_z()
                self.compute_sigma_y()

                print("    * Overall performance of SEMF for iteration {}:".format(self.i))
                train_vals = self.infer_semf(self.x_train, use_wide_intervals=False)
                self.train_perf.append(utils.calculate_performance(self.y_train, train_vals))
                valid_vals = self.infer_semf(self.x_valid.copy(), use_wide_intervals=False)
                self.valid_perf.append(utils.calculate_performance(self.y_valid, valid_vals))
                utils.print_diagnostics(self.train_perf[self.i - 1], self.valid_perf[self.i - 1], indent=6)

                if self.verbose:
                    test_vals = self.infer_semf(self.x_test, use_wide_intervals=False)
                    overall_test_metrics = utils.calculate_performance(self.y_test, test_vals, print_results=False)
                    self.print_metrics(overall_test_metrics["R2"], overall_test_metrics["RMSE"], overall_test_metrics["MAE"], " ** Test model metrics", indent=5)

                print("#------------------------------------------------------------------------------#")
                if self.stopping_rule():
                    if self.zero_weights:
                        print("Stopping early: weights have become too small!")
                    else:
                        print("Stopping early: no further improvement detected!")
                    print(f"Returning the best results from iteration {self.i}")
                    return self

        except Exception as e:
            print(f"Encountered error during training iteration: {e}")
            print("Returning model from the last successful iteration.")
            return self
        return self

    def compute_mode_exact(self, data_to_predict, y_R_infer, final_sd):
        """Compute the exact mode of the predicted data.
        
        Args:
            data_to_predict (DataFrame): Data to predict.
            y_R_infer (array-like): Inferred values of y_R.
            final_sd (float): Final standard deviation.

        Returns:
            array: Computed mode values.
        """
        y_seq = np.linspace(np.min(self.y_R), np.max(self.y_R), num=data_to_predict.shape[0])
        densities = np.zeros((y_R_infer.shape[0], len(y_seq)))
        for i in range(y_R_infer.shape[0]):
            densities[i, :] = np.mean(norm.pdf(x=y_seq[:, np.newaxis], loc=y_R_infer[i, :], scale=final_sd), axis=1)
        mode_indices = np.argmax(densities, axis=1)
        return y_seq[mode_indices]

    @property
    def return_mean(self):
        """bool: Return the mean prediction by default."""
        return self._return_mean

    @return_mean.setter
    def return_mean(self, value):
        """Set the return_mean property.
        
        Args:
            value (bool): Boolean value to set the return_mean property.
        """
        if not isinstance(value, bool):
            raise ValueError("'return_mean' should be a boolean value.")
        self._return_mean = value

    def compute_mode_approximate(self, data_to_predict, y_R_infer, final_sd, subset_size=100):
        """Compute the approximate mode of the predicted data.
        
        Args:
            data_to_predict (DataFrame): Data to predict.
            y_R_infer (array-like): Inferred values of y_R.
            final_sd (float): Final standard deviation.
            subset_size (int, optional): Size of the subset used for density calculation. Default is 100.

        Returns:
            array: Computed mode values.
        """
        y_seq = np.linspace(np.min(self.y_R), np.max(self.y_R), num=data_to_predict.shape[0])
        subset_indices = np.unique(np.round(np.linspace(0, len(y_seq) - 1, subset_size)).astype(int))
        y_seq_subset = y_seq[subset_indices]
        densities_subset = np.zeros((y_R_infer.shape[0], len(y_seq_subset)))
        for i in range(y_R_infer.shape[0]):
            densities_subset[i, :] = np.mean(norm.pdf(x=y_seq_subset[:, np.newaxis], loc=y_R_infer[i, :], scale=final_sd), axis=1)
        densities = np.zeros((y_R_infer.shape[0], len(y_seq)))
        for i in range(y_R_infer.shape[0]):
            interpolator = interp1d(y_seq_subset, densities_subset[i, :], kind="linear", fill_value="extrapolate")
            densities[i, :] = interpolator(y_seq)
        mode_indices = np.argmax(densities, axis=1)
        return y_seq[mode_indices]

    @staticmethod
    def compute_modes_quantile(y_R_infer, quantile=0.5):
        """Compute the quantile-based modes.
        
        Args:
            y_R_infer (array-like): Inferred values of y_R.
            quantile (float, optional): Quantile value for mode calculation. Default is 0.5.

        Returns:
            array: Computed quantile-based modes.
        """
        return np.quantile(y_R_infer, quantile, axis=1)

    def infer_semf(self, data_to_predict, return_type="point", use_wide_intervals=False, infer_seed=None, R=None):
        """Perform inference on the given data.
        
        Args:
            data_to_predict (DataFrame): Data for prediction.
            return_type (str, optional): Controls the return type ('point', 'interval', or 'both'). Default is 'point'.
            use_wide_intervals (bool, optional): Controls whether to use wide prediction intervals. Default is False.
            infer_seed (int, optional): Random seed for generating different values. Default is None.
            R (int, optional): Number of samples to use for inference. Default is None.

        Returns:
            array: Point predictions, prediction intervals, or both, depending on return_type.
        """
        assert return_type in ["point", "interval", "both"], "return_type must be 'point', 'interval', or 'both'"
        if infer_seed is None:
            infer_seed = self.seed
        utils.set_seed(infer_seed)

        if R is None:
            R = self.R

        infer_input_length = data_to_predict.shape[0]
        z_R_sep_infer = self.simulate_complete_data(data_to_predict=data_to_predict, input_length=infer_input_length, R=R)

        z_R_infer = np.concatenate(z_R_sep_infer, axis=1)
        y_R_infer = np.zeros((infer_input_length, R))
        y_R_infer = self.compute_yR(z_R_infer, y_R_infer, input_length=infer_input_length, wide_intervals=use_wide_intervals, R=R)

        if return_type == "interval":
            return y_R_infer
        else:
            if self.return_mean:
                point_pred = np.mean(y_R_infer, axis=1)
            else:
                if self.mode_type == "exact":
                    point_pred = self.compute_mode_exact(data_to_predict, y_R_infer, self.sigmaR_p)
                elif self.mode_type == "approximate":
                    point_pred = self.compute_mode_approximate(data_to_predict, y_R_infer, self.sigmaR_p)
                elif self.mode_type == "quantile_50th":
                    point_pred = self.compute_modes_quantile(y_R_infer, 0.5)
                elif self.mode_type == "scipy.stats.mode":
                    point_pred = st.mode(y_R_infer, axis=1).mode.flatten()
            if return_type == "point":
                return point_pred
            elif return_type == "both":
                return point_pred, y_R_infer

    def save_semf(self, data_preprocessor, ds_name, base_dir):
        """
        Save the SEMF model and data preprocessor to the specified directory.

        Args:
            data_preprocessor: The data preprocessor to be saved.
            ds_name (str): The name of the dataset.
            base_dir (str): The base directory where models will be saved.
        """

        # Define the paths for the model files
        models_dir = os.path.join(base_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"semf_{ds_name}.pkl")
        data_preprocessor_path = os.path.join(models_dir, f"data_preprocessor_{ds_name}.pkl")

        # Save the model and data preprocessor
        with open(model_path, "wb") as f:
            pickle.dump(self, f)
        with open(data_preprocessor_path, "wb") as f:
            pickle.dump(data_preprocessor, f)

    @staticmethod
    def load_semf(ds_name, base_dir):
        """
        Load the SEMF model and data preprocessor from the specified directory.

        Args:
            ds_name (str): The name of the dataset.
            base_dir (str): The base directory where models are saved.

        Returns:
            SEMF: The loaded SEMF model.
            DataPreprocessor: The loaded data preprocessor.
        """
        models_dir = os.path.join(base_dir, "models")
        model_path = os.path.join(models_dir, f"semf_{ds_name}.pkl")
        data_preprocessor_path = os.path.join(models_dir, f"data_preprocessor_{ds_name}.pkl")

        # Load the model and data preprocessor
        with open(model_path, "rb") as f:
            semf = pickle.load(f)
        with open(data_preprocessor_path, "rb") as f:
            data_preprocessor = pickle.load(f)
        
        return semf, data_preprocessor
    
if __name__ == "__main__":
    # np.random.seed(10)
    utils.set_seed(10)
    n_R = 10
    n_obs = 10000
    df = pd.DataFrame(np.random.rand(n_obs, 4), columns=["x1", "x2", "x3", "y"])
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
    data_preprocessor = DataPreprocessor(df_train, y_col="y", complete_x_col="x1", rate=0)
    data_preprocessor.split_data(df_test)
    data_preprocessor.scale_data(scale_output=True)
    df_train, df_valid, df_test = data_preprocessor.get_train_valid_test()

    semf = SEMF(data_preprocessor, R=n_R, nodes_per_feature=np.array([3, 4, 10]), seed=10, z_norm_sd=0.01, return_mean_default=True, stopping_metric="RMSE", stopping_patience=5, max_it=100, verbose=True)
    st = time.time()
    result = semf.train_semf()
    if result is not None:
        print(result)
    et = time.time()
    elapsed_time = et - st
    print("-----------------------------------------")
    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print("-----------------------------------------")
