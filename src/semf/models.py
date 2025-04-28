import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from joblib import Parallel, delayed
import warnings
import xgboost as xgb
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
import torch.cuda.amp as amp
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
sys.path.append("..")
import utils
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)

# Note: Uncomment and set the start method if necessary
# from torch.multiprocessing import set_start_method
# utils.set_seed(0)
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass

class MultiModelBase:
    """Base class for training multiple models with different parallelization strategies.

    Args:
        parallel_type (str, optional): Type of parallelism. Options are None, 'semf_joblib', and 'model_specific'.
        n_jobs (int, optional): Number of parallel jobs. Defaults to None.
        force_n_jobs (bool, optional): Whether to force the number of jobs specified in `n_jobs`. Defaults to False.
        device (str, optional): Device for training models, 'cpu' or 'gpu'. Defaults to "cpu".
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        nested_parallelism (bool, optional): Whether to enable nested parallelism. Defaults to True.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, parallel_type=None, n_jobs=None, force_n_jobs=False, device="cpu", seed=0, nested_parallelism=True, **kwargs):
        self.models = []
        self.n_jobs = n_jobs
        self.force_n_jobs = force_n_jobs
        self.device = utils.enable_gpu(device)
        self.seed = seed
        self.parallel_type = parallel_type
        self.nested_parallelism = nested_parallelism

    def train_multiple(self, data_list):
        """Train multiple models using the specified parallelization strategy.

        Args:
            data_list (list): List of data dictionaries containing 'inputs', 'outputs', and 'weights'.
        """
        start_time = time.time()
        self._validate_data(data_list)
        
        if not self.force_n_jobs:
            self._adjust_n_jobs(len(data_list))
        
        if self.parallel_type == "semf_joblib" and self.n_jobs > 1:
            job_assignments = self._calculate_job_assignments(len(data_list)) if not self.force_n_jobs else [self.n_jobs] * len(data_list)
            print(f"       ** Job assignments: {job_assignments}")
            self.models = Parallel(n_jobs=self.n_jobs)(
                delayed(self.train_single)(d['inputs'], d['outputs'], d['weights'], n_jobs=job_assignments[i]) for i, d in enumerate(data_list)
            )
        elif self.parallel_type == "model_specific" and self.n_jobs > 1:
            self.models = [self.train_single(d['inputs'], d['outputs'], d['weights'], n_jobs=self.n_jobs) for d in data_list]
        else:
            self.models = [self.train_single(d['inputs'], d['outputs'], d['weights'], n_jobs=None) for d in data_list]
        
        print(f"       ** Training time: {time.time() - start_time:.4f} seconds")

    def predict(self, inputs, model):
        """Predict outputs using a trained model.

        Args:
            inputs (array-like): Input data.
            model: Trained model.

        Returns:
            array: Predictions.
        """
        return model.predict(inputs)

    def predict_multiple(self, new_data_list):
        """Predict outputs for multiple models.

        Args:
            new_data_list (list): List of data dictionaries containing 'inputs'.

        Returns:
            list: List of predictions for each model.
        """
        if not self.models:
            raise ValueError("No models have been trained.")
        return [self.predict(data['inputs'], self.models[i]) for i, data in enumerate(new_data_list)]

    def _adjust_n_jobs(self, n_models):
        """Adjust the number of jobs based on the number of models and available cores.

        Args:
            n_models (int): Number of models to be trained.
        """
        if not self.force_n_jobs:
            available_cores = os.cpu_count() or 1
            self.n_jobs = min(n_models, available_cores - 1)
            print(f"       ** Adjusted n_jobs to {self.n_jobs} to match {n_models} models & {available_cores} available cores. If this is unwanted behaviour, pass `force_n_jobs=True`.")

    def _validate_data(self, data_list):
        """Validate the input data.

        Args:
            data_list (list): List of data dictionaries.

        Raises:
            AssertionError: If inputs, outputs, and weights do not have the same length.
        """
        for data in data_list:
            inputs, outputs, weights = data['inputs'], data['outputs'], data['weights']
            assert len(inputs) == len(outputs) == len(weights), "Inputs, outputs, and weights must have the same length."

    def _calculate_job_assignments(self, num_models):
        """Calculate job assignments for parallel training.

        Args:
            num_models (int): Number of models to be trained.

        Returns:
            list: List of job assignments for each model.
        """
        total_cores = os.cpu_count() or 1
        available_cores = total_cores - 1  # Reserve one core for the main process
        if self.nested_parallelism:
            return [-2] * num_models  # Use nested parallelism: set n_jobs=-2 for each model
        job_assignments = [1] * num_models  # Start with assigning one core per model
        remaining_cores = available_cores - num_models
        for i in range(remaining_cores):
            job_assignments[i % num_models] += 1
        return job_assignments


class MultiXGBs(MultiModelBase):
    """Class for training multiple XGBoost models.

    Args:
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = kwargs.get('tree_n_estimators', 100)
        self.xgb_max_depth = kwargs.get('xgb_max_depth', 5)
        self.xgb_patience = kwargs.get('xgb_patience', 10)
        self.models_val_split = kwargs.get('models_val_split', 0.15)
        print(f"       ** Number of estimators: {self.n_estimators}, Max depth: {self.xgb_max_depth}, Early stopping patience: {self.xgb_patience}, Validation split: {self.models_val_split}")

    def train_single(self, inputs, outputs, weights, n_jobs=None, **kwargs):
        """Train a single XGBoost model.

        Args:
            inputs (array-like): Input data.
            outputs (array-like): Output data.
            weights (array-like): Sample weights.
            n_jobs (int, optional): Number of parallel jobs. Defaults to None.

        Returns:
            xgb.XGBRegressor: Trained XGBoost model.
        """
        utils.set_seed(self.seed)

        # Instead of using a negative value, compute a valid positive number if necessary.
        if n_jobs is None or n_jobs < 0:
            # For example, use all available cores - 2 (but at least 1)
            available_cores = os.cpu_count() or 1
            n_jobs_valid = max(1, available_cores - 2)
        else:
            n_jobs_valid = n_jobs

        if self.models_val_split > 0 and self.xgb_patience > 0:
            X_train, X_val, y_train, y_val, w_train, _ = train_test_split(
                inputs, outputs, weights, test_size=self.models_val_split, shuffle=False
            )
            model = xgb.XGBRegressor(
                tree_method="hist",
                n_estimators=self.n_estimators,
                device=self.device,
                random_state=self.seed,
                n_jobs=n_jobs_valid,
                max_depth=self.xgb_max_depth,
                early_stopping_rounds=self.xgb_patience
            )
            model.fit(X=X_train, y=y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model = xgb.XGBRegressor(
                tree_method="hist",
                n_estimators=self.n_estimators,
                device=self.device,
                random_state=self.seed,
                n_jobs=n_jobs_valid,
                max_depth=self.xgb_max_depth
            )
            model.fit(X=inputs, y=outputs, sample_weight=weights)

        # Call the helper function to print weights
        # utils.print_first_and_last_weights("XGBoost", weights)
        return model


class MultiETs(MultiModelBase):
    """Class for training multiple Extra Trees models.

    Args:
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = kwargs.get('tree_n_estimators', 100)
        self.et_max_depth = kwargs.get('et_max_depth', 10)
        print(f"       ** Number of estimators: {self.n_estimators}, Max depth: {self.et_max_depth}")

    def train_single(self, inputs, outputs, weights, n_jobs=None, **kwargs):
        """Train a single Extra Trees model.

        Args:
            inputs (array-like): Input data.
            outputs (array-like): Output data.
            weights (array-like): Sample weights.
            n_jobs (int, optional): Number of parallel jobs. Defaults to None.

        Returns:
            ExtraTreesRegressor: Trained Extra Trees model.
        """
        utils.set_seed(self.seed)
        model = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            n_jobs=n_jobs,
            max_depth=self.et_max_depth
        )
        model.fit(X=inputs, y=outputs, sample_weight=weights)
        # Call the helper function to print weights
        # utils.print_first_and_last_weights("Extra Trees", weights)
        return model


class NeuralNetwork(nn.Module):
    """Base class for defining a neural network.

    Args:
        input_size (int): Size of the input layer.
        output_size (int): Size of the output layer.
        device (str, optional): Device for training, 'cpu' or 'gpu'. Defaults to "cpu".
    """

    def __init__(self, input_size, output_size, device="cpu"):
        super(NeuralNetwork, self).__init__()
        self.device = torch.device(utils.enable_gpu(device))
        self.layers = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, output_size)
        )

    def forward(self, x, **kwargs):
        """Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.layers(x)


class MLP(NeuralNetwork):
    """Multi-Layer Perceptron class for training neural network models.

    Args:
        input_size (int): Size of the input layer.
        output_size (int): Size of the output layer.
        device (str, optional): Device for training, 'cpu' or 'gpu'. Defaults to "cpu".
    """

    def __init__(self, input_size, output_size, device="cpu"):
        super().__init__(input_size, output_size, device)
        self.data_handler = utils.DataHandler(self.device)

    def train_model(self, inputs, outputs, weights=None, batch_size=None, epochs=5000, lr=0.001, load_into_memory=True, nn_patience=50, val_split=0.15, verbose=False):
        """Train the MLP model.

        Args:
            inputs (array-like): Input data.
            outputs (array-like): Output data.
            weights (array-like, optional): Sample weights. Defaults to None.
            batch_size (int, optional): Batch size for training. Defaults to None.
            epochs (int, optional): Number of epochs for training. Defaults to 5000.
            lr (float, optional): Learning rate. Defaults to 0.001.
            load_into_memory (bool, optional): Whether to load data into memory. Defaults to True.
            nn_patience (int, optional): Early stopping patience. Defaults to 50.
            val_split (float, optional): Validation split ratio. Defaults to 0.15.
            verbose (bool, optional): Verbosity of training process. Defaults to False.

        Returns:
            MLP: Trained MLP model.
        """
        if weights is None:
            weights = torch.ones(len(inputs), device=self.device)

        # Prepare data for training and validation
        X_train, X_val, y_train, y_val, w_train, _ = train_test_split(inputs, outputs, weights, test_size=val_split, shuffle=False)
        train_dataloader = self.data_handler.prepare_data(X_train, y_train, w_train, batch_size=batch_size, load_into_memory=load_into_memory)
        val_dataloader = None
        if X_val is not None and y_val is not None:
            val_dataloader = self.data_handler.prepare_data(X_val, y_val, batch_size=batch_size, load_into_memory=load_into_memory)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss(reduction='none')
        scaler = amp.GradScaler()
        early_stopping = utils.EarlyStopping(patience=nn_patience, verbose=verbose)
        
        self.to(self.device)

        for epoch in range(epochs):
            self.train()
            train_loss = 0
            for batch_inputs, batch_outputs, batch_weights in train_dataloader:
                if verbose == True and epoch == 0:
                    utils.print_first_and_last_weights("MLP first training batch", batch_weights)
                optimizer.zero_grad()
                with amp.autocast(enabled=self.device.type == 'cuda'):
                    predictions = self(batch_inputs)
                    loss = criterion(predictions, batch_outputs)
                    weighted_loss = (loss * batch_weights.unsqueeze(1)).mean()
                scaler.scale(weighted_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += weighted_loss.item()

            if X_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = 0
                    for batch_inputs, batch_outputs in val_dataloader:
                        with amp.autocast(enabled=self.device.type == 'cuda'):
                            predictions = self(batch_inputs)
                            loss = criterion(predictions, batch_outputs).mean()
                        val_loss += loss.item()
                    val_loss /= len(val_dataloader)

                if early_stopping(val_loss, self, epoch):
                    break

            if verbose:
                # if X_val is not None:
                #     print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}")
                # else:
                #     print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.3f}")

                # print the training every 100 epochs
                if X_val is not None and (epoch + 1) % 100 == 0:
                    print(f"       *** Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}")
                elif X_val is None and (epoch + 1) % 100 == 0:
                    print(f"       *** Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.3f}")

        early_stopping.load_best_weights(self)
        return self

    def predict(self, inputs):
        """Predict outputs for given inputs.

        Args:
            inputs (array-like): Input data.

        Returns:
            np.ndarray: Predicted outputs.
        """
        inputs = utils.to_tensor(inputs).to(self.device)
        self.eval()
        with torch.no_grad():
            with amp.autocast(enabled=self.device.type == 'cuda'):
                outputs = self(inputs)
            return outputs.cpu().numpy()


class MultiMLPs(MultiModelBase):
    """Class for training multiple MLP models.

    Args:
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device(self.device)
        self.nn_batch_size = kwargs.get('nn_batch_size', None)
        self.nn_epochs = kwargs.get('nn_epochs', 5000)
        self.nn_lr = kwargs.get('nn_lr', 0.001)
        print(f"       ** Batch size: {self.nn_batch_size}, Epochs: {self.nn_epochs}, Learning rate: {self.nn_lr}")
        self.nn_patience = kwargs.get('nn_patience', 500)
        self.models_val_split = kwargs.get('models_val_split', 0.1)        
        # Ensure nn_load_into_memory is True if nn_batch_size is None
        if self.nn_batch_size is None:
            self.nn_load_into_memory = True
        else:
            self.nn_load_into_memory = kwargs.get('nn_load_into_memory', True)
    
    def train_single(self, inputs, outputs, weights, **kwargs):
        """Train a single MLP model.

        Args:
            inputs (array-like): Input data.
            outputs (array-like): Output data.
            weights (array-like): Sample weights.

        Returns:
            MLP: Trained MLP model.
        """
        utils.set_seed(self.seed)
        model = MLP(inputs.shape[1], outputs.shape[1], device=self.device)
        # Call the helper function to print weights
        # utils.print_first_and_last_weights("MLP", weights)
        
        return model.train_model(
            inputs=inputs,
            outputs=outputs,
            weights=weights,
            batch_size=self.nn_batch_size,
            epochs=self.nn_epochs,
            lr=self.nn_lr,
            load_into_memory=self.nn_load_into_memory,
            nn_patience=self.nn_patience,
            val_split=self.models_val_split,
            verbose=False
        )


class QuantileLoss(nn.Module):
    """Quantile Loss for quantile regression."""

    def __init__(self):
        super(QuantileLoss, self).__init__()

    def forward(self, predictions, targets, tau):
        """Compute quantile loss.

        Args:
            predictions (torch.Tensor): Predictions from the model.
            targets (torch.Tensor): True values.
            tau (float): Quantile to be estimated.

        Returns:
            torch.Tensor: Quantile loss.
        """
        errors = targets - predictions
        return torch.max((tau - 1) * errors, tau * errors).mean()


class QNN(NeuralNetwork):
    """Quantile Neural Network class for quantile regression.

    Args:
        input_size (int): Size of the input layer.
        output_size (int): Size of the output layer.
        device (str): Device for training, 'cpu' or 'gpu'.
    """

    def __init__(self, input_size, output_size, device):
        super().__init__(input_size+1, output_size, device)  # +1 for tau input
        self.data_handler = utils.DataHandler(self.device)
        self.fit = self.train_model

    def train_model(self, x, y, batch_size=64, epochs=5000, lr=0.001, load_into_memory=True, nn_patience=500, val_split=0.1, verbose=True):
        """Train the QNN model.

        Args:
            x (array-like): Input data.
            y (array-like): Output data.
            batch_size (int, optional): Batch size for training. Defaults to 64.
            epochs (int, optional): Number of epochs for training. Defaults to 5000.
            lr (float, optional): Learning rate. Defaults to 0.001.
            load_into_memory (bool, optional): Whether to load data into memory. Defaults to True.
            nn_patience (int, optional): Early stopping patience. Defaults to 500.
            val_split (float, optional): Validation split ratio. Defaults to 0.1.
            verbose (bool, optional): Verbosity of training process. Defaults to True.

        Returns:
            QNN: Trained QNN model.
        """
        if val_split > 0 and val_split < 1:
            X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=val_split, shuffle=False)
        else:
            X_train, y_train = x, y
            X_val, y_val = None, None
        
        train_dataloader = self.data_handler.prepare_data(X_train, y_train, batch_size=batch_size, load_into_memory=load_into_memory)
        if X_val is not None and y_val is not None:
            val_dataloader = self.data_handler.prepare_data(X_val, y_val, batch_size=batch_size, load_into_memory=load_into_memory)

        optimizer = Adam(self.parameters(), lr=lr)
        criterion = QuantileLoss()
        early_stopping = utils.EarlyStopping(patience=nn_patience, verbose=verbose, delta=0)

        self.to(self.device)

        for epoch in range(epochs):
            self.train()
            for x_batch, y_batch in train_dataloader:
                optimizer.zero_grad()
                taus = torch.rand(x_batch.size(0), 1, device=self.device)
                augmented_x = torch.cat((x_batch, taus), dim=1)
                predictions = self(augmented_x)
                loss = criterion(predictions, y_batch, taus)
                loss.backward()
                optimizer.step()

            if X_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = 0
                    for x_batch, y_batch in val_dataloader:
                        taus = torch.rand(x_batch.size(0), 1, device=self.device)
                        augmented_x = torch.cat((x_batch, taus), dim=1)
                        predictions = self(augmented_x)
                        val_loss += criterion(predictions, y_batch, taus).item()
                    val_loss /= len(val_dataloader)
                
                if early_stopping(val_loss, self, epoch): 
                    break

            if verbose:
                if X_val is not None:
                    print(f"       *** Epoch {epoch+1}/{epochs}, Training Loss: {loss:.3f}, Validation Loss: {val_loss:.3f}")
                else:
                    print(f"       *** Epoch {epoch+1}/{epochs}, Training Loss: {loss:.3f}")

        early_stopping.load_best_weights(self)

        return self

    def predict(self, x, quantiles):
        """Predict quantiles for given inputs.

        Args:
            x (array-like): Input data.
            quantiles (list): List of quantiles to predict.

        Returns:
            np.ndarray: Predicted quantiles.
        """
        self.eval()
        # Ensure x is a tensor and on the correct device
        if not isinstance(x, torch.Tensor):
             if isinstance(x, pd.DataFrame):
                 x = x.copy().values
             x = torch.tensor(x, dtype=torch.float32) # Create tensor (likely on CPU first)

        x = x.to(self.device) # Move to the model's device

        predictions = []
        for tau_value in quantiles:
            tau = torch.full((x.size(0), 1), tau_value, device=self.device) # tau is already on self.device
            augmented_x = torch.cat((x, tau), dim=1) # Now both x and tau should be on self.device
            prediction = self(augmented_x)
            predictions.append(prediction.detach().cpu().numpy()) # Move result back to CPU for numpy conversion
        return np.stack(predictions, axis=-1).squeeze()


# Optional implementation for random forest
# Note: Uncomment and implement the MultiRFs class in the future.
# class MultiRFs(MultiModelBase):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.n_estimators = kwargs.get('tree_n_estimators', 100)
#         self.rf_max_depth = kwargs.get('rf_max_depth', 10)

#     def train_single(self, inputs, outputs, weights, n_jobs=None, **kwargs):
#         model = xgb.xgb.XGBRFRegressor(n_estimators=self.n_estimators, random_state=self.seed, n_jobs=n_jobs, max_depth=20)
#         model.fit(X=inputs, y=outputs, sample_weight=weights)
#         return model

############################################################################################################

def main(model_classes, n_train, n_infer, k, parallel_processes, output_dim, parallel_type, device, seed=0):
    """Main function to train and predict using multiple models.

    Args:
        model_classes (list): List of model classes to be used for training.
        n_train (int): Number of training samples.
        n_infer (int): Number of inference samples.
        k (int): Number of different datasets to generate.
        parallel_processes (int): Number of parallel processes.
        output_dim (int): Output dimension of the models.
        parallel_type (str): Type of parallelism. Options are None, 'semf_joblib', and 'model_specific'.
        device (str): Device for training, 'cpu' or 'gpu'.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        list: List of predictions for each model class.
    """
    utils.set_seed(seed)

    data_list = [
        {'inputs': torch.from_numpy(np.random.rand(n_train, 1).astype(np.float32)),
         'outputs': torch.from_numpy(np.random.rand(n_train, output_dim).astype(np.float32)),
         'weights': torch.from_numpy(np.random.rand(n_train).astype(np.float32))}
        for _ in range(k)  # Preparing k different datasets
    ]

    new_data_list = [
        {'inputs': torch.from_numpy(
            np.random.rand(n_infer, 1).astype(np.float32))}
        for _ in range(k)  # Preparing k different datasets
    ]

    # Determine mode of execution
    mode = "parallel" if parallel_type else "normal"

    results = []
    for ModelClass in model_classes:
        model = ModelClass(parallel_type=parallel_type, n_jobs=parallel_processes, device=device)
        print(f"\n{ModelClass.__name__} {mode.capitalize()} Training and Prediction")
        start_time = time.time()
        model.train_multiple(data_list)
        end_time = time.time()
        predictions = model.predict_multiple(new_data_list)

        # Ensure predictions are concatenated and converted to numpy array
        if isinstance(predictions, list):
            if all(isinstance(pred, torch.Tensor) for pred in predictions):
                predictions = torch.cat(predictions, dim=0).numpy()
            elif all(isinstance(pred, np.ndarray) for pred in predictions):
                predictions = np.concatenate(predictions, axis=0)

        results.append(predictions)
        print(f"First few results for {ModelClass.__name__}: {predictions[:3, :3]}")
        print(f" --> {mode.capitalize()} training time: {end_time - start_time:.4f} seconds")
    return results


if __name__ == '__main__':
    warnings.formatwarning = utils.custom_formatwarning
    warnings.simplefilter("once", UserWarning)

    n_train = 10000
    n_infer = n_train // 10
    k = 10
    output_dim = 10
    model_classes = [MultiMLPs]  # MultiMLPs, MultiXGBs, MultiETs
    device = "gpu"  # "cpu" or "gpu"
    parallel_type = None  # None, semf_joblib, model_specific
    parallel_processes = 1

    seed = 0
    num_runs = 2
    all_results_normal = []
    all_results_parallel = []

    print(f"Number of instances: {n_train}, Number of k: {k}, Output dimension: {output_dim}, Number of parallel processes: {parallel_processes}")

    # Normal execution with zero parallelization
    for i in range(num_runs):
        print(f"Normal run {i+1} started.")
        result_normal = main(model_classes=model_classes, n_train=n_train, n_infer=n_infer, k=k,
                             parallel_processes=parallel_processes, output_dim=output_dim, parallel_type=parallel_type, device=device, seed=0)
        all_results_normal.append(result_normal)
        print(f"Normal run {i+1} completed.")
        print("-" * 25)

    # Parallel execution with semf joblib
    # Note: Uncomment for parallel execution
    # for i in range(num_runs):
    #     print(f"Parallel run {i+1} with {parallel_processes} cores has started.")
    #     result_parallel = main(model_classes=model_classes, n_train=n_train, n_infer=n_infer, k=k, 
    #                            parallel_processes=parallel_processes, output_dim=output_dim, parallel_type=parallel_type , device=device, seed=0)
    #     all_results_parallel.append(result_parallel)
    #     print(f"Parallel run {i+1} completed.")
    #     print("-" * 25)

    # ---------------------------------------------------------
    # QNN
    # ---------------------------------------------------------
    utils.set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} for training and predictions.")
    quantiles = [0.1, 0.5, 0.9]

    # QNN: Univariate sine wave data
    n_samples = 10000
    x = torch.linspace(0, 20, n_samples).view(-1, 1).to(device)
    y = torch.sin(x) + 0.1 * torch.randn(n_samples, 1).to(device)

    utils.set_seed(0)
    qnn = QNN(x.shape[1], y.shape[1], device=device)

    utils.set_seed(0)
    model = qnn.train_model(x, y, batch_size=None, epochs=10000, lr=0.001, load_into_memory=True)
    uni_quantiles = qnn.predict(x, quantiles)

    plt.figure(figsize=(15, 6))
    plt.title("Joint Estimation of the Sine Wave Function with Quantile Regression")
    plt.plot(x.cpu().numpy(), y.cpu().numpy(), '.', label='Data')
    plt.plot(x.cpu().numpy(), uni_quantiles[:, 1], label='Median')
    plt.fill_between(x.cpu().numpy().flatten(), uni_quantiles[:, 0], uni_quantiles[:, 2], alpha=0.2, label='10th-90th quantile range')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # QNN: Multivariate synthetic data (with amplitude and noise features)
    # Note: Uncomment below for multivariate synthetic data
    # Generate synthetic data
    # n_samples = 10000
    # x1 = torch.linspace(0, 10, n_samples).unsqueeze(1)  # Main feature for the sine function
    # x2 = torch.rand(n_samples, 1) * 2  # Amplitude variation
    # x3 = torch.randn(n_samples, 1) * 0.1  # Noise
    # x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)
    # x = torch.cat([x1, x2, x3], dim=1).to(device)
    # y = (x2 * torch.sin(x1) + x3).squeeze(1).to(device)  # Output depends on all three features

    # qnn = QNN(x.shape[1], 1, device=device)
    # model = qnn.train_model(x, y, batch_size=64, epochs=10000, lr=0.001, load_into_memory=True)

    # x_test = torch.cat([
    #     torch.linspace(0, 10, 100).unsqueeze(1),
    #     torch.ones(100, 1) * 1,  # Keep amplitude fixed at 1 for testing
    #     torch.zeros(100, 1)  # No noise for testing
    # ], dim=1).to(device)

    # multi_quantiles = qnn.predict(x_test, quantiles)

    # plt.figure(figsize=(10, 6))
    # plt.plot(x_test[:, 0].cpu(), multi_quantiles[:, 1], label='50% quantile (Median)', color='orange')
    # plt.fill_between(x_test[:, 0].cpu().numpy(), multi_quantiles[:, 0], multi_quantiles[:, 2], alpha=0.5, label='10th-90th quantile range', color='gray')
    # plt.scatter(x[:, 0].cpu(), y.cpu(), color='red', s=10, alpha=0.5, label='Data points')
    # plt.title('Multivariate Quantile Regression Predictions')
    # plt.xlabel('Feature x1 (Main feature)')
    # plt.ylabel('Output')
    # plt.legend()
    # plt.show()
