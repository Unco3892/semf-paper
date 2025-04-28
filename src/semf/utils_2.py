import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from openml import datasets
import os
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
import json
from argparse import ArgumentTypeError
import warnings
import copy
from sklearn.model_selection import train_test_split
shown_warnings = set()

def sample_z_r(n_outcomes, z_means, a_size, sampling_R, desired_sd):
    """
    Sample latent variable Z_R from a normal distribution.
    
    Args:
        n_outcomes (int): Number of outcomes.
        z_means (array-like): Mean values for Z.
        a_size (int): Size of the array.
        sampling_R (int): Number of samples to take.
        desired_sd (array-like): Desired standard deviation.
    
    Returns:
        array: Sampled Z_R values.
    """
    # Ensure inputs are numpy arrays with proper dimensions
    z_means = np.asarray(z_means)
    desired_sd = np.asarray(desired_sd)
    
    # Check that input shapes are consistent
    if z_means.shape[0] != a_size:
        print(f"Warning: z_means shape[0] ({z_means.shape[0]}) does not match a_size ({a_size})")
        if z_means.shape[0] != desired_sd.shape[0]:
            print(f"Warning: z_means shape[0] ({z_means.shape[0]}) does not match desired_sd shape[0] ({desired_sd.shape[0]})")
    
    # Ensure positive standard deviations to avoid numerical issues
    desired_sd = np.maximum(desired_sd, 1e-6)
    
    # Check for NaN or Inf values
    if np.any(np.isnan(z_means)) or np.any(np.isinf(z_means)):
        print("Warning: NaN or Inf values found in z_means, replacing with zeros")
        z_means = np.nan_to_num(z_means, nan=0.0, posinf=0.0, neginf=0.0)
        
    if np.any(np.isnan(desired_sd)) or np.any(np.isinf(desired_sd)):
        print("Warning: NaN or Inf values found in desired_sd, replacing with ones")
        desired_sd = np.nan_to_num(desired_sd, nan=1.0, posinf=1.0, neginf=1.0)

    # Initialize the output array
    desired_z_R = np.zeros((a_size, n_outcomes, sampling_R))
    
    # Generate the samples more safely
    for l in range(n_outcomes):
        for r in range(sampling_R):
            try:
                # Get the specific mean and std for this dimension
                # Make sure we're using the right shape
                mean_col = z_means[:, l] if z_means.shape[0] == a_size else None
                sd_col = desired_sd[:, l] if desired_sd.shape[0] == a_size else None
                
                if mean_col is not None and sd_col is not None:
                    # Use normal distribution with means and standard deviations
                    desired_z_R[:, l, r] = np.random.normal(
                        loc=mean_col,
                        scale=sd_col,
                        size=a_size
                    )
                else:
                    # Fallback: generate random samples with global parameters
                    print(f"Using fallback random generation for l={l}, r={r}")
                    # Use mean of z_means column if available
                    loc = np.mean(z_means[:, l]) if z_means.shape[1] > l else 0.0
                    # Use mean of desired_sd column if available  
                    scale = np.mean(desired_sd[:, l]) if desired_sd.shape[1] > l else 1.0
                    desired_z_R[:, l, r] = np.random.normal(
                        loc=loc,
                        scale=scale,
                        size=a_size
                    )
            except Exception as e:
                print(f"Error generating samples for l={l}, r={r}: {e}")
                # Fallback with a safer approach that won't try to broadcast incompatible shapes
                if z_means.shape[0] == a_size:
                    desired_z_R[:, l, r] = z_means[:, l]
                else:
                    # If shapes don't match, generate values with mean of column
                    print(f"Shape mismatch, using column mean for l={l}, r={r}")
                    mean_val = np.mean(z_means[:, l]) if z_means.shape[1] > l else 0.0
                    desired_z_R[:, l, r] = np.full(a_size, mean_val)

    return desired_z_R

def reshape_z_t(mat, n_rows):
    """
    Reshapes a 2D matrix into a 3D array.

    Args:
        mat (np.ndarray): A 2D array to be reshaped.
        n_rows (int): The number of rows in each slice of the output array.

    Returns:
        np.ndarray: A 3D array with dimensions (n_rows, ncol, n_slices).
    """
    ncol = mat.shape[1]
    n_slices = mat.shape[0] // n_rows

    if n_slices != n_slices:
        raise ValueError("Number of rows must be a multiple of original number of rows")

    array_3d = np.zeros((n_rows, n_slices, ncol))
    for i in range(mat.shape[0]):
        row_idx = i % n_rows
        slice_idx = i // n_rows
        for j in range(ncol):
            array_3d[row_idx, slice_idx, j] = mat[i, j]

    array_3d = np.transpose(array_3d, (0, 2, 1))

    return array_3d

def set_seed(seed):
    """
    Sets the random seeds for various libraries (os, numpy, random, torch) to ensure reproducibility.

    Args:
        seed (int): The seed value to use for all random number generators.

    Returns:
        None
    """
    np.random.seed(seed)
    random.seed(seed)
        
    # Set seeds for Torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Additional flags for deterministic behavior
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HOROVOD_FUSION_THRESHOLD"] = "0"
    torch.use_deterministic_algorithms(True)

    # Set seeds for Python libraries
    os.environ['PYTHONHASHSEED'] = str(seed)

def sample_z_r_reshaped(z_R_means, desired_sd):
    """
    Samples a 3D array of size (a_size, n_hiddens, sampling_R) from a normal distribution with mean z_R_means and standard deviation desired_sd.

    Args:
        z_R_means (np.ndarray): A 3D array of size (a_size, n_hiddens, sampling_R) containing the means of the normal distribution.
        desired_sd (np.ndarray): A 3D array of standard deviations of the normal distribution.

    Returns:
        np.ndarray: A 3D array of size (a_size, n_hiddens, sampling_R) containing the samples.

    Raises:
        ValueError: If the input is not a 3D array.
    """
    if len(z_R_means.shape) != 3:
        raise ValueError("Error: z_R_means must be a 3D array")

    a_size, n_hiddens, sampling_R = z_R_means.shape

    desired_z_R = np.copy(z_R_means)
    for l in range(n_hiddens):
        for r in range(sampling_R):
            desired_z_R[:, l, r] = np.random.normal(
                loc=z_R_means[:, l, r],
                scale=desired_sd[:, l, r],
                size=a_size
            )
    return desired_z_R

def print_diagnostics(train_perf, valid_perf, tabular_format=True, metrics=["R2", "RMSE", "MAE"], indent=0):
    """
    Prints the training and validation performance metrics.

    Args:
        train_perf (dict): A dictionary containing the training performance metrics.
        valid_perf (dict): A dictionary containing the validation performance metrics.
        tabular_format (bool, optional): Whether to print the metrics in a tabular format. Defaults to True.
        metrics (list, optional): The list of metrics to print. Defaults to ["R2", "RMSE", "MAE"].
        indent (int, optional): The number of spaces to indent the output. Defaults to 0.

    Returns:
        None
    """
    indent_space = ' ' * indent
    
    if tabular_format:    
        train_metrics = {metric: round(train_perf.get(metric, float('nan')), 4) for metric in metrics}
        valid_metrics = {metric: round(valid_perf.get(metric, float('nan')), 4) for metric in metrics}

        diagnostics = pd.DataFrame([train_metrics, valid_metrics], index=["Train", "Validation"])
        table_str = diagnostics.to_string()
        lines = table_str.split('\n')
        table_width = len(lines[0])
        
        print(f"{indent_space}+{'-' * (table_width)}+")
        for line in lines:
            print(f"{indent_space}|{line}|")
        print(f"{indent_space}+{'-' * (table_width)}+")
    else:
        metrics_str_train = ', '.join([f"{metric}: {round(train_perf.get(metric, float('nan')), 4)}" for metric in metrics])
        metrics_str_valid = ', '.join([f"{metric}: {round(valid_perf.get(metric, float('nan')), 4)}" for metric in metrics])

        print(f"{indent_space}Overall Train [{metrics_str_train}], Validation [{metrics_str_valid}]")

def check_early_stopping(i, valid_perf, stopping_metric, num_steps_without_improvement):
    """
    Checks if the validation performance has stopped improving and returns the number of steps without improvement.

    Args:
        i (int): The current iteration.
        valid_perf (list): A list containing the validation performance metrics.
        stopping_metric (str): The metric to use for early stopping. Must be one of "MAE", "RMSE" or "R2".
        num_steps_without_improvement (int): The current number of steps without improvement.

    Returns:
        int: The updated number of steps without improvement.
    """
    if i > 2:
        last_metrics = [perf[stopping_metric] for perf in valid_perf[0:(i-1)]]
        current_metric = valid_perf[i-1][stopping_metric]

        if stopping_metric in ["R2", "Adjusted_R2"]:
            is_improving = current_metric > max(last_metrics)
        else:
            is_improving = current_metric < min(last_metrics)

        num_steps_without_improvement = 0 if is_improving else num_steps_without_improvement + 1

    return num_steps_without_improvement

def calculate_performance(y_true, y_pred, x=None, print_results=False, indent=0):
    """
    Calculates the performance metrics (MAE, RMSE, R2, and Adjusted R2) for the given true and predicted values.

    Args:
        y_true (np.ndarray): The true values.
        y_pred (np.ndarray): The predicted values.
        x (np.ndarray, optional): The input features used for prediction. Defaults to None.
        print_results (bool, optional): Whether to print the performance metrics. Defaults to False.
        indent (int, optional): The number of spaces to indent the output. Defaults to 0.

    Returns:
        dict: A dictionary containing the performance metrics.
    """
    indent_space = ' ' * indent
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    performance_metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    
    print_str = f"{indent_space}MAE = {mae:.4f}, RMSE = {rmse:.4f}, R2 = {r2:.4f}"
    
    if x is not None:
        n = len(y_true)
        k = x.shape[1]
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
        print_str += f", Adjusted R2 = {adj_r2:.4f}"
        performance_metrics['Adjusted_R2'] = adj_r2
    
    if print_results:
        print(print_str)

    return performance_metrics

def flatten_3d_array(arr):
    """
    Flattens a 3D array into a 2D array.

    Args:
        arr (np.ndarray): A 3D array to be flattened.

    Returns:
        np.ndarray: A 2D array.
    """
    split_arrays = np.split(arr, arr.shape[2], axis=2)
    squeezed_arrays = [np.squeeze(a, axis=2) for a in split_arrays]
    return np.vstack(squeezed_arrays)

def format_model_metrics(data):
    """
    Formats model metrics into a dictionary suitable for logging.

    Args:
        data (dict): Dictionary containing the datasets and their corresponding model metrics.

    Returns:
        dict: Dictionary with formatted metric names and their values.
    """
    formatted_metrics = {}
    for dataset_name, models in data.items():
        best_metrics = {}
        for model_data in models:
            model_name = model_data['Model']
            imputation = model_data.get('Imputation', 'Original')

            for metric_name, value in model_data.items():
                if metric_name not in ['Model', 'Imputation']:
                    formatted_metrics[f"{dataset_name}_{metric_name}_{model_name}_{imputation}"] = value

                    if isinstance(value, tuple):
                        value, imputation_method = value
                        metric_key = (dataset_name, metric_name, model_name)
                        if metric_key not in best_metrics or value > best_metrics[metric_key][0]:
                            best_metrics[metric_key] = (value, imputation_method)

        for metric_key, best_value in best_metrics.items():
            dataset_name, metric_name, model_name = metric_key

            if isinstance(best_value, tuple) and len(best_value) == 2:
                value, best_imputation = best_value
                formatted_metrics[f"{dataset_name}_{metric_name}_{model_name}"] = value
                formatted_metrics[f"{dataset_name}_{metric_name}_{model_name}_imputation"] = best_imputation
            else:
                formatted_metrics[f"{dataset_name}_{metric_name}_{model_name}"] = best_value
                formatted_metrics[f"{dataset_name}_{metric_name}_{model_name}_imputation"] = "Unknown"

    return formatted_metrics

def load_openml_dataset(ds_name, dataset_names, cache_dir):
    """
    Load a dataset from OpenML by its name and cache it to a specified directory.
    
    Args:
        ds_name (str): Name of the dataset.
        dataset_names (dict): Dictionary mapping dataset names to OpenML dataset IDs.
        cache_dir (str): Directory to cache the dataset.

    Returns:
        tuple: DataFrame containing the dataset with the target column as the last column and the name of the target column.
    """
    ds_id = dataset_names[ds_name]
    dataset = datasets.get_dataset(ds_id, download_data=True, download_qualities=True, download_features_meta_data=True)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name=dataset.default_target_attribute)

    data = pd.concat([X, y], axis=1)
    file_path = os.path.join(cache_dir, f'{ds_name}.csv')
    data.to_csv(file_path, index=False)

    return data, dataset.default_target_attribute

def print_data_completeness(df_train, df_valid, df_test):
    """
    Prints the data completeness ratio for each dataset.

    Args:
        df_train (pd.DataFrame): The training dataset.
        df_valid (pd.DataFrame): The validation dataset.
        df_test (pd.DataFrame): The test dataset.

    Returns:
        None
    """
    print("Train data completeness: ", len(df_train.dropna()) / len(df_train))
    print("Validation data completeness: ", len(df_valid.dropna()) / len(df_valid))
    print("Test data completeness: ", len(df_test.dropna()) / len(df_test))

def load_json_config(provided_config, default_config):
    """
    Load a JSON configuration with default fallbacks.

    Args:
        provided_config (str): The provided JSON configuration string.
        default_config (dict): The default configuration dictionary.

    Returns:
        dict: The updated configuration dictionary.

    Raises:
        ArgumentTypeError: If the provided JSON configuration is invalid.
    """
    try:
        updated_config = json.loads(provided_config)
        default_config.update(updated_config)
        return default_config
    except json.JSONDecodeError as e:
        raise ArgumentTypeError(f"Invalid JSON format: {provided_config}")

def add_boolean_argument(parser, name, default, help_true, help_false):
    """
    Add a boolean argument to an argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser.
        name (str): The name of the argument.
        default (bool): The default value of the argument.
        help_true (str): The help message for the true value.
        help_false (str): The help message for the false value.

    Returns:
        None
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--{name}", dest=name, action='store_true', default=default, help=help_true
    )
    group.add_argument(
        f"--no-{name}", dest=name, action='store_false', help=help_false
    )

def to_tensor(data):
    """
    Convert inputs and outputs to PyTorch tensors.

    Args:
        data (Union[np.ndarray, pd.DataFrame, torch.Tensor]): The data to convert.

    Returns:
        torch.Tensor: The converted data as a PyTorch tensor.
    """
    if isinstance(data, pd.DataFrame):
        return torch.tensor(data.values, dtype=torch.float32)
    if isinstance(data, torch.Tensor):
        return data.clone().detach().float()
    return torch.tensor(data, dtype=torch.float32)

def showwarning_once(message, category, filename, lineno, file=None, line=None):
    """
    Custom show warning function that displays each warning only once per session based on
    the message and category.

    Args:
        message (str): The warning message.
        category (Warning): The category of the warning.
        filename (str): The file where the warning occurred.
        lineno (int): The line number where the warning occurred.
        file (file, optional): The file object to write the warning message. Defaults to None.
        line (str, optional): The line of code where the warning occurred. Defaults to None.

    Returns:
        None
    """
    warning_key = (str(message), category)
    if warning_key not in shown_warnings:
        shown_warnings.add(warning_key)
        warnings._showwarning_orig(message, category, filename, lineno, file, line)

def custom_formatwarning(msg, category, *args, **kwargs):
    """
    Custom formatter for warnings that omits file location, line number, and code context. 
    It formats the warning to include only the message and the category of the warning.
    
    Args:
        msg (str): The warning message.
        category (Warning): The category of the warning.
        *args: Variable length argument list (not used in this formatter).
        **kwargs: Arbitrary keyword arguments (not used in this formatter).

    Returns:
        str: Formatted warning string.
    """
    return f'{category.__name__}: {msg}\n'

warnings._showwarning_orig = warnings.showwarning
warnings.showwarning = showwarning_once
warnings.formatwarning = custom_formatwarning

def enable_gpu(device):
    """
    Enable GPU for computation if available.

    Args:
        device (str): The device to use for computation ('cpu', 'gpu', or 'cuda').

    Returns:
        str: The device to use for computation.
    """
    if device in ['gpu', 'cuda']:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            warnings.warn("       ** CUDA is not available. Using CPU instead.", UserWarning)
            device = "cpu"
    return device

class CustomDataset:
    """
    A custom dataset class for loading data in batches.

    Args:
        inputs (torch.Tensor): The input data.
        outputs (torch.Tensor): The output data.
        weights (torch.Tensor, optional): The weights for each sample. Defaults to None.
        batch_size (int, optional): The batch size for loading data. Defaults to 64.
    """
    def __init__(self, inputs, outputs, weights=None, batch_size=64):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = weights
        self.batch_size = batch_size

    def __len__(self):
        """
        Returns the number of batches in the dataset.

        Returns:
            int: The number of batches.
        """
        return (len(self.inputs) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """
        Returns an iterator over the batches of data.

        Yields:
            tuple: A tuple containing the batch inputs, outputs, and weights (if available).
        """
        for start_index in range(0, len(self.inputs), self.batch_size):
            end_index = start_index + self.batch_size
            batch_inputs = self.inputs[start_index:end_index]
            batch_outputs = self.outputs[start_index:end_index]
            if self.weights is not None:
                batch_weights = self.weights[start_index:end_index]
                yield batch_inputs, batch_outputs, batch_weights
            else:
                yield batch_inputs, batch_outputs

class DataHandler:
    """
    A class for handling data preparation and loading.

    Args:
        device (str): The device to use for computation ('cpu' or 'cuda').
    """
    def __init__(self, device):
        self.device = device

    def prepare_data(self, inputs, outputs, weights=None, batch_size=None, load_into_memory=True, num_workers=4):
        """
        Prepare data for loading into a PyTorch DataLoader.

        Args:
            inputs (Union[np.ndarray, pd.DataFrame]): The input data.
            outputs (Union[np.ndarray, pd.DataFrame]): The output data.
            weights (Union[np.ndarray, pd.DataFrame], optional): The weights for each sample. Defaults to None.
            batch_size (int, optional): The batch size for loading data. Defaults to the number of inputs.
            load_into_memory (bool, optional): Whether to load the data into memory. Defaults to True.
            num_workers (int, optional): The number of worker threads for loading data. Defaults to 4.

        Returns:
            DataLoader or CustomDataset: The prepared data loader.
        """
        batch_size = batch_size or len(inputs)
        optimize_gpu = self.device == "cuda"
        
        if isinstance(inputs, (pd.DataFrame, pd.Series)):
            inputs = torch.tensor(inputs.values, dtype=torch.float32)
        if isinstance(outputs, (pd.DataFrame, pd.Series)):
            outputs = torch.tensor(outputs.values, dtype=torch.float32)
        if weights is not None and isinstance(weights, (pd.DataFrame, pd.Series)):
            weights = torch.tensor(weights.values, dtype=torch.float32)
        
        inputs = inputs.to(self.device, non_blocking=optimize_gpu)
        outputs = outputs.to(self.device, non_blocking=optimize_gpu)
        weights = weights.to(self.device, non_blocking=optimize_gpu) if weights is not None else None

        if load_into_memory:
            return CustomDataset(inputs=inputs, outputs=outputs, weights=weights, batch_size=batch_size)
        else:
            dataset = TensorDataset(inputs, outputs, *(weights,)) if weights is not None else TensorDataset(inputs, outputs)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=optimize_gpu, num_workers=num_workers)

class EarlyStopping:
    """
    A class for implementing early stopping in training models.

    Args:
        patience (int, optional): The number of epochs to wait for improvement before stopping. Defaults to 10.
        verbose (bool, optional): Whether to print early stopping messages. Defaults to True.
        delta (float, optional): Minimum change to qualify as an improvement. Defaults to 0.
    """
    def __init__(self, patience=10, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = float('inf')
        self.best_model = None
        self.wait = 0
        self.stopped_epoch = 0
        self.early_stop = False

    def __call__(self, val_loss, model, current_epoch):
        """
        Call the early stopping instance to check if training should stop.

        Args:
            val_loss (float): The validation loss for the current epoch.
            model (torch.nn.Module): The current model.
            current_epoch (int): The current epoch number.

        Returns:
            bool: Whether to stop training.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.early_stop = True
                self.stopped_epoch = current_epoch
                if self.verbose:
                    print(f"Early stopping at epoch {self.stopped_epoch + 1}")  # Display as a 1-based index
        return self.early_stop

    def load_best_weights(self, model):
        """
        Load the best weights saved during early stopping.

        Args:
            model (torch.nn.Module): The model to load the weights into.

        Returns:
            None
        """
        if self.best_model is not None:
            model.load_state_dict(self.best_model)

def print_first_and_last_weights(model_name, weights, shape = True):
    """
    Prints the first three and last three values of the weights along with its shape.
    Converts torch.Tensor to a NumPy array if necessary.
    """
    # Convert weights to a numpy array if they are a torch.Tensor
    if isinstance(weights, torch.Tensor):
        weights_np = weights.detach().cpu().numpy()
    else:
        weights_np = weights  # assume it's already a numpy array or similar

    # Get first three and last three values formatted as strings
    first_three = " ".join(f"{x:.3f}" for x in weights_np[:3])
    last_three = " ".join(f"{x:.3f}" for x in weights_np[-3:])
    
    print(f"       ** The {model_name} weights are: [{first_three} ... {last_three}]")
    if shape:
        print(f"       ** The {model_name} weights shape is: {weights_np.shape}")
    
if __name__ == "__main__":
    # Test the functions
    print(sample_z_r(2, np.array([[0.5, 1], [1.5, 2], [2.5, 3]]), 3, 4, np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])))
    print(reshape_z_t(np.array([[0.5, 1], [1.5, 2], [2.5, 3], [1.4, 8], [9, 2.7], [0.6, 1.9]]), 2))
    print(sample_z_r_reshaped(np.array([[[0, 1, 0, 1], [0, 1, 0, 1]], [[0, 1, 0, 1], [0, 1, 0, 1]], [[0, 1, 0, 1], [0, 1, 0, 1]]]), np.array([[[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]], [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]], [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]])))

    # Example usage of calculate_performance
    true_vals = np.array([1.0, 2.0, 3.0])
    pred_vals = np.array([1.1, 2.1, 2.9])
    result = calculate_performance(true_vals, pred_vals, print_results=True)
    
    # Example usage of print_diagnostics
    train_perf = {'R2': 0.9, 'Adjusted_R2': 0.88, 'RMSE': 0.1, 'MAE': 0.05}
    valid_perf = {'R2': 0.85, 'Adjusted_R2': 0.83, 'RMSE': 0.15, 'MAE': 0.07}
    print_diagnostics(train_perf, valid_perf, metrics=["R2", "Adjusted_R2", "RMSE", "MAE"])
    
    # Example usage of check_early_stopping
    steps_without_improvement = check_early_stopping(3, [{'MAE': 0.6}, {'MAE': 0.7}, {'MAE': 0.8}], 'MAE', 5)
    print(steps_without_improvement)

    # Example usage of load_openml_dataset
    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)
    data, target_name = load_openml_dataset("iris", {"iris": 61}, cache_dir=cache_dir)
    print(data.head(), target_name)
    os.remove(os.path.join(cache_dir, "iris.csv"))
    os.rmdir(cache_dir)

    # Example usage of print_data_completeness
    df_train, df_valid, df_test = train_test_split(data, test_size=0.2, random_state=0)
    print_data_completeness(df_train, df_valid, df_test)

    # Example usage of flatten_3d_array
    arr_3d = np.random.rand(4, 3, 2)
    flattened_arr = flatten_3d_array(arr_3d)
    print(flattened_arr)

    # Example usage of format_model_metrics
    sample_data = {
        "dataset1": [
            {"Model": "ModelA", "MAE": 0.1, "Imputation": "Method1"},
            {"Model": "ModelA", "MAE": 0.2, "Imputation": "Method2"}
        ]
    }
    formatted_metrics = format_model_metrics(sample_data)
    print(formatted_metrics)
